"""Byte data: one-time memmap builder, weighted multi-source train loader, single-file eval loader.

Pseudocode:
    prepare(parquet files, [(file, bytes)...], column):
        for doc in docs: bytes = utf8(doc.text) with NUL removed, minus a leading header if strip_header matches;
                         write bytes + b"\\0"
        targets fill in order; the stream moves to the next target when one is full (disjoint docs)
    MixLoader.next():                                  # train
        src = rng.choice(n_sources, size=B, p=weights)  # one source per batch row
        o   = rng.integers(0, N[src] - L - 1)           # random offset inside that source's file
        x, y = mm[src][o : o + L], mm[src][o + 1 : o + L + 1]
    ByteLoader: the same without the source draw (one file; eval batches and probe slices).
    MixLoader.state_dict(): the rng state right after the last batch that next() returned (the prefetch thread
        runs ahead, so every batch carries the state taken after its own draw). MixLoader(..., state=sd) continues
        the exact batch stream (train.py resume).
    PairLoader.next(): P random record indices i; row r of view a = a.bin record i cut to width_a, zero right-pad,
        and the same for view b (coarse JEPA, hnet/jepa.py). Its own rng, so the MixLoader stream is unchanged.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import torch

SEP = 0  # document separator byte


# StarCoderData prepends synthetic metadata ("<reponame>r<filename>f<gh_stars>n\n") to about half the docs.
STARCODER_HEADER = rb"^(?:<(?:reponame|filename|gh_stars)>[^<\n]*)+\n?"


def prepare(parquet_glob: str, out_dir: str, targets: list[tuple[str, int]], column: str = "text",
            strip_header: bytes | None = None) -> dict:
    """Encode parquet docs into `targets` = [(file name, byte budget)], filled in order. Returns per-file stats."""
    import pyarrow.parquet as pq

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(parquet_glob))
    if not files:
        raise FileNotFoundError(f"no parquet files match {parquet_glob!r}")
    header = re.compile(strip_header) if strip_header is not None else None
    ti, written, n_docs = 0, 0, 0
    fh = open(out / targets[0][0], "wb")
    stats = {}
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=1024, columns=[column]):
            for text in batch.column(column).to_pylist():
                b = text.encode("utf-8").replace(b"\x00", b"")
                if header is not None:
                    b = header.sub(b"", b, count=1)
                b += bytes([SEP])
                fh.write(b)
                written += len(b)
                n_docs += 1
                if written >= targets[ti][1]:
                    fh.close()
                    stats[targets[ti][0]] = dict(bytes=written, docs=n_docs)
                    ti += 1
                    if ti == len(targets):
                        return stats
                    fh, written, n_docs = open(out / targets[ti][0], "wb"), 0, 0
    fh.close()
    raise RuntimeError(f"parquet files exhausted before reaching {targets[ti]}; wrote {written}")


class Prefetcher:
    """Background thread that runs `produce() -> ((B, L+1) ndarray, meta)` in order and keeps `depth` batches on the
    device. get() returns (tensor, meta); meta is any host object made with the batch (MixLoader: its rng state).

    One producer thread means the RNG stream and so the batch order are identical to calling `produce()` inline.
    The host array is copied into a reusable pinned buffer, moved to the device on a side stream, and the thread waits
    for that copy before reuse, so the consumer never waits on disk or on H2D.
    """

    def __init__(self, produce, shape, device, depth: int, dtype: torch.dtype = torch.uint8):
        import queue
        import threading

        self.produce, self.device = produce, torch.device(device)
        self.cuda = self.device.type == "cuda"
        self.pinned = torch.empty(shape, dtype=dtype, pin_memory=self.cuda)
        self.stream = torch.cuda.Stream(self.device) if self.cuda else None
        self.q: "queue.Queue" = queue.Queue(maxsize=max(depth, 1))
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="byte-prefetch")
        self.thread.start()

    def _run(self):
        try:
            while not self._stop.is_set():
                arr, meta = self.produce()
                self.pinned.numpy()[:] = arr
                if self.cuda:
                    with torch.cuda.stream(self.stream):
                        dev = self.pinned.to(self.device, non_blocking=True)
                        done = torch.cuda.Event()
                        done.record(self.stream)
                    done.synchronize()
                else:
                    dev = self.pinned.clone()
                while not self._stop.is_set():
                    try:
                        self.q.put((dev, meta), timeout=0.2)
                        break
                    except Exception:  # queue.Full
                        continue
        except BaseException as e:  # surface producer errors in next()
            self.q.put(e)

    def get(self):
        item = self.q.get()
        if isinstance(item, BaseException):
            raise item
        t, meta = item
        if self.cuda:
            t.record_stream(torch.cuda.current_stream(self.device))
        return t, meta

    def close(self):
        self._stop.set()


def _split_xy(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    t = t.long()
    return t[:, :-1], t[:, 1:]


def row_probs(byte_weights, bytes_per_token) -> np.ndarray:
    """Per-row source probabilities that give byte fractions `byte_weights` (normalised) when rows are token windows.

    A row of source i holds about bytes_per_token[i] * seq_len bytes, so p_i is proportional to w_i / bpt_i.
    """
    w = np.asarray(byte_weights, dtype=np.float64)
    bpt = np.asarray(bytes_per_token, dtype=np.float64)
    assert (bpt > 0).all(), f"bad bytes_per_token {bpt}"
    p = (w / w.sum()) / bpt
    return p / p.sum()


class MixLoader:
    """Train loader over weighted sources: per batch row pick a source by weight, then a random offset in its file.

    sources: [{name, file, weight}] (absolute or already-resolved paths). Weights are normalised.
    `last_sources` holds the source index of every row of the most recent batch (for tests and logging).

    Token mode (dtype=np.uint16): a row is `seq_len` tokens, not bytes. Source weights stay BYTE weights, so each
    source also needs `bytes_per_token`, and a row is drawn with probability proportional to weight / bytes_per_token.
    Then the expected bytes of every source in a batch follow the weights (see `row_probs`). Rows reach the device as
    int32 (torch has poor uint16 support). With dtype=np.uint8 and no `bytes_per_token` the code path is the byte path.
    """

    def __init__(self, sources, seq_len: int, batch_size: int, seed: int, device, prefetch: int = 0,
                 dtype=np.uint8, state: dict | None = None):
        assert len(sources) > 0, "need at least one train source"
        self.names = [s["name"] for s in sources]
        w = np.array([float(s["weight"]) for s in sources], dtype=np.float64)
        assert (w >= 0).all() and w.sum() > 0, f"bad source weights {w}"
        self.token_mode = np.dtype(dtype) != np.uint8
        if self.token_mode:
            self.p = row_probs(w, [float(s["bytes_per_token"]) for s in sources])
        else:
            self.p = w / w.sum()
        self.mms = [np.memmap(s["file"], dtype=dtype, mode="r") for s in sources]
        for n, mm in zip(self.names, self.mms):
            assert len(mm) > seq_len + 2, f"source {n!r} too small"
        self.L, self.B, self.device = seq_len, batch_size, device
        self.rng = np.random.default_rng(seed)
        if state is not None:
            self.rng.bit_generator.state = state["rng"]
        self._consumed = self.rng.bit_generator.state   # rng state after the last batch returned by next()
        self.last_sources = np.zeros(batch_size, dtype=np.int64)
        pf_dtype = torch.int32 if self.token_mode else torch.uint8
        self.pf = (Prefetcher(self._read, (batch_size, seq_len + 1), device, prefetch, pf_dtype)
                   if prefetch > 0 else None)

    def draw(self) -> list[tuple[int, int]]:
        """[(source index, byte offset)] for one batch."""
        src = self.rng.choice(len(self.mms), size=self.B, p=self.p)
        self.last_sources = src
        return [(int(i), int(self.rng.integers(0, len(self.mms[i]) - self.L - 1))) for i in src]

    def _read(self) -> tuple[np.ndarray, dict]:
        arr = np.stack([self.mms[i][o: o + self.L + 1] for i, o in self.draw()])
        return (arr.astype(np.int32) if self.token_mode else arr), self.rng.bit_generator.state

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        """With prefetch > 0 the batch comes from the background thread (same order; `last_sources` runs ahead)."""
        if self.pf is not None:
            t, self._consumed = self.pf.get()
            return _split_xy(t)
        arr, self._consumed = self._read()
        return _split_xy(torch.from_numpy(arr).to(self.device, non_blocking=True))

    def state_dict(self) -> dict:
        return {"rng": self._consumed}


class ByteLoader:
    def __init__(self, path: str, seq_len: int, batch_size: int, seed: int, device, prefetch: int = 0):
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        self.L, self.B, self.device = seq_len, batch_size, device
        self.rng = np.random.default_rng(seed)
        assert len(self.mm) > seq_len + 2, f"{path} too small"
        self.pf = Prefetcher(self._read_next, (batch_size, seq_len + 1), device, prefetch) if prefetch > 0 else None

    def _batch(self, offs) -> tuple[torch.Tensor, torch.Tensor]:
        arr = np.stack([self.mm[o: o + self.L + 1] for o in offs])
        t = torch.from_numpy(arr).to(self.device, non_blocking=True).long()
        return t[:, :-1], t[:, 1:]

    def _read_next(self) -> tuple[np.ndarray, None]:
        offs = self.rng.integers(0, len(self.mm) - self.L - 1, self.B)
        return np.stack([self.mm[o: o + self.L + 1] for o in offs]), None

    def next(self):
        if self.pf is not None:
            return _split_xy(self.pf.get()[0])
        return _split_xy(torch.from_numpy(self._read_next()[0]).to(self.device, non_blocking=True))

    def fixed(self, n_batches: int, seed: int = 1234):
        """Deterministic held-out batches (same every call)."""
        rng = np.random.default_rng(seed)
        return [self._batch(rng.integers(0, len(self.mm) - self.L - 1, self.B)) for _ in range(n_batches)]

    def slice(self, start: int, length: int) -> np.ndarray:
        return np.asarray(self.mm[start: start + length])


class PairLoader:
    """Paired views for the coarse JEPA (hnet/jepa.py): record i of `a.bin` and record i of `b.bin` are two views of
    one thing (scripts/goviews: a Go function's code and its doc comment). Records are separated by SEP.

    next() -> (xa (P, len_a), la (P,), xb (P, len_b), lb (P,)): P random pairs, each view cut to its width and
    right-padded with zeros; la / lb are the kept lengths. Its own rng (seed), so the LM stream is untouched.
    state_dict() / state=: the rng state after the last batch (train.py resume).
    """

    def __init__(self, dir_path: str, names: tuple[str, str], widths: tuple[int, int], pairs: int, seed: int, device,
                 state: dict | None = None):
        self.mm = [np.memmap(Path(dir_path) / f"{n}.bin", dtype=np.uint8, mode="r") for n in names]
        self.bounds = []
        for n, mm in zip(names, self.mm):
            ends = np.flatnonzero(np.asarray(mm) == SEP)
            self.bounds.append((np.concatenate([[0], ends[:-1] + 1]), ends))
        assert len(self.bounds[0][0]) == len(self.bounds[1][0]) > 0, f"{names}: record counts differ or are zero"
        self.n, self.widths, self.P, self.device = len(self.bounds[0][0]), widths, pairs, device
        self.rng = np.random.default_rng(seed)
        if state is not None:
            self.rng.bit_generator.state = state["rng"]

    def next(self):
        idx = self.rng.integers(0, self.n, self.P)
        out = []
        for (starts, ends), mm, w in zip(self.bounds, self.mm, self.widths):
            x = np.zeros((self.P, w), dtype=np.int64)
            lens = np.zeros(self.P, dtype=np.int64)
            for r, i in enumerate(idx):
                rec = mm[starts[i]: min(ends[i], starts[i] + w)]
                x[r, : len(rec)], lens[r] = rec, len(rec)
            out += [torch.from_numpy(x).to(self.device), torch.from_numpy(lens).to(self.device)]
        return tuple(out)

    def state_dict(self) -> dict:
        return {"rng": self.rng.bit_generator.state}


def pad_rows(rows: list[np.ndarray], multiple: int, device):
    """Right-pad byte rows with zeros to a common length (multiple of `multiple`). Returns (x, lengths)."""
    n = max(len(r) for r in rows)
    width = -(-n // multiple) * multiple
    x = np.zeros((len(rows), width), dtype=np.int64)
    for i, r in enumerate(rows):
        x[i, : len(r)] = r
    return torch.from_numpy(x).to(device), torch.tensor([len(r) for r in rows], device=device)


def doc_starts(bin_path, n: int, length: int, seed: int) -> list[np.ndarray]:
    """n byte rows that each start at a document start (after a 0 separator) and contain no separator."""
    mm = np.memmap(bin_path, dtype=np.uint8, mode="r")
    rng, rows = np.random.default_rng(seed), []
    while len(rows) < n:
        o = int(rng.integers(0, len(mm) - 4 * length))
        sep = np.flatnonzero(mm[o:o + 4 * length] == 0)
        if len(sep) == 0:
            continue
        s = o + int(sep[0]) + 1
        row = np.array(mm[s:s + length])
        if (row == 0).any():
            continue
        rows.append(row)
    return rows
