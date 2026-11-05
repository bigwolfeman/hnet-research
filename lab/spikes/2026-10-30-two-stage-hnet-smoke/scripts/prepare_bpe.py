"""Train a byte-level BPE tokenizer on our train text and tokenize every .bin file to a uint16 memmap.

    python scripts/prepare_bpe.py --data-dir <hnet-bytes> --out-dir <hnet-bytes>/bpe32k

Pseudocode:
    sample  = docs from train.bin files, edu 0.9 / go 0.1 by bytes, ~1 GB, taken from `segments` evenly spaced
              places in each file (whole docs only)
    tok     = BPE(vocab 32768, GPT-2 pretokenizer regex, byte-level alphabet, specials = [<|endoftext|>])
    for each .bin (docs end with byte 0):
        tokens = concat(tok.encode(doc) + [EOS] for doc in docs)     # EOS stands for the 0 byte
        write uint16 memmap
    check:   sum(token_nbytes[tokens]) == size of the .bin           # exact: every byte is covered once
    meta.json: bytes, docs, tokens, bytes_per_token per file + every setting + library versions
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnet.tokens import META, NBYTES, token_nbytes_table  # noqa: E402

EOS = "<|endoftext|>"
FILES = {  # output name -> (path under data dir, role)
    "edu_train": ("fineweb-edu/train.bin", "train"),
    "go_train": ("code/go_train.bin", "train"),
    "edu_val": ("fineweb-edu/val.bin", "val"),
    "go_val": ("code/go_val.bin", "val"),
    "sql_val": ("code/sql_val.bin", "val"),
}
MIX = {"edu_train": 0.9, "go_train": 0.1}  # tokenizer-training mix by bytes (same as the train mix)


def iter_docs(mm: np.ndarray, start: int = 0, stop: int | None = None, block: int = 1 << 27):
    """Yield the bytes of each doc (split on byte 0, separator dropped) in mm[start:stop]; start must be a doc start."""
    stop = len(mm) if stop is None else stop
    carry = b""
    for b0 in range(start, stop, block):
        buf = carry + mm[b0: min(b0 + block, stop)].tobytes()
        parts = buf.split(b"\x00")
        carry = parts.pop()
        yield from parts
    if carry:
        yield carry


def sample_docs(mm: np.ndarray, budget: int, segments: int):
    """Whole docs totalling ~budget bytes from `segments` evenly spaced places in the file."""
    seg_bytes = budget // segments
    for k in range(segments):
        s = int(k * (len(mm) - 2 * seg_bytes) / segments)
        if s > 0:  # move to the first doc start after s
            while mm[s - 1] != 0:
                s += 1
        got = 0
        for d in iter_docs(mm, s, min(len(mm), s + 4 * seg_bytes)):
            if got >= seg_bytes:
                break
            got += len(d) + 1
            yield d.decode("utf-8")


def train_tokenizer(texts, vocab_size: int, min_frequency: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)  # GPT-2 regex
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=[EOS],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts, trainer)
    return tok


def encode_file(tok, src: Path, dst: Path, nbytes: np.ndarray, eos_id: int, docs_per_batch: int) -> dict:
    mm = np.memmap(src, dtype=np.uint8, mode="r")
    n_tok = n_docs = covered = 0
    with open(dst, "wb") as out:
        batch = []

        def flush():
            nonlocal n_tok, n_docs, covered
            encs = tok.encode_batch([d.decode("utf-8") for d in batch])
            arr = np.empty(sum(len(e.ids) + 1 for e in encs), dtype=np.uint16)
            i = 0
            for e in encs:
                arr[i: i + len(e.ids)] = e.ids
                arr[i + len(e.ids)] = eos_id
                i += len(e.ids) + 1
            arr.tofile(out)
            covered += int(nbytes[arr].astype(np.int64).sum())
            n_tok += len(arr)
            n_docs += len(batch)
            batch.clear()

        for d in iter_docs(mm):
            batch.append(d)
            if len(batch) >= docs_per_batch:
                flush()
        if batch:
            flush()
    size = src.stat().st_size
    assert covered == size, f"{src}: tokens cover {covered} bytes, file has {size}"
    return dict(src=str(src), dst=str(dst), bytes=size, docs=n_docs, tokens=n_tok, bytes_per_token=size / n_tok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--sample-bytes", type=int, default=1_000_000_000)
    ap.add_argument("--segments", type=int, default=64)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--docs-per-batch", type=int, default=8192)
    ap.add_argument("--only", nargs="*", default=None, help="tokenize only these output names (tokenizer must exist)")
    a = ap.parse_args()
    import tokenizers

    a.out_dir.mkdir(parents=True, exist_ok=True)
    tok_path = a.out_dir / "tokenizer.json"
    t0 = time.time()
    timing = {}
    if tok_path.exists():
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(tok_path))
        print("reusing", tok_path, flush=True)
    else:
        mms = {n: np.memmap(a.data_dir / FILES[n][0], dtype=np.uint8, mode="r") for n in MIX}

        def texts():
            for n, w in MIX.items():
                yield from sample_docs(mms[n], int(a.sample_bytes * w), a.segments)

        tok = train_tokenizer(texts(), a.vocab_size, a.min_frequency)
        tok.save(str(tok_path))
        timing["train_tokenizer_s"] = time.time() - t0
        print(f"tokenizer trained in {timing['train_tokenizer_s']:.0f}s vocab {tok.get_vocab_size()}", flush=True)
    assert tok.get_vocab_size() == a.vocab_size, tok.get_vocab_size()
    eos_id = tok.token_to_id(EOS)
    nbytes = token_nbytes_table(tok.get_vocab(), a.vocab_size, {EOS})
    np.save(a.out_dir / NBYTES, nbytes)

    meta_path = a.out_dir / META
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"files": {}}
    for name, (rel, role) in FILES.items():
        if a.only is not None and name not in a.only:
            continue
        t1 = time.time()
        stats = encode_file(tok, a.data_dir / rel, a.out_dir / f"{name}.u16", nbytes, eos_id, a.docs_per_batch)
        stats.update(role=role, seconds=time.time() - t1)
        meta["files"][name] = stats
        print(name, json.dumps(stats), flush=True)
        meta.update(settings=vars(a) | {"data_dir": str(a.data_dir), "out_dir": str(a.out_dir)}, eos_id=eos_id,
                    eos_token=EOS, vocab_size=a.vocab_size, dtype="uint16", tokenizers=tokenizers.__version__,
                    pretokenizer="ByteLevel(add_prefix_space=False, use_regex=True) (GPT-2 regex)",
                    tokenizer_mix_by_bytes=MIX, timing=timing | {"total_s": time.time() - t0},
                    note="bytes_per_token = file bytes / tokens; the EOS token stands for the one 0 separator byte per doc")
        meta_path.write_text(json.dumps(meta, indent=1, default=str))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
