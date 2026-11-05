"""Semantic chunking score: does the router rank true unit boundaries above the other positions?

Pseudocode:
    score_s(row)[c] = router p at the stage-s input position whose chunk would start at byte c
                      (start anchor: c = in_pos; end anchor: c = in_pos + 1); NaN where stage s has no input
                      position (stage 1 sees only stage-0 anchors)
    level = rows, candidates C (cut indices), labels R (a subset of C), strata (an int per candidate, or none),
            group (optional): a whitespace byte maps to the word start after it; that word start scores
            max p over its gap (the whitespace run and the first letter), so a router that starts its chunks at the
            space (" the") and one that starts them at the letter ("the") both count
    AUROC_s(level) = sum_k w_k * AUROC(score at R in stratum k, score at C - R in stratum k),  w_k = |R in k| / |R|,
                     over strata with >= min_n on both sides (ties count one half)
    0.5 = the score carries no information about the unit; 1 = every unit boundary outranks every other candidate.
    The AUROC does not depend on the compression ratio (it ranks p, it does not threshold it).
Levels (each one removes the cue of the level below it):
    word     edu rows; C = word starts after whitespace (scored over their gap) and the bytes inside words (k >= 1);
             R = the word starts; no strata
    hidden   edu rows with single spaces between ASCII letters deleted (p = 0.5); C = inside non-space runs (k >= 1);
             R = joins (the first byte of the second word); strata = k
    morph    edu rows; C = inside gold words (k >= 1); R = gold morph boundaries (SIGMORPHON 2022 English, word level,
             dev + test.gold, the words whose morphs spell the word: surface boundaries are exact); strata = k
    phrase   UD English-EWT dev + test documents; C = word starts after whitespace (scored over their gap; not the first
             word of a document); R = starts of multi-word dependency subtrees (constituents); no strata
    gotok    Go val rows; C = inside non-space runs (k >= 1), outside string and comment interiors;
             R = Go lexical token starts; strata = k
    k = byte offset inside the run, capped at K_CAP.
Experiment: lab/experiments/successes/2026-11-04-chunks-find-hidden-words.md
Note: .agents/notes/implemented/testing/2026-11-04-semantic-chunking-score.md
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .data import doc_starts, pad_rows
from .probes import _SPACE

K_CAP = 8
LETTER = np.zeros(256, dtype=bool)
LETTER[ord("A"):ord("Z") + 1] = True
LETTER[ord("a"):ord("z") + 1] = True
LEVELS = ("word", "hidden", "morph", "phrase", "gotok")


# ---------------------------------------------------------------- text helpers

def word_pos(x: np.ndarray) -> np.ndarray:
    """k[i] = offset of byte i in its non-whitespace run; -1 on whitespace."""
    sp = _SPACE[x]
    idx = np.arange(len(x))
    k = idx - np.maximum.accumulate(np.where(sp, idx, -1)) - 1
    k[sp] = -1
    return k


def delete_spaces(x: np.ndarray, rng: np.random.Generator, p: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Delete single spaces between two ASCII letters with probability p. Returns (edited row, bool join mask)."""
    cand = np.zeros(len(x), dtype=bool)
    cand[1:-1] = (x[1:-1] == 0x20) & LETTER[x[:-2]] & LETTER[x[2:]]
    drop = cand & (rng.random(len(x)) < p)
    out = x[~drop]
    after = np.flatnonzero(drop) + 1          # the byte after a dropped space, as an index of the old row
    join = np.zeros(len(out), dtype=bool)
    join[after - np.cumsum(drop)[after]] = True
    return out, join


# ---------------------------------------------------------------- statistics

def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score at a positive > score at a negative), ties one half (Mann-Whitney U / (n_pos n_neg))."""
    x = np.concatenate([pos, neg])
    order = np.argsort(x, kind="mergesort")
    _, inv, cnt = np.unique(x[order], return_inverse=True, return_counts=True)
    avg = np.cumsum(cnt) - (cnt - 1) / 2.0                # 1-based mean rank of each tie group
    ranks = np.empty(len(x))
    ranks[order] = avg[inv]
    n1 = len(pos)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * len(neg)))


def stratified_auroc(score: np.ndarray, label: np.ndarray, strata: np.ndarray | None, min_n: int = 50) -> dict:
    """AUROC within each stratum, weighted by its positives. Strata with < min_n on either side are left out."""
    strata = np.zeros(len(score), dtype=np.int64) if strata is None else strata
    total, num, n_pos, n_neg = 0, 0.0, 0, 0
    for k in np.unique(strata):
        m = strata == k
        pos, neg = score[m & label], score[m & ~label]
        if len(pos) < min_n or len(neg) < min_n:
            continue
        num += len(pos) * auroc(pos, neg)
        total += len(pos)
        n_pos += len(pos)
        n_neg += len(neg)
    return {"auroc": num / total if total else float("nan"), "n_pos": n_pos, "n_neg": n_neg,
            "pos_kept": n_pos / max(int(label.sum()), 1)}


# ---------------------------------------------------------------- levels

@dataclass
class Level:
    name: str
    rows: list[np.ndarray]                  # byte rows the model reads
    cand: list[np.ndarray]                  # bool (L+1,) candidate cut indices
    label: list[np.ndarray]                 # bool (L+1,) unit boundaries, a subset of cand
    strata: list[np.ndarray] | None         # int (L+1,) per cut index, or None
    group: list[np.ndarray] | None = None   # int (L+1,): cut index -> the candidate it scores for (-1: itself only)


def _strata(x: np.ndarray) -> np.ndarray:
    s = np.full(len(x) + 1, -1, dtype=np.int64)
    s[:len(x)] = np.minimum(word_pos(x), K_CAP)
    return s


def gap_group(x: np.ndarray) -> np.ndarray:
    """(L+1,) int: each whitespace byte -> the index of the next non-whitespace byte (the word start), else -1."""
    L = len(x)
    sp = _SPACE[x]
    idx = np.arange(L)
    nxt = np.minimum.accumulate(np.where(~sp, idx, L)[::-1])[::-1]     # next non-whitespace index at or after i
    g = np.full(L + 1, -1, dtype=np.int64)
    m = sp & (nxt < L)
    g[:L][m] = nxt[m]
    return g


def word_starts_after_space(x: np.ndarray) -> np.ndarray:
    """(L+1,) bool: word starts (k = 0) with whitespace before them."""
    w = np.zeros(len(x) + 1, dtype=bool)
    w[1:len(x)] = (word_pos(x)[1:] == 0) & _SPACE[x[:-1]]
    return w


def word_level(rows: list[np.ndarray]) -> Level:
    cand, label = [], []
    for x in rows:
        w = word_starts_after_space(x)
        c = w.copy()
        c[:len(x)] |= word_pos(x) >= 1
        cand.append(c); label.append(w)
    return Level("word", rows, cand, label, None, [gap_group(x) for x in rows])


def hidden_level(rows: list[np.ndarray], seed: int = 0) -> Level:
    rng = np.random.default_rng(seed)
    out, cand, label = [], [], []
    for x in rows:
        xe, join = delete_spaces(x, rng)
        c = np.zeros(len(xe) + 1, dtype=bool)
        c[:len(xe)] = word_pos(xe) >= 1
        j = np.zeros(len(xe) + 1, dtype=bool)
        j[:len(xe)] = join
        out.append(xe); cand.append(c); label.append(j & c)
    return Level("hidden", out, cand, label, [_strata(x) for x in out])


def load_morphs(gold_dir: Path) -> dict[bytes, list[int]]:
    """word -> morph boundary offsets, for gold words whose morphs spell the word (surface-concatenative)."""
    d = Path(gold_dir) / "sigmorphon2022"
    out: dict[bytes, list[int]] = {}
    for f in ("eng.word.dev.tsv", "eng.word.test.gold.tsv"):
        for line in (d / f).read_text(encoding="utf-8").splitlines():
            word, segs = line.split("\t")[:2]
            morphs = segs.split(" @@")
            if not (word.isascii() and word.isalpha()) or "".join(morphs) != word or word in out:
                continue
            out[word.encode()] = np.cumsum([len(m) for m in morphs])[:-1].tolist()
    if not out:
        raise ValueError(f"no surface-concatenative gold words in {d}")
    return out


def morph_level(rows: list[np.ndarray], morphs: dict[bytes, list[int]]) -> Level:
    cand, label, strata = [], [], []
    for x in rows:
        c = np.zeros(len(x) + 1, dtype=bool)
        r = np.zeros(len(x) + 1, dtype=bool)
        st = np.full(len(x) + 1, -1, dtype=np.int64)
        k = word_pos(x)
        for a in np.flatnonzero(k == 0):
            b = a
            while b < len(x) and k[b] >= 0:
                b += 1
            while a < b and not LETTER[x[a]]:
                a += 1
            while b > a and not LETTER[x[b - 1]]:
                b -= 1
            core = bytes(x[a:b])
            if b - a < 2 or not LETTER[x[a:b]].all():
                continue
            cuts = morphs.get(core.lower())
            if cuts is None:
                continue
            c[a + 1:b] = True
            st[a + 1:b] = np.minimum(np.arange(1, b - a), K_CAP)
            r[[a + j for j in cuts]] = True
        cand.append(c); label.append(r); strata.append(st)
    return Level("morph", rows, cand, label, strata)


def _conllu_docs(path: Path):
    """Yield documents as lists of sentences; a sentence is (text, [(id, form, head)])."""
    doc, sent, text = [], [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# newdoc") and doc:
            yield doc
            doc = []
        elif line.startswith("# text = "):
            text = line[len("# text = "):]
        elif not line.strip():
            if sent and text is not None:
                doc.append((text, sent))
            sent, text = [], None
        elif not line.startswith("#"):
            cols = line.split("\t")
            if cols[0].isdigit():
                sent.append((int(cols[0]), cols[1], int(cols[6])))
    if sent and text is not None:
        doc.append((text, sent))
    if doc:
        yield doc


def _constituent_starts(sent: list[tuple[int, str, int]]) -> set[int]:
    """Ids of the first token of every multi-word dependency subtree."""
    kids: dict[int, list[int]] = {}
    for i, _, h in sent:
        kids.setdefault(h, []).append(i)
    lo, hi = {}, {}

    def span(i: int) -> tuple[int, int]:
        a = b = i
        for c in kids.get(i, []):
            ca, cb = span(c)
            a, b = min(a, ca), max(b, cb)
        lo[i], hi[i] = a, b
        return a, b

    for root in kids.get(0, []):
        span(root)
    return {lo[i] for i in lo if hi[i] > lo[i]}


def phrase_level(gold_dir: Path, max_len: int = 2048, min_len: int = 256, max_rows: int | None = None) -> Level:
    rows, cand, label = [], [], []
    d = Path(gold_dir) / "ud-ewt"
    for f in ("en_ewt-ud-dev.conllu", "en_ewt-ud-test.conllu"):
        for doc in _conllu_docs(d / f):
            text, starts, is_const = "", [], []
            for stext, sent in doc:
                base = len(text) + (1 if text else 0)
                cur, offs = 0, []
                for _, form, _ in sent:
                    j = stext.find(form, cur)
                    if j < 0 or stext[cur:j].strip():
                        offs = None
                        break
                    offs.append(j)
                    cur = j + len(form)
                text += (" " if text else "") + stext
                if offs is None:
                    continue
                const = _constituent_starts(sent)
                for (i, _, _), o in zip(sent, offs):
                    starts.append(base + o)
                    is_const.append(i in const)
            x = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)[:max_len].copy()
            if len(x) < min_len:
                continue
            c = np.zeros(len(x) + 1, dtype=bool)
            r = np.zeros(len(x) + 1, dtype=bool)
            for ch, k in zip(starts, is_const):
                b = len(text[:ch].encode("utf-8"))
                if 0 < b < len(x) and _SPACE[x[b - 1]]:
                    c[b] = True
                    r[b] = k
            rows.append(x); cand.append(c); label.append(r)
            if max_rows and len(rows) >= max_rows:
                break
        if max_rows and len(rows) >= max_rows:
            break
    return Level("phrase", rows, cand, label, None, [gap_group(x) for x in rows])


_GO_TOKEN = re.compile(
    rb'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|`[^`]*`|\'(?:\\.|[^\'\\\n])*\'|[A-Za-z_][A-Za-z0-9_]*|\d[\w.]*'
    rb'|<<=|>>=|&\^=|\.\.\.|&&|\|\||<-|\+\+|--|==|!=|<=|>=|:=|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<|>>|&\^|\S', re.S)


def gotok_level(rows: list[np.ndarray]) -> Level:
    cand, label, strata = [], [], []
    for x in rows:
        k = word_pos(x)
        c = np.zeros(len(x) + 1, dtype=bool)
        r = np.zeros(len(x) + 1, dtype=bool)
        c[:len(x)] = k >= 1
        for m in _GO_TOKEN.finditer(bytes(x)):
            s, e = m.span()
            r[s] = True
            if m.group()[:1] in (b"/", b'"', b"`", b"'") and e - s > 1:
                c[s + 1:e] = False                # string and comment interiors are not code structure
        cand.append(c); label.append(r & c); strata.append(_strata(x))
    return Level("gotok", rows, cand, label, strata)


def build_levels(text_rows: list[np.ndarray], go_rows: list[np.ndarray], gold_dir: Path | None,
                 morph_rows: list[np.ndarray] | None = None, phrase_rows: int = 64) -> list[Level]:
    """The five levels; morph and phrase need the gold data (gold_dir None: word, hidden, gotok only)."""
    levels = [word_level(text_rows), hidden_level(text_rows), gotok_level(go_rows)]
    if gold_dir is not None:
        levels += [morph_level(morph_rows if morph_rows is not None else text_rows, load_morphs(gold_dir)),
                   phrase_level(gold_dir, max_rows=phrase_rows)]
    return levels


def probe_levels(data_dir: Path, sc) -> list[Level]:
    """Levels for the training probe, from configs/data/bytes.yaml `semantic` (rows from fixed doc-start seeds)."""
    text = doc_starts(Path(data_dir) / sc.text_file, sc.text_rows, sc.row_len, seed=3)
    go = doc_starts(Path(data_dir) / sc.go_file, sc.go_rows, sc.row_len, seed=3)
    morph = doc_starts(Path(data_dir) / sc.text_file, sc.morph_rows, sc.row_len, seed=4)
    return build_levels(text, go, Path(sc.gold_dir), morph_rows=morph, phrase_rows=sc.phrase_rows)


# ---------------------------------------------------------------- model scores

def cut_scores(aux, row: int, length: int) -> list[np.ndarray]:
    """Per stage: (length+1,) router p by cut index, NaN where the stage has no input position."""
    out = []
    for a in aux:
        if a.p is None or a.in_pos is None:
            raise ValueError("semantic score needs StageAux.p and StageAux.in_pos")
        n = int(a.lengths[row])
        c = a.in_pos[row, :n].cpu().numpy() + (1 if a.anchor == "end" else 0)
        p = a.p[row, :n].float().cpu().numpy()
        s = np.full(length + 1, np.nan)
        ok = (c > 0) & (c < length)
        s[c[ok]] = p[ok]
        out.append(s)
    return out


def group_max(score: np.ndarray, group: np.ndarray) -> np.ndarray:
    """score[r] <- max over score[r] and every score[i] with group[i] = r (NaN-ignoring)."""
    out = score.copy()
    m = group >= 0
    np.fmax.at(out, group[m], score[m])
    return out


@torch.no_grad()
def level_scores(model, level: Level, device, batch: int = 16) -> dict:
    per_stage: dict[int, list[tuple]] = {}
    for b in range(0, len(level.rows), batch):
        rows = level.rows[b:b + batch]
        x, lengths = pad_rows(rows, model.multiple, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            aux = model(x, lengths).aux
        for i, r in enumerate(rows):
            for s, sc in enumerate(cut_scores(aux, i, len(r))):
                if level.group is not None:
                    sc = group_max(sc, level.group[b + i])
                per_stage.setdefault(s, []).append((sc, level.cand[b + i], level.label[b + i],
                                                    None if level.strata is None else level.strata[b + i]))
    out = {}
    for s, items in per_stage.items():
        score = np.concatenate([sc[c] for sc, c, _, _ in items])
        label = np.concatenate([lb[c] for _, c, lb, _ in items])
        strata = None if level.strata is None else np.concatenate([st[c] for _, c, _, st in items])
        ok = ~np.isnan(score)
        res = stratified_auroc(score[ok], label[ok], None if strata is None else strata[ok])
        res["seen_frac"] = float(ok.mean()) if len(ok) else float("nan")   # candidates the stage scores at all
        out[s] = res
    return out


def semantic_report(model, levels: list[Level], device) -> dict:
    """Flat metrics: sem/<level>_s<stage>_auroc, plus counts."""
    was = model.training
    model.eval()
    m = {}
    for lv in levels:
        for s, r in level_scores(model, lv, device).items():
            m[f"sem/{lv.name}_s{s}_auroc"] = r["auroc"]
            m[f"sem/{lv.name}_s{s}_n_pos"] = r["n_pos"]
            m[f"sem/{lv.name}_s{s}_seen_frac"] = r["seen_frac"]
    model.train(was)
    return m
