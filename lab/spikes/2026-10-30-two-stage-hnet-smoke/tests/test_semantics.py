"""hnet/semantics.py: AUROC, the reference levels, and the router-score map."""
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from hnet import semantics as sem
from hnet.model import HNetLM
from tests.conftest import make_cfg, randomize


def b(s: str) -> np.ndarray:
    return np.frombuffer(s.encode(), dtype=np.uint8).copy()


def test_auroc_anchors():
    assert sem.auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert sem.auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 0.0
    assert sem.auroc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5          # ties count one half
    rng = np.random.default_rng(0)
    assert abs(sem.auroc(rng.random(4000), rng.random(4000)) - 0.5) < 0.02


def test_stratified_auroc_removes_a_position_effect():
    # the score depends only on the stratum; the label rate does too: pooled AUROC is high, stratified is 0.5
    strata = np.repeat([0, 1], 1000)
    score = strata.astype(float)
    label = np.concatenate([np.arange(1000) < 100, np.arange(1000) < 900])
    assert sem.stratified_auroc(score, label, None)["auroc"] > 0.8
    assert sem.stratified_auroc(score, label, strata)["auroc"] == 0.5


def test_hidden_level_candidates_are_inside_runs():
    lv = sem.hidden_level([b("the cat sat on the mat " * 20)], seed=0)
    x, c, r = lv.rows[0], lv.cand[0], lv.label[0]
    assert r.sum() > 0 and not (r & ~c).any()
    assert (sem.word_pos(x)[c[:len(x)]] >= 1).all()


def test_morph_level(tmp_path: Path):
    d = tmp_path / "sigmorphon2022"; d.mkdir()
    (d / "eng.word.dev.tsv").write_text("undiscounted\tun @@discount @@ed\t110\nsubsidised\tsubside @@y @@ise @@ed\t110\n")
    (d / "eng.word.test.gold.tsv").write_text("cats\tcat @@s\t100\n")
    morphs = sem.load_morphs(tmp_path)
    assert morphs == {b"undiscounted": [2, 10], b"cats": [3]}                   # "subsidised" is not concatenative
    x = b("The (Undiscounted) cats.")
    lv = sem.morph_level([x], morphs)
    labels = [bytes(x[i:i + 3]).decode() for i in np.flatnonzero(lv.label[0])]
    assert labels == ["dis", "ed)", "s."]
    assert lv.cand[0].sum() == 11 + 3                                           # inside "Undiscounted" and "cats"


def test_constituent_starts():
    # "the big cat sat": the(1)->cat, big(2)->cat, cat(3)->sat, sat(4) root
    sent = [(1, "the", 3), (2, "big", 3), (3, "cat", 4), (4, "sat", 0)]
    assert sem._constituent_starts(sent) == {1}                                 # [the big cat], [the big cat sat]


def test_gotok_level():
    x = b('x := fmt.Println(a[i]) // say hi')
    lv = sem.gotok_level([x])
    s = bytes(x).decode()
    assert {s[i] for i in np.flatnonzero(lv.label[0])} >= {".", "P", "(", "[", "]", ")"}
    assert not lv.cand[0][s.index("say") + 1]                                   # comment interior is out


def test_cut_scores_follow_the_router_and_stage1_sees_only_stage0_anchors():
    torch.manual_seed(0)
    c = make_cfg("lookahead", 2)
    model = randomize(HNetLM(c), seed=1).eval()
    x = torch.randint(0, 256, (2, 128))
    with torch.no_grad():
        aux = model(x).aux
    s0, s1 = sem.cut_scores(aux, 0, 128)
    assert np.isnan(s0[0]) and not np.isnan(s0[1:128]).any()                    # stage 0 scores every byte
    cut0 = np.zeros(129, dtype=bool)
    from hnet.probes import stage_cuts
    cut0 = stage_cuts(aux[0], 0, 128)
    assert np.array_equal(s0[1:128] >= 0.5, cut0[1:128])                        # p >= 0.5 is a cut
    assert set(np.flatnonzero(~np.isnan(s1))) <= set(np.flatnonzero(cut0))      # stage 1 only at stage-0 cuts


def test_word_gap_counts_a_cut_at_the_space_or_at_the_letter():
    x = b("the cat sat")
    lv = sem.word_level([x])
    assert [int(i) for i in np.flatnonzero(lv.label[0])] == [4, 8]
    g = lv.group[0]
    assert g[3] == 4 and g[7] == 8 and g[4] == -1
    at_space = np.full(12, 0.1); at_space[[3, 7]] = 0.9                       # " cat" chunks
    at_letter = np.full(12, 0.1); at_letter[[4, 8]] = 0.9                     # "cat" chunks
    for sc in (at_space, at_letter):
        assert np.allclose(sem.group_max(sc, g)[[4, 8]], 0.9)


def test_group_max_ignores_nan():
    sc = np.array([np.nan, np.nan, 0.3, np.nan])
    assert sem.group_max(sc, np.array([-1, 2, -1, -1]))[2] == 0.3
    assert np.isnan(sem.group_max(np.full(3, np.nan), np.array([1, -1, -1]))[1])
