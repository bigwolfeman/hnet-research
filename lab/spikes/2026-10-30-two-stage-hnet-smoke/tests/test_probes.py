"""Probe bookkeeping: cuts, word reference, F1, occurrence indexing."""
from types import SimpleNamespace

import numpy as np
import torch

from hnet import probes


def aux_with(cuts_bytes, anchor="start", n_row=1):
    pos = torch.tensor([cuts_bytes])
    return SimpleNamespace(counts=torch.tensor([len(cuts_bytes)]), pos=pos, anchor=anchor)


def test_word_cuts_and_prf():
    x = np.frombuffer(b"ab cd  e", dtype=np.uint8)
    ref = probes.word_cuts(x)
    assert list(np.flatnonzero(ref)) == [2, 3, 5, 7]       # ab|_|cd|__|e transitions
    perfect = ref.copy()
    assert probes.prf([perfect], [ref]) == (1.0, 1.0, 1.0)
    assert probes.prf([np.zeros_like(ref)], [ref])[2] == 0.0


def test_start_and_end_anchor_cuts():
    # start anchor at byte 4 -> cut 4 ; end anchor at byte 4 -> cut 5 ; byte 0 start and last end are not interior cuts
    s = probes.stage_cuts(aux_with([0, 4], "start"), 0, 10)
    e = probes.stage_cuts(aux_with([4, 9], "end"), 0, 10)
    assert list(np.flatnonzero(s)) == [4] and list(np.flatnonzero(e)) == [5]


def test_in_context_counts_by_occurrence(monkeypatch):
    text = "xx foo yy foo zz foo"
    paras = [dict(word="foo", text=text)]
    # occurrences at bytes [3,6) [10,13) [17,20); cut inside 1st twice, inside 2nd once, none in 3rd
    cuts = [4, 5, 11]
    monkeypatch.setattr(probes, "run_rows", lambda model, rows, device: [aux_with(cuts, "start")])
    m = probes.in_context(None, paras, torch.device("cpu"), max_occ=3)
    assert (m["probe/icl_s0_occ1"], m["probe/icl_s0_occ2"], m["probe/icl_s0_occ3"]) == (3.0, 2.0, 1.0)
