"""scripts/chunk_semantics.py: word positions, the space deletion and its join mask, the matched control."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chunk_semantics import join_vs_control, position_rates  # noqa: E402
from hnet.semantics import delete_spaces, word_pos  # noqa: E402


def b(s: str) -> np.ndarray:
    return np.frombuffer(s.encode(), dtype=np.uint8).copy()


def test_word_pos():
    assert word_pos(b("ab  cde f")).tolist() == [0, 1, -1, -1, 0, 1, 2, -1, 0]


def test_delete_spaces_marks_the_first_byte_of_the_second_word():
    x = b("the cat sat, on 1 mat")
    out, join = delete_spaces(x, np.random.default_rng(0), p=1.0)
    assert bytes(out).decode() == "thecatsat, on 1 mat"    # only letter-space-letter spaces go
    assert [bytes(out).decode()[i] for i in np.flatnonzero(join)] == ["c", "s"]


def test_join_vs_control_counts_matched_positions():
    x, join = delete_spaces(b("ab cd, " + "wxyz, " * 120), np.random.default_rng(0), p=1.0)
    assert bytes(x[:6]).decode() == "abcd, "
    cuts = np.zeros(len(x) + 1, dtype=bool)
    cuts[2] = True                                         # cut at the join 'c' (k = 2)
    cuts[np.flatnonzero(word_pos(x) == 2)[1:61]] = True    # half of the intact words cut at k = 2
    r = join_vs_control([x], [join], [cuts])
    assert r["n_joins"] == 1 and r["join_rate"] == 1.0
    assert abs(r["matched_control_rate"] - 0.5) < 1e-9


def test_position_rates():
    x = b("abcd efgh")
    cuts = np.zeros(len(x) + 1, dtype=bool)
    cuts[[0, 5, 6]] = True
    r = position_rates([x], [cuts])
    assert r["k0"] == 1.0 and r["k1"] == 0.5 and r["k3+"] == 0.0 and r["ws"] == 0.0
