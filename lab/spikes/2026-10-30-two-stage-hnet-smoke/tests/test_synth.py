"""Synthetic recall language (hnet/synth.py): the oracle is a normalised distribution, and the generator obeys its spec."""
import numpy as np
import pytest

from hnet.data import SEP
from hnet.synth import SPACE, SynthSpec, generate, make_lexicon, oracle_bits, read_lexicon, write_lexicon

TINY = dict(n_lex=12, min_len=1, max_len=3, p_end=0.2)


@pytest.mark.parametrize("doc_vocab", [3, None])
def test_oracle_sums_to_one(doc_vocab):
    spec = SynthSpec(doc_vocab=doc_vocab, **TINY)
    rng = np.random.default_rng(0)
    lex = make_lexicon(spec, rng)
    stream = np.concatenate([[SEP], generate(spec, lex, 300, rng)]).astype(np.uint8)
    alphabet = sorted({c for w in lex for c in w} | {SPACE, SEP})
    for t in range(1, 120):
        total = 0.0
        for c in alphabet:
            probe = np.concatenate([stream[:t], [c]]).astype(np.uint8)
            b = oracle_bits(spec, lex, probe)[-1]
            total += 0.0 if np.isinf(b) else 2.0 ** -b
        assert abs(total - 1) < 1e-9, (t, total)


def test_generator_spec():
    spec = SynthSpec(n_lex=64, doc_vocab=5, p_end=0.05)
    rng = np.random.default_rng(1)
    lex = make_lexicon(spec, rng)
    assert len(set(lex)) == 64 and all(3 <= len(w) <= 8 for w in lex)
    data = generate(spec, lex, 50_000, rng).tobytes()
    docs = data.split(bytes([SEP]))[:-1]
    assert len(docs) > 20
    for d in docs:
        words = d.split(b" ")
        assert set(words) <= set(lex) and len(set(words)) <= 5


def test_oracle_is_finite_on_generated_data():
    spec = SynthSpec(n_lex=256, doc_vocab=8, p_end=0.02)
    rng = np.random.default_rng(2)
    lex = make_lexicon(spec, rng)
    bits = oracle_bits(spec, lex, generate(spec, lex, 20_000, rng))
    assert np.isfinite(bits[~np.isnan(bits)]).all()


def test_reach_limits_the_oracle():
    spec = SynthSpec(n_lex=40, min_len=2, max_len=4, doc_vocab=4, p_end=0.02)
    rng = np.random.default_rng(3)
    lex = make_lexicon(spec, rng)
    stream = np.concatenate([[SEP], generate(spec, lex, 20_000, rng)]).astype(np.uint8)
    no_mem = SynthSpec(spec.n_lex, spec.min_len, spec.max_len, None, spec.p_end)
    full, zero = oracle_bits(spec, lex, stream), oracle_bits(spec, lex, stream, reach=0)
    np.testing.assert_allclose(zero, oracle_bits(no_mem, lex, stream), equal_nan=True)
    np.testing.assert_allclose(oracle_bits(spec, lex, stream, reach=len(stream)), full, equal_nan=True)
    means = [np.nanmean(oracle_bits(spec, lex, stream, reach=r)) for r in (0, 6, 20, 80)]
    assert all(a > b for a, b in zip(means, means[1:])) and means[-1] > np.nanmean(full), means


def test_lexicon_round_trip(tmp_path):
    spec = SynthSpec(n_lex=50)
    lex = make_lexicon(spec, np.random.default_rng(0))
    write_lexicon(tmp_path / "lexicon.txt", lex)
    assert read_lexicon(tmp_path / "lexicon.txt", spec) == lex
    with pytest.raises(AssertionError):
        read_lexicon(tmp_path / "lexicon.txt", SynthSpec(n_lex=51))
