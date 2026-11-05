"""Byte and BPE controls: bpb math, token loader, token FLOPs, control shapes vs the FLOPs gate."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from hnet import tokens
from hnet.data import MixLoader, row_probs
from hnet.evaluation import evaluate_tokens, token_domain_metrics
from hnet.flops import block_flops, flops_per_byte, token_flops_per_byte
from hnet.model import HNetLM

SPIKE = Path(__file__).resolve().parents[1]
GATE_FLOPS, GATE_TOL = 22.4e6, 0.05


# ---------- bpb math ----------

def test_bpb_oracle_is_eight_bits():
    """A model that pays ln(256) nats per byte scores 8 bpb, however the text is cut into tokens."""
    nbytes = np.array([1, 2, 3, 4, 1], dtype=np.uint16)
    toks = np.array([[0, 3, 1, 2, 4, 3, 0]], dtype=np.int32)
    n_bytes = tokens.window_bytes(toks, nbytes)              # targets 3 1 2 4 3 0 -> 4+2+3+1+4+1
    assert n_bytes == 15
    ce_sum = n_bytes * math.log(256)
    assert tokens.bpb(ce_sum, n_bytes) == pytest.approx(8.0)
    m = token_domain_metrics("edu", ce_sum, n_tokens=6, n_bytes=n_bytes)
    assert m["val/edu/bpb"] == pytest.approx(8.0)
    assert m["val/edu/ce_per_token"] == pytest.approx(ce_sum / 6)
    assert m["val/edu/bytes_per_token"] == pytest.approx(15 / 6)
    assert m["val/edu/ce"] == pytest.approx(math.log(256))   # nats per byte, same meaning as the byte models' val/*/ce


class _Uniform(torch.nn.Module):
    """Stub LM: every position predicts uniform over V ids. CE per token = ln V exactly."""

    def __init__(self, V):
        super().__init__()
        self.V = V

    def forward(self, x):
        return SimpleNamespace(logits=torch.zeros(*x.shape, self.V))


def test_evaluate_tokens_uniform_stub():
    V, L = 8, 16
    rng = np.random.default_rng(0)
    nbytes = rng.integers(1, 6, V).astype(np.uint16)
    wins = rng.integers(0, V, (6, L + 1)).astype(np.int32)
    mcfg = dict(d_model=[16], stages=[], mlp_ratio=2.6667, mlp_multiple_of=16, vocab_size=V, main=dict(layers=1))
    m = evaluate_tokens(_Uniform(V), {"edu": torch.from_numpy(wins)}, nbytes, mcfg, L, 3, torch.device("cpu"),
                        torch.bfloat16, "edu")
    n_bytes = int(nbytes[wins[:, 1:]].astype(np.int64).sum())
    assert m["val/edu/bpb"] == pytest.approx(6 * L * math.log(V) / (math.log(2) * n_bytes), rel=1e-6)
    assert m["val/bpb"] == m["val/edu/bpb"]
    assert m["val/edu/ce_per_token"] == pytest.approx(math.log(V), rel=1e-6)
    assert m["val/edu/bytes_per_token"] == pytest.approx(n_bytes / (6 * L))
    assert m["flops/fwd_per_byte"] == pytest.approx(token_flops_per_byte(mcfg, L, n_bytes / (6 * L))["fwd"])


def test_eval_windows_cover_distinct_targets_and_are_deterministic():
    toks = np.arange(10_000, dtype=np.uint16)
    w = tokens.eval_windows(toks, 64, 40, multiple_of=8)
    assert w.shape == (40, 65) and w.dtype == np.int32
    tgt = w[:, 1:].ravel()
    assert len(np.unique(tgt)) == len(tgt)                           # no target counted twice
    assert (w[:, 1:] == w[:, :-1] + 1).all()                         # each window is contiguous
    assert np.array_equal(w, tokens.eval_windows(toks, 64, 40, multiple_of=8))
    assert tokens.eval_windows(toks, 64, 10_000, multiple_of=8).shape[0] == (9999 // 64) // 8 * 8   # cut to what fits


def test_nbytes_table_covers_every_byte_of_real_text():
    """With a real byte-level BPE: sum of token byte lengths == utf-8 length, for ascii, accents and emoji."""
    tk = pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    corpus = ["the cat sat on the mat. the café is open. naïve résumé, 日本語のテキスト 🙂🙂 func main() {}\n"] * 50
    tok.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=400, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False))
    table = tokens.token_nbytes_table(tok.get_vocab(), tok.get_vocab_size(), {"<|endoftext|>"})
    assert table[tok.token_to_id("<|endoftext|>")] == 1
    for text in ["hello world", "café ünïcödé", "日本語🙂🙂 test", "  \n\n\tfunc main() {\n}\n", ""]:
        ids = tok.encode(text).ids
        assert int(table[ids].astype(np.int64).sum()) == len(text.encode("utf-8")), text
    assert tokens.bytes_to_unicode()[ord("a")] == "a" and len(set(tokens.bytes_to_unicode().values())) == 256


# ---------- token loader ----------

def _token_sources(tmp_path, bpts, size=20_000):
    srcs = []
    for i, (name, (w, bpt)) in enumerate(bpts.items()):
        f = tmp_path / f"{name}.u16"
        np.full(size, 1000 + i, dtype=np.uint16).tofile(f)         # ids above 255: the loader must not truncate to uint8
        srcs.append(dict(name=name, file=str(f), weight=w, bytes_per_token=bpt))
    return srcs


def test_row_probs_give_byte_fractions():
    p = row_probs([0.9, 0.1], [4.5, 3.0])
    tok_bytes = p * np.array([4.5, 3.0])
    assert tok_bytes[0] / tok_bytes.sum() == pytest.approx(0.9)
    assert p.sum() == pytest.approx(1.0)


def test_token_loader_mix_dtype_and_shift(tmp_path):
    srcs = _token_sources(tmp_path, {"edu": (0.9, 4.5), "go": (0.1, 3.0)})
    ld = MixLoader(srcs, seq_len=32, batch_size=64, seed=0, device="cpu", dtype=np.uint16)
    bytes_from = np.zeros(2)
    n_batches = 300
    for _ in range(n_batches):
        x, y = ld.next()
        assert x.dtype == torch.long and x.shape == (64, 32) and y.shape == (64, 32)
        assert int(x.max()) >= 1000                                  # no uint8 wrap
        assert (x == x[:, :1]).all()                                 # a row never mixes sources
        for s in x[:, 0].tolist():
            bytes_from[s - 1000] += 32 * (4.5, 3.0)[s - 1000]
    frac = bytes_from / bytes_from.sum()
    assert abs(frac[1] - 0.1) < 0.01, frac                           # byte mix is 0.9 / 0.1, not row mix


def test_token_loader_shift_and_prefetch_parity(tmp_path):
    f = tmp_path / "ramp.u16"
    np.arange(5000, dtype=np.uint16).tofile(f)
    src = [dict(name="a", file=str(f), weight=1.0, bytes_per_token=4.0)]
    a = MixLoader(src, 16, 8, seed=3, device="cpu", dtype=np.uint16)
    b = MixLoader(src, 16, 8, seed=3, device="cpu", prefetch=2, dtype=np.uint16)
    for _ in range(5):
        xa, ya = a.next()
        xb, yb = b.next()
        assert torch.equal(xa, xb) and torch.equal(ya, yb)           # prefetch keeps the order
        assert (ya == xa + 1).all() and (xa[:, 1:] == xa[:, :-1] + 1).all()
    b.pf.close()


def test_byte_loader_path_unchanged(tmp_path):
    """Default dtype is uint8 with the plain weight normalisation (golden path keeps its RNG stream)."""
    f = tmp_path / "b.bin"
    np.arange(4000, dtype=np.int64).astype(np.uint8).tofile(f)
    ld = MixLoader([dict(name="a", file=str(f), weight=3.0), dict(name="b", file=str(f), weight=1.0)], 8, 4, 0, "cpu")
    assert not ld.token_mode and np.allclose(ld.p, [0.75, 0.25])


# ---------- FLOPs ----------

def _cfg(d, layers, V, ratio=2.6667, mult=64, head_dim=64):
    return dict(d_model=[d], stages=[], mlp_ratio=ratio, mlp_multiple_of=mult, vocab_size=V, main=dict(layers=layers),
                head_dim=head_dim)


def test_token_flops_counts_blocks_and_head_once_per_token():
    d, layers, V, L, bpt = 128, 3, 1000, 64, 4.0
    cfg = _cfg(d, layers, V)
    hidden = 384                                                    # ceil(128 * 2.6667 / 64) * 64
    per_token = layers * block_flops(d, hidden, L, None) + 2 * d * V
    out = token_flops_per_byte(cfg, L, bpt)
    assert out["fwd_per_token"] == pytest.approx(per_token)
    assert out["fwd"] == pytest.approx(per_token / bpt)
    assert out["train"] == pytest.approx(3 * per_token / bpt)
    assert out["fwd_per_token"] == flops_per_byte(cfg, L, [])["fwd"]   # same counter as the byte models
    with pytest.raises(AssertionError):
        token_flops_per_byte(dict(cfg, stages=[{}]), L, bpt)


def _compose(experiment: str):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(str(SPIKE / "configs"), version_base="1.3"):
        return compose("config", overrides=[f"experiment={experiment}", "train=router"])


def test_byte_control_meets_flops_gate_and_builds():
    cfg = _compose("byte_transformer")
    m = OmegaConf.to_container(cfg.model, resolve=True)
    assert m["stages"] == [] and cfg.train.seq_len == 4096 and len(cfg.train.lr_mult) == 1
    fl = flops_per_byte(m, cfg.train.seq_len, [])["fwd"]
    assert abs(fl / GATE_FLOPS - 1) < GATE_TOL, fl
    model = HNetLM(m)
    assert model.n_stages == 0 and len(model.stage_param_groups()) == 1
    assert model.net.stack.window is None                           # global attention


def test_bpe_control_meets_flops_gate_and_train_bytes():
    cfg = _compose("bpe_transformer")
    meta = json.loads((SPIKE / "runs" / "bpe32k_meta.json").read_text())
    m = OmegaConf.to_container(cfg.model, resolve=True)
    assert m["kind"] == "token" and m["stages"] == [] and m["vocab_size"] == meta["vocab_size"] == 32768
    assert cfg.train.seq_len == 1024 and cfg.train.batch_size == 16 and m["d_model"] == [512]
    bpt_edu_val = meta["files"]["edu_val"]["bytes_per_token"]
    fl = token_flops_per_byte(m, cfg.train.seq_len, bpt_edu_val)["fwd"]
    assert abs(fl / GATE_FLOPS - 1) < GATE_TOL, fl
    w = {s.name: s.weight for s in cfg.data.train_sources}
    bpt_mix = 1 / sum(w[n] / meta["files"][f"{n}_train"]["bytes_per_token"] for n in w) * sum(w.values())
    planned = cfg.train.max_steps * cfg.train.batch_size * cfg.train.seq_len * bpt_mix
    assert abs(planned / cfg.data.token.target_train_bytes - 1) < 0.01, planned
    HNetLM(dict(m, d_model=[64], main=dict(layers=1), vocab_size=300))   # kind=token builds


def test_token_kind_rejects_stages(cfg_factory):
    with pytest.raises(AssertionError):
        HNetLM(dict(cfg_factory("eoc"), kind="token"))
    HNetLM(dict(cfg_factory("eoc"), kind="hnet"))
