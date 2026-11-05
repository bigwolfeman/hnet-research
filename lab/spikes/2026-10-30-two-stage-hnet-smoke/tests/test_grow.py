"""hnet/grow.py: an N-stage model's weights go into an (N+1)-stage model; only the new stage starts fresh."""
import copy

import pytest
import torch

from hnet.grow import grow_state_dict
from hnet.model import HNetLM
from hnet.probes import ablated
from tests.conftest import make_cfg, randomize


def _pair(outer_d=32, new_d=32):
    """(1-stage cfg, 2-stage cfg): the 2-stage inserts a stage of width new_d between stage 0 and main."""
    src = make_cfg("lookahead", 1, ratios=(3.0,))
    tgt = copy.deepcopy(src)
    tgt["d_model"] = [outer_d, new_d, src["d_model"][-1]]
    new = copy.deepcopy(src["stages"][0])
    new["window"] = 16
    tgt["stages"] = [src["stages"][0], new]
    return src, tgt


def test_every_source_weight_lands_in_place():
    src_cfg, tgt_cfg = _pair()
    torch.manual_seed(0)
    src = randomize(HNetLM(src_cfg), seed=1)
    torch.manual_seed(1)
    tgt = HNetLM(tgt_cfg)
    fresh_before = {k: v.clone() for k, v in tgt.state_dict().items()}
    sd, fresh = grow_state_dict(src.state_dict(), 1, tgt)
    tgt.load_state_dict(sd, strict=True)
    s, t = src.state_dict(), tgt.state_dict()
    for k, v in s.items():
        tk = k.replace("net.inner.", "net.inner.inner.", 1) if k.startswith("net.inner.") else k
        tk = "net.inner.pad_dim" if k == "net.pad_dim" else tk
        assert torch.equal(t[tk], v), k
    assert fresh and all(k.startswith("net.inner.") and not k.startswith("net.inner.inner.") for k in fresh)
    for k in fresh:
        assert torch.equal(t[k], fresh_before[k]), f"fresh key {k} changed"
    # the grown model's own stage 0 and main are the source's: same modules, same outputs on their own inputs
    h = torch.randn(2, 64, 64)
    with torch.no_grad():
        assert torch.equal(src.net.inner.stack(h), tgt.net.inner.inner.stack(h))


def test_grown_model_runs_and_its_main_is_live_under_the_probe():
    src_cfg, tgt_cfg = _pair()
    torch.manual_seed(0)
    src = randomize(HNetLM(src_cfg), seed=1)
    tgt = HNetLM(tgt_cfg)
    tgt.load_state_dict(grow_state_dict(src.state_dict(), 1, tgt)[0], strict=True)
    x = torch.randint(0, 256, (3, 128), generator=torch.Generator().manual_seed(2))
    tgt.eval()
    with torch.no_grad():
        base = tgt(x).logits
        with ablated(tgt.stages()[1], "shuffle"):
            shuf = tgt(x).logits
    assert torch.isfinite(base).all()
    assert (shuf - base).abs().max() > 0, "the new stage's decoder must see the main output"


def test_width_mismatch_fails_loud():
    src_cfg, tgt_cfg = _pair(new_d=48)
    src, tgt = HNetLM(src_cfg), HNetLM(tgt_cfg)
    with pytest.raises(ValueError, match="new stage width"):
        grow_state_dict(src.state_dict(), 1, tgt)


def test_stage_count_mismatch_fails_loud():
    src_cfg, _ = _pair()
    src = HNetLM(src_cfg)
    with pytest.raises(ValueError, match="stages"):
        grow_state_dict(src.state_dict(), 1, HNetLM(src_cfg))


def test_unused_source_key_fails_loud():
    src_cfg, tgt_cfg = _pair()
    src, tgt = HNetLM(src_cfg), HNetLM(tgt_cfg)
    sd = dict(src.state_dict(), **{"net.extra": torch.zeros(1)})
    with pytest.raises(KeyError, match="no place"):
        grow_state_dict(sd, 1, tgt)
