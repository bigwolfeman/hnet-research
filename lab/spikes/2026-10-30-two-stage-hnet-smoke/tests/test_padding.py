"""Gate 2: a row's logits do not depend on the batch pad length, other rows, or pad byte content."""
import pytest
import torch

from tests.conftest import make_cfg, randomize
from hnet.model import HNetLM

TOL = 1e-4


@pytest.mark.parametrize("router", ["eoc", "eoc_lag", "lookahead"])
@pytest.mark.parametrize("n_stages", [1, 2])
def test_padding_invariance(router, n_stages):
    torch.manual_seed(0)
    m = randomize(HNetLM(make_cfg(router, n_stages, multiple=32)).eval())
    g = torch.Generator().manual_seed(3)
    n = 200
    seq = torch.randint(0, 256, (n,), generator=g)
    other = torch.randint(0, 256, (512,), generator=g)

    def run(width, with_other, pad_seed):
        gp = torch.Generator().manual_seed(pad_seed)
        rows = [torch.cat([seq, torch.randint(0, 256, (width - n,), generator=gp)])]
        lens = [n]
        if with_other:
            rows.append(torch.cat([other[:width], torch.randint(0, 256, (max(0, width - 512),), generator=gp)])[:width])
            lens.append(min(512, width))
        with torch.no_grad():
            return m(torch.stack(rows), torch.tensor(lens)).logits[0, :n]

    ref = run(224, False, 1)
    for width, with_other, seed in [(256, False, 2), (512, False, 3), (512, True, 4), (640, True, 5)]:
        err = (run(width, with_other, seed) - ref).abs().max().item()
        assert err < TOL, f"padding changed logits: width={width} other={with_other} err={err}"
