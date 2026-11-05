"""Write tests/golden/{cpu_fp32,gpu_bf16}.json from the CURRENT code. Run only on code whose math is the reference.

    CUDA_VISIBLE_DEVICES= python scripts/make_golden.py cpu
    python scripts/make_golden.py gpu        # on the DGX Spark
"""
import json
import sys
from pathlib import Path

import torch

spike = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(spike))
from tests import golden_util as g  # noqa: E402

which = sys.argv[1]
g.GOLDEN_DIR.mkdir(exist_ok=True)
if which == "cpu":
    m, x, y, lengths = g.build_cpu()
    res = g.capture(m, x, y, torch.device("cpu"), amp=False, lengths=lengths)
    name = "cpu_fp32.json"
else:
    m, x, y = g.build_gpu(compile_=False)
    res = g.capture(m, x, y, torch.device("cuda"), amp=True)
    name = "gpu_bf16.json"
(g.GOLDEN_DIR / name).write_text(json.dumps(res, indent=1))
print(name, res["loss"], [s["counts"] for s in res["stages"]])
