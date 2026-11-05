"""One-time: encode parquet text to uint8 memmaps (docs separated by byte 0, vocab 256).

    # FineWeb-Edu (done already; do not rebuild)
    python scripts/prepare_data.py --out ignored/data/fineweb-edu --target val.bin=20 --target train.bin=1024 \
        --glob '<fineweb-edu>/data/*/*.parquet'
    # Code (StarCoderData; text column is `content`). Targets fill in order, so val docs are disjoint from train.
    nice -n 15 python scripts/prepare_data.py --glob 'ignored/data/raw/starcoderdata/go/*.parquet' \
        --out ignored/data/code --column content --target go_val.bin=32 --target go_train.bin=1536
    nice -n 15 python scripts/prepare_data.py --glob 'ignored/data/raw/starcoderdata/sql/*.parquet' \
        --out ignored/data/code --column content --target sql_val.bin=32

`--target NAME=MB` is repeated, no defaults. pyarrow runs on 1 thread so the job stays light next to a busy GPU.
Writes `<out>/<first target stem>.meta.json` with the per-file byte and doc counts.
"""
import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnet.data import STARCODER_HEADER, prepare  # noqa: E402

HEADERS = {"none": None, "starcoder": STARCODER_HEADER}

ap = argparse.ArgumentParser()
ap.add_argument("--glob", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--column", default="text")
ap.add_argument("--strip-header", choices=sorted(HEADERS), default="none")
ap.add_argument("--target", action="append", required=True, metavar="NAME=MB")
a = ap.parse_args()
pa.set_cpu_count(1)
pa.set_io_thread_count(1)
targets = [(n, int(mb) * 2**20) for n, mb in (t.split("=") for t in a.target)]
stats = prepare(a.glob, a.out, targets, a.column, HEADERS[a.strip_header])
meta = Path(a.out) / f"{Path(targets[0][0]).stem}.meta.json"
meta.write_text(json.dumps(dict(glob=a.glob, column=a.column, strip_header=a.strip_header, **stats), indent=1))
print(json.dumps(stats, indent=1))
