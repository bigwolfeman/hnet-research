"""Inference samples from trained checkpoints: chunk renderings and generated continuations, as one markdown file.

    python scripts/sample.py --out <file.md> <run_dir_or_final.pt> [...]

Pseudocode:
    for ckpt in checkpoints:
        model = build_model(ckpt.config.model); load weights
        render(val rows, probe paragraphs)          # '|' stage-0 cut, '#' cut shared with stage 1
        rows = prompts                              # all prompts in one batch, right-padded, per-row lengths
        repeat max_new times:                       # exact without a cache: prefix logits == full-run logits
            logits = model(rows, lengths)[row, lengths-1] / temperature
            append one sampled byte per row
        model.kind == token (BPE control): the same loop over BPE ids (prompts encoded with the run's tokenizer.json,
            EOS masked) until each row has max_new new bytes, then decode. Byte control (no stages) uses the byte loop.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnet import probes  # noqa: E402
from hnet.data import doc_starts  # noqa: E402
from hnet.model import HNetLM, build_model  # noqa: E402

SPIKE = Path(__file__).resolve().parents[1]
HAND_PROMPTS = ["The history of the Roman Empire", "Photosynthesis is the process by which",
                "package main\n\nimport \"fmt\"\n\nfunc main() {\n"]


def load(path: Path, device) -> tuple[HNetLM, dict]:
    ckpt_path = path / "final.pt" if path.is_dir() else path
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"]["model"])
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    return model.to(device).eval(), ckpt["config"]


@torch.no_grad()
def generate(model: HNetLM, prompts: list[bytes], max_new: int, temperature: float, seed: int, device) -> list[bytes]:
    g = torch.Generator(device=device).manual_seed(seed)
    rows = [list(p) for p in prompts]
    for _ in range(max_new):
        lengths = [len(r) for r in rows]
        width = -(-max(lengths) // model.multiple) * model.multiple
        x = torch.zeros(len(rows), width, dtype=torch.long, device=device)
        for i, r in enumerate(rows):
            x[i, :len(r)] = torch.tensor(r, device=device)
        lt = torch.tensor(lengths, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x, lt).logits
        last = logits[torch.arange(len(rows), device=device), lt - 1].float() / temperature
        nxt = torch.multinomial(torch.softmax(last, -1), 1, generator=g).squeeze(1).tolist()
        for r, b in zip(rows, nxt):
            r.append(b if b != 0 else 10)  # a doc separator would end the text; show it as a newline
    return [bytes(r[len(p):]) for r, p in zip(rows, prompts)]


@torch.no_grad()
def generate_tokens(model: HNetLM, tok, nbytes: np.ndarray, eos_id: int, prompts: list[bytes], max_new: int,
                    temperature: float, seed: int, device) -> list[bytes]:
    """BPE control: encode prompts, sample ids until every row has >= max_new new bytes, decode. EOS is masked out."""
    g = torch.Generator(device=device).manual_seed(seed)
    rows = [tok.encode(p.decode("utf-8", "replace")).ids for p in prompts]
    n0 = [len(r) for r in rows]
    got = [0] * len(rows)
    while min(got) < max_new:
        lengths = [len(r) for r in rows]
        width = -(-max(lengths) // model.multiple) * model.multiple
        x = torch.zeros(len(rows), width, dtype=torch.long, device=device)
        for i, r in enumerate(rows):
            x[i, :len(r)] = torch.tensor(r, device=device)
        lt = torch.tensor(lengths, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x, lt).logits
        last = logits[torch.arange(len(rows), device=device), lt - 1].float() / temperature
        last[:, eos_id] = float("-inf")
        nxt = torch.multinomial(torch.softmax(last, -1), 1, generator=g).squeeze(1).tolist()
        for i, t in enumerate(nxt):
            rows[i].append(t)
            got[i] += int(nbytes[t])
    return [tok.decode(r[n:]).encode("utf-8")[:max_new] for r, n in zip(rows, n0)]


def load_tokenizer(cfg: dict, data_dir: Path):
    from tokenizers import Tokenizer

    from hnet.tokens import load_meta, load_nbytes

    tdir = data_dir / cfg["data"]["token"]["dir"]
    meta = load_meta(tdir)
    return Tokenizer.from_file(str(tdir / "tokenizer.json")), load_nbytes(tdir), meta["eos_id"]


def fence(text: str) -> str:
    return "```text\n" + text.replace("```", "'''") + "\n```"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("~/data/hnet-bytes").expanduser())
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    edu = doc_starts(a.data_dir / "fineweb-edu/val.bin", 3, 700, a.seed)
    go = doc_starts(a.data_dir / "code/go_val.bin", 2, 500, a.seed)
    probe = json.loads((SPIKE / "data/probe_paragraphs.json").read_text())[:2]
    render_rows = edu[:2] + go[:1] + [np.frombuffer(p["text"].encode(), dtype=np.uint8) for p in probe]
    prompts = [bytes(r[:160]) for r in edu] + [bytes(r[:200]) for r in go] + [p.encode() for p in HAND_PROMPTS]

    out = ["# Inference samples", "",
           f"temperature {a.temperature}, {a.max_new} new bytes, seed {a.seed}. "
           "Chunk renderings: `|` = stage-0 cut, `#` = cut shared with stage 1, `↵` = newline.", ""]
    for path in a.checkpoints:
        model, cfg = load(path, device)
        name = cfg.get("wandb", {}).get("name") or path.name
        out += [f"## {name}", "", f"checkpoint `{path}`", ""]
        if model.n_stages:
            out += ["### Chunks", ""]
            for row in render_rows:
                out += [fence(probes.render(model, row, device)[0]), ""]
        out += ["### Generations", ""]
        if cfg["model"].get("kind", "hnet") == "token":
            tok, nbytes, eos_id = load_tokenizer(cfg, a.data_dir)
            gens = generate_tokens(model, tok, nbytes, eos_id, prompts, a.max_new, a.temperature, a.seed, device)
        else:
            gens = generate(model, prompts, a.max_new, a.temperature, a.seed, device)
        for p, g in zip(prompts, gens):
            out += [fence(p.decode("utf-8", "replace") + "【" + g.decode("utf-8", "replace") + "】"), ""]
        del model
        torch.cuda.empty_cache()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(out))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
