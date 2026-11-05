# H-Net: a research reimplementation of Cartesia's "From Tokens to Bytes" claims

This repository tests, at small scale, the claims that Cartesia makes about its multi-stage H-Net in two blog posts:

- [From Tokens to Bytes, Part I](https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1)
- [From Tokens to Bytes, Part II](https://www.cartesia.ai/blog/from-tokens-to-bytes-part-2)
- The original H-Net paper: [Dynamic Chunking for End-to-End Hierarchical Sequence Modeling](https://arxiv.org/abs/2507.07955) (code: [goombalab/hnet](https://github.com/goombalab/hnet))

It is an independent research implementation. It is not affiliated with Cartesia or the paper's authors. The posts describe an improved H-Net, but they do not publish its router. We build on the 2025 lookahead router and test the posts' claims against controls at matched FLOPs per byte.

Scale: one RTX 5090. The models have about 42M parameters and train on 12.58B bytes (about 8e17 training FLOPs). The Cartesia runs are about 3B parameters. We test directions and orderings, not their numbers.

## What we test

An H-Net reads raw bytes. Each stage has an encoder and a decoder. The router between them learns where chunks start. The main network runs only on the coarsest chunks.

```
bytes ─▶ stage 0 encoder ─▶ router ─▶ ~3-byte chunks
                                         │
              stage 1 encoder ─▶ router ─▶ ~9-byte chunks
                                               │
                                    main network (global Transformer)
                                               │
              stage 1 decoder ◀─ dechunk ◀─────┘
                                         │
bytes ◀─ stage 0 decoder ◀─ dechunk ◀────┘
```

The reference layout follows the posts: 4 Mamba-2 blocks per side at stage 0, 1 sliding-window attention block plus 4 Mamba-2 blocks per side at stage 1, and a 10-layer global Transformer main network. Widths are 256, 256, and 512.

## Results so far

Every claim is an experiment. Each experiment commits its predictions before the run and gets one verdict: supported, not supported, or can't tell. The checklist below is Phase 1 of [the project constitution](docs/constitution.md).

| # | Blog claim | Status at our scale | Record |
|---|---|---|---|
| 1 | A 2-stage H-Net beats BPE at matched FLOPs/byte | **Supported.** Edu val bpb: H-Net 1.0905, 1.0979, 1.0913 (3 seeds) against BPE 1.1156, 1.1109 (2 seeds). The matched 1-stage H-Net scored 1.1581. | [seeds vs BPE](lab/experiments/successes/2026-11-04-reference-layout-seeds-vs-bpe.md) |
| 2 | Each stage holds its compression ratio | Holds in every run (gate: within 15% of 3 x 3) | the experiment gates |
| 3 | Boundaries follow words | **Supported.** Stage 0 finds word starts with the spaces deleted. Stage-1 cuts match the blog's own figure chunks (F1 0.59 and 0.38, against 0.27 for a fixed-stride chunker). | [hidden words](lab/experiments/successes/2026-11-04-chunks-find-hidden-words.md), [blog texts](lab/experiments/successes/2026-11-05-chunks-match-blog-texts.md) |
| 4 | Code compresses more than prose | Holds. H-Net bytes per stage-1 chunk divided by BPE bytes per token: Go 5.33, SQL 3.43, English 1.83. | [seeds vs BPE](lab/experiments/successes/2026-11-04-reference-layout-seeds-vs-bpe.md) |
| 5 | Learned boundaries beat entropy-proxy boundaries | Running. The first entropy seed ended at 1.1378, worse than every learned seed. | [plan](lab/experiments/planned/2026-11-05-learned-vs-entropy-boundaries.md) |
| 6 | Morpheme-like stage-0 cuts | Not seen. Our in-word cuts look like the 2025 router's, as the blog says. | [blog texts](lab/experiments/successes/2026-11-05-chunks-match-blog-texts.md) |
| 7 | In-context tokenization: fewer chunks as a new word repeats | Not seen. Our chunk counts stay flat. The blog credits this to its new router. | [blog texts](lab/experiments/successes/2026-11-05-chunks-match-blog-texts.md) |

Items 6 and 7 are reported but do not gate the next phase, because the blog credits them to the router it did not publish.

Many experiments failed, and they are kept on purpose. Most of `lab/experiments/refuted/` is the search for why our first implementation's main network went unused ("dead"). The answer was mostly initialization and layout. The reference layout and init fixed it.

### Throughput

- Packed inner levels (no padding) and compiled stack segments cut the step from 86.8 to about 71 ms ([note](.agents/notes/implemented/feature/2026-11-05-throughput-compile-and-packing.md)).
- At matched FLOPs, the Mamba-2 H-Net trains at 1.46x the wall clock of BPE (73 against 50 ms/step). At d=256 the Mamba-2 kernels take 31 of the 71 ms.
- An H-Net with attention-only outer stages runs at 54 ms/step. Its loss experiment is in progress ([plan](lab/experiments/planned/2026-11-05-attention-only-outer-stages.md)).

## Next

Phase 2 makes the main network looped (weight-tied, iterated). The loop runs over a sequence about 9x shorter than the bytes, so each extra iteration is cheap. A side quest tests Lorentz-model hyperbolic parts in the H-Net. Both are described in [the constitution](docs/constitution.md).

## Layout

| Path | Contents |
|---|---|
| `lab/spikes/2026-10-30-two-stage-hnet-smoke/` | The model, trainer, configs, probes, and tests. Start with its [README](lab/spikes/2026-10-30-two-stage-hnet-smoke/README.md). |
| `lab/spikes/2026-11-01-reference-hnet-probe/` | Runs the official `goombalab/hnet` code as a reference, plus the Mamba-2 build recipe. |
| `lab/experiments/` | Every experiment: `planned/`, `successes/`, `refuted/`, with artifacts in `results/`. |
| `.agents/notes/` | Design decisions and the alternatives we weighed. |
| `docs/constitution.md` | Project goals, scope, and success criteria. |
| `docs/references/cartesia-tokens-to-bytes.md` | Our notes on what the posts disclose. |

`src/` is empty on purpose. Code moves there only after human review.

## Running it

From `lab/spikes/2026-10-30-two-stage-hnet-smoke/`:

```sh
uv venv .venv && uv pip install -e .
# Mamba-2 blocks also need mamba_ssm and causal_conv1d (build recipe: lab/spikes/2026-11-01-reference-hnet-probe/README.md)
python scripts/prepare_data.py ...          # uint8 byte memmaps, see the spike README
HNET_DATA_DIR=<data dir> python train.py experiment=reflayout_2stage train=ladder
python -m pytest -q                         # HNET_TEST_CUDA=1 adds the GPU tests
```

Configs use Hydra (`configs/`). Runs log the full resolved config to Weights & Biases. Data is FineWeb-Edu plus Go from `bigcode/starcoderdata`, with SQL kept out of training as a held-out domain.

## How this repository works

This repo is built on an [AI project template](https://github.com/bigwolfeman/ai-project-template) for research done with AI agents. Agents write the code, run the experiments, and write them up. A person reviews them. The rules are in [AGENTS.md](AGENTS.md), and the experiment protocol is in [lab/experiments/AGENTS.md](lab/experiments/AGENTS.md). Predictions are committed before results, and failed experiments stay in the record.

The public history starts from one squashed commit. Commit hashes cited inside the records (for example "code 0252bf6") refer to the private development history before publication.
