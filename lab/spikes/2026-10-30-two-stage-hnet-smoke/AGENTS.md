# AGENTS.md — two-stage H-Net smoke spike

Spike package, not production. Pseudocode and layout: [README.md](README.md). Rules: [lab/AGENTS.md](../../AGENTS.md).

- Every level is a `Stage` or `Main` with the same call: `net(h, lengths, pos, pack=None) -> (h, aux)` (`pack`: hnet/packing.py, built by the outer Stage). Do not special-case stage 0 or 1.
- Routers register by name in `hnet/routers.py` and return a `ChunkPlan`. Chunking code must not branch on the router name.
- Causality tests (`tests/test_causality.py`) gate every change to routers, chunking, or stages. Run on CPU with `CUDA_VISIBLE_DEVICES=`.
- GPU jobs run on the local 5090, one at a time, with `nice -n 19` and at most 2 CPU threads (the power budget does not allow GPU plus CPU load). `HNET_TEST_CUDA=1` turns on the GPU tests. A second machine (env in `~/hnet-smoke`) is for side jobs.
- Hyperparameters live in `configs/`. A new knob goes in YAML, not in Python.
- `runs/` holds the summaries of the 2026-10-30 smoke runs.
- Train data is a weighted list of sources in `configs/data/bytes.yaml` (`train_sources`). Eval runs on every entry of `val_sets`.
  SQL is eval only. Never add it to `train_sources`; `tests/test_data.py` fails if you do.
- `hnet_over_bpe` needs `tiktoken`. The cl100k bytes-per-token cache (`*.bpe-cl100k_base.json`) sits next to each val `.bin`; copy it with the file.
- Throughput: `Stage.forward` has exactly one device-to-host sync (anchor width M). Do not add `.item()`, `nonzero` or boolean indexing to the step path. Profile with `scripts/profile_step.py` (`PROF_SYNC=1` lists syncs). Golden values in `tests/golden/` pin the math; regenerate them only from reference code (`scripts/make_golden.py`).
- Compile structure: each Stack compiles per segment, the code between two Mamba-2 calls (`hnet/blocks.py` `run_segment`); inner levels run packed (`model.pack_inner`, `hnet/packing.py`). Note: `.agents/notes/implemented/feature/2026-11-05-throughput-compile-and-packing.md`.
- Train loaders prefetch on a thread (`data.prefetch`). Keep the data on local NVMe, not the FUSE HDD.
- Compile-vs-eager parity (`tests/test_gpu.py`) uses default init in bf16 and random weights in fp32. Random weights in bf16 flip hard router decisions, which is noise, not a compile bug.
- `model.main.kind: parcae` (prelude, injected core loop, coda; `hnet/parcae.py`) is the looped main. `model.main.loops` > 1 erases the input on text; keep it at 1. `train.grow_from` (N-stage final.pt -> (N+1)-stage model, new stage above main) is the growth knob. Notes: `.agents/notes/implemented/feature/2026-11-04-parcae-main.md`, `2026-11-03-looped-main.md`, `2026-11-03-grown-hierarchy.md`.
- Frozen boundaries: `model.boundaries` (`hnet/boundaries.py`, proxy byte LM entropy) with router `given` on every stage; the proxy is a plain attribute of `HNetLM`, never a submodule. Thresholds: `scripts/calibrate_entropy.py`. Note: `.agents/notes/implemented/feature/2026-11-05-frozen-entropy-boundaries.md`.
- Queues source `scripts/chain_lib.sh` (log, wait_gpu, suite_gate, smoke). Results dirs pull wandb with `scripts/pull_runs.py`.
- Chunk semantics: `hnet/semantics.py` logs `sem/<level>_s<stage>_auroc` at every probe (router-p AUROC, anchor 0.5; gold data in `HNET_GOLD_DIR`). Any stage code must fill `StageAux.p` and `in_pos`. Per-checkpoint report and Fig-4-style `chunks.html`: `scripts/chunk_semantics.py`.
- Controls: `byte_transformer` is `HNetLM` with `stages: []` (global attention over 4096 bytes). `bpe_transformer` sets `model.kind: token` (no stages, vocab 32768) and reads uint16 token memmaps from `HNET_DATA_DIR/bpe32k/` (`scripts/prepare_bpe.py`). Both must stay within 5% of 22.4M forward FLOPs/byte (`tests/test_controls.py`).
- BPE bpb is exact: sum of token CE / (ln 2 x bytes the target tokens stand for). `token_nbytes.npy` gives each token's bytes; EOS counts 1 (the 0 separator byte). Never use `ce / ln 2` of a token model as bpb. Token-loader rows are drawn by byte weight / bytes per token, so the byte mix stays 0.9 / 0.1.
- `train.max_steps` of a token model is set so planned train bytes equal `data.token.target_train_bytes` (train.py asserts 1%). If you retokenize, update `runs/bpe32k_meta.json` and `max_steps`.
