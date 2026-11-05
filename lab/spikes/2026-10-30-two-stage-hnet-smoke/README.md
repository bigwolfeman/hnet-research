# Two-stage byte H-Net smoke test (spike)

Status: spike. Classified per [lab/AGENTS.md](../../AGENTS.md): informal exploration, no hypothesis. Verdicts on the
`eoc` router belong in `lab/experiments/`, not here. Vision: [constitution](../../../docs/constitution.md).
Background: [Cartesia reference](../../../docs/references/cartesia-tokens-to-bytes.md).
Source of truth for the 2025 mechanics: `goombalab/hnet` (cloned to `ignored/ref/hnet`).

Goal: a clean package (promotable to `src/hnet/`) that trains a recursive N-stage byte H-Net with all-Transformer stages,
two router variants selectable per stage in config, with FLOPs/byte accounting and hierarchy probes inside training.

## Pseudocode

```
model(bytes x (B,L), lengths):
    h = embed(x)
    h, aux = net(h, lengths, pos = arange(L))        # net is Stage or Main (same interface)
    logits = head(h)                                 # next-byte; loss = CE + sum_s w_s * ratio_loss_s

Main(h, lengths, pos):  return GlobalCausalTransformer(h), []

Stage(h, lengths, pos):                              # Stage(encoder, router, inner, decoder)
    e      = encoder(h)                              # causal sliding-window attention (flex_attention)
    resid  = residual_proj(e)                        # fp32, zero init
    plan   = router(e, lengths)                      # mask (anchors), p, conf, read_idx
    idx, counts = select_chunks(plan.mask)           # right-pad count to a multiple of 128
    x      = gather(e, idx)
    y, aux = inner(pad_dims(x), counts, gather(pos, idx))      # inner is Stage or Main
    ema    = ema_scan(y, gather(plan.p, idx))        # out_j = p_j y_j + (1-p_j) out_{j-1}, fp32 parallel scan
    up     = gather(ema, plan.read_idx)              # read_idx = -1 -> learned start vector
    z      = up * STE(plan.conf) + resid             # STE: forward 1, backward identity
    return decoder(z), [ratio_loss(plan)] + aux

router "lookahead" (2025):                           # byte t STARTS a chunk
    p_t = (1 - cos(q(h_{t-1}), k(h_t))) / 2 ; p_0 = 1 ; b_t = p_t >= 0.5
    chunk vector = h at start ; read_idx_t = cumsum(b)_t - 1

router "eoc" (our hypothesis):                       # byte t is the LAST of a chunk
    p_t = sigmoid(w . h_t + c) ; last valid position forced to 1 ; e_t = p_t >= 0.5
    chunk vector = h at end ; read_idx_t = cumsum(e)_t - e_t - 1      # chunk k reads main output k-1
    chunk 0 reads a learned start vector. Position t uses only h_{<=t} and ends strictly before t.

ratio_loss(p, mask, N) = N/(N-1) * ((1-F)(1-G) + F G (N-1)),  F = mean(mask), G = mean(p)   # F has no gradient
```

Data mix and per-domain eval (`hnet/data.py`, `hnet/evaluation.py`, `hnet/bpe.py`):

```
MixLoader.next():                                    # sources = cfg.data.train_sources [{name, file, weight}]
    src = rng.choice(n_sources, size=B, p=weights)   # one source per batch row; files are never re-mixed on disk
    o   = rng.integers(0, N[src] - L - 1)            # random offset inside that source's file
    x, y = mm[src][o : o+L], mm[src][o+1 : o+L+1]

evaluate(model, cfg.data.val_sets = [edu, go, sql]):           # sql is never in train_sources
    for domain d, fixed batches:
        bpb_d             = mean CE / ln 2
        bytes_per_chunk_k = total input bytes / sum(anchors at stage k)     # raw bytes per stage-k chunk (cumulative)
        ratio_k           = anchors_{k-1} / anchors_k                       # relative to the stage's own input
        bpt_d             = bytes / tokens of cl100k on the whole val_d.bin # once, cached in val_d.bpe-cl100k_base.json
        hnet_over_bpe_k   = bytes_per_chunk_k / bpt_d                       # Cartesia Fig. 6; > 1: H-Net compresses more
    wandb: val/{d}/{bpb, bytes_per_chunk_s{k}, ratio_s{k}, hnet_over_bpe_s{k}, bpe_bytes_per_token}
    primary_val (edu) also fills val/bpb, val/ratio_s{k}, flops/* and the probe slices
```

## Layout

| Path | Role |
|---|---|
| `hnet/blocks.py` | RMSNorm, RoPE, SwiGLU, attention (flex SWA, dense-mask SWA fallback, global SDPA), `parse_layout`, `Stack` (block layout string: T/t attention, M/m Mamba-2; `stages[i].enc_layout` / `dec_layout`, [note](../../../.agents/notes/implemented/feature/2026-11-02-stack-layouts-mamba2.md)) |
| `hnet/ssm.py` | Mamba-2 mixer (mamba_ssm `Mamba2`, settings from `model.ssm`) for m/M blocks; needs mamba_ssm and causal_conv1d ([recipe](../2026-11-01-reference-hnet-probe/README.md)) |
| `hnet/routers.py` | router registry: `lookahead`, `eoc`; `ChunkPlan` |
| `hnet/chunking.py` | `select_chunks`, `ema_scan` (Hillis-Steele, fp32), STE, `ratio_loss`, `RatioController` (`train.ratio_integral_lr > 0`: signed integral term that pins each ratio to its target; [note](../../../.agents/notes/implemented/feature/2026-10-31-ratio-controller.md)) |
| `hnet/stage.py` | `Stage`, `Main`, `StageAux`, init |
| `hnet/model.py` | `HNetLM`, `build_model` (dispatch on `model.kind`), loss, per-stage parameter groups, `describe()` |
| `hnet/synth.py` | Synthetic recall / no-recall byte languages and their exact next-byte oracle; built by `scripts/make_synth.py` into `$HNET_DATA_DIR/synth`; `oracle_bits(reach=R)` and `scripts/reach_anchors.py` give the best bpb of a model whose paths reach R bytes back; used with `data=synth_recall` / `data=synth_norecall` ([note](../../../.agents/notes/implemented/feature/2026-11-02-synthetic-recall-language.md)) |
| `hnet/jepa.py` | `CoarseJEPA`: auxiliary loss on the pooled main-network output over view pairs (`train.jepa`, `PairLoader` in `hnet/data.py`); Go code / doc-comment pairs from `scripts/goviews/main.go` ([note](../../../.agents/notes/implemented/feature/2026-11-02-coarse-jepa.md)) |
| `hnet/refmodel.py` | `RefHNetLM`: the reference H-Net (goombalab/hnet, vendored as `refnet`) behind the `HNetLM` interface (`model.kind=ref_hnet`, `configs/model/ref_hnet.yaml`). Needs the [reference environment](../2026-11-01-reference-hnet-probe/README.md); [note](../../../.agents/notes/implemented/feature/2026-11-01-reference-hnet-adapter.md) |
| `hnet/flops.py` | `flops_per_byte(cfg, seq_len, measured_ratios)`; `kind=ref_hnet` counts Mamba-2 blocks (recurrent form) and reference layouts |
| `hnet/probes.py` | boundary alignment, in-context tokenization, rendered sample |
| `hnet/data.py`, `scripts/prepare_data.py` | uint8 memmap builder, `MixLoader` (weighted sources), `ByteLoader` (eval) |
| `hnet/evaluation.py`, `hnet/bpe.py` | per-domain eval hook, bytes/chunk per stage, cl100k bytes-per-token with a JSON cache |
| `train.py`, `configs/` | Hydra + wandb entry point; `experiment/*.yaml` arms; `train=ladder` adds WSD compute-ladder rungs (`ladder_fracs`) off one run; `scripts/run_arms.sh` (`TRAIN=` picks the train group; fixed run dir per arm, rerun = skip done arms and resume partial ones via `train.resume_every`, [note](../../../.agents/notes/implemented/feature/2026-11-01-resumable-runs.md)), `scripts/run_ladder.sh` |
| `hnet/tokens.py`, `scripts/prepare_bpe.py` | BPE control: byte lengths of tokens, eval windows, bpb math; tokenizer trainer + uint16 tokenizer of every .bin |
| `scripts/ablate_coarse.py`, `probes.coarse_use`, `Stage.ablate` | ablation of each stage's coarse path (`up` zeroed, or taken from another document); logged at every eval as `probe/coarse_*` |
| `experiment/{byte,bpe}_transformer.yaml`, `scripts/run_controls_queue.sh` | the two controls (`model.kind: hnet` with no stages, and `model.kind: token`) and their queue |
| `tests/` | causality, padding, EMA, finite fwd/bwd, FLOPs, weighted loader, BPE math, GPU flex and compile-vs-eager checks |

## Run

```
uv venv .venv && uv pip install torch numpy hydra-core omegaconf wandb pyarrow pytest
CUDA_VISIBLE_DEVICES= python -m pytest tests                       # CPU gates
HNET_TEST_CUDA=1 python -m pytest tests/test_gpu.py                # on a GPU box
nice -n 15 python scripts/prepare_data.py --glob '<go parquet>' --out ignored/data/code --column content \
    --strip-header starcoder --target go_val.bin=32 --target go_train.bin=1536     # also sql_val.bin=32; see the script docstring
HNET_DATA_DIR=... python train.py experiment=lookahead_2stage train.max_steps=...
python scripts/prepare_bpe.py --data-dir <hnet-bytes> --out-dir <hnet-bytes>/bpe32k      # BPE control data (run on the Spark)
HNET_DATA_DIR=... python train.py experiment=bpe_transformer train=router               # or byte_transformer
```

## Notes on the design

- **Padding.** Attention is causal, so a real position never reads a pad position to its right. `tests/test_padding.py`
  checks logits against the batch pad length, other rows, and pad byte content.
- **Causality.** `tests/test_causality.py` checks two things: changing bytes after `t` leaves logits up to `t` unchanged,
  and running only the prefix `x[:t+1]` gives the same logits. The prefix form also covers the forced last boundary.
  Both fail if `read_index` is moved by one chunk (mutation check done by hand).
- **Shapes.** Chunk counts round up to a multiple of `chunk_multiple` (128), so `torch.compile` sees at most `L/128` shapes per stage.
  `select_chunks` costs one host sync per stage.
- **Widths.** `d_model` has one entry per stage plus main. Narrower outer stages are padded with a learned vector, as in 2025.
- **Deviations from the 2025 code** are listed in the final report of the run that produced this spike
  and in [the Agent Note](../../../.agents/notes/proposed/architecture/2026-10-30-two-stage-hnet-smoke-spike.md).

## Smoke results (DGX Spark GB10, torch 2.14.1+cu130, 2000 steps x 8 x 4096 bytes, compiled blocks)

Not experiment results. Single seed, 65 M bytes seen, FineWeb (not Edu) CC-MAIN-2013-20. Raw data: `runs/`.

| | eoc 2-stage | lookahead 2-stage |
|---|---|---|
| final val bpb | 2.033 | 1.992 |
| measured ratios (target 4, 2) | 4.08, 1.94 | 3.16, 2.01 |
| bytes/s (post-warmup mean) | 100.7 k | 93.7 k |
| step time | 325 ms | 350 ms |
| peak mem | 6.8 GB | 7.1 GB |
| fwd FLOPs/byte at measured ratios | 19.7 M | 23.2 M |
| stage-0 boundary F1 (stride baseline) | 0.66 (0.29) | 0.77 (0.33) |
| stage-1 boundary F1 (stride baseline) | 0.50 (0.18) | 0.49 (0.21) |
| in-context tokenization (occ1 to occ5) | flat | flat |

wandb: eoc `8xz1kxja`, lookahead `giqio0uu` (project `hnet-smoke`).
The step time includes recompiles for new chunk-length shapes (about 10 s each, early in the run).

## Domain smoke (eoc 2-stage, `read_own_end: true, force_last: false`, edu 0.9 / go 0.1, DGX Spark)

Not an experiment result. One seed, 2000 steps x 16 x 4096 bytes, compiled blocks, 1 GB edu + 0.5 GB go train prefix.
These runs used code data that still had StarCoderData's `<reponame>/<filename>/<gh_stars>` headers on about half the docs. The data now strips them (`--strip-header starcoder`), so go/sql numbers from later runs are not comparable with this table.
Raw data: `runs/eoc_2stage_domain_*`. wandb `pn0ng405` (project `hnet-smoke`, group `two-stage-domain-smoke`).
Bytes per chunk is cumulative in raw bytes. `hnet/bpe` is `hnet_over_bpe` at stage 0 and stage 1. BPE = cl100k bytes per token.

| val set | bpb | bytes/chunk s0, s1 | BPE bytes/token | hnet/bpe s0, s1 |
|---|---|---|---|---|
| edu | 1.820 | 3.90, 7.88 | 4.683 | 0.832, 1.683 |
| go | 1.564 | 3.18, 6.46 | 3.499 | 0.908, 1.847 |
| sql (eval only) | 3.228 | 1.99, 3.74 | 2.684 | 0.742, 1.392 |

## Coarse-path ablation (diagnostic, no prior prediction)

Not an experiment result: a spike probe run after the router and control experiments, on their final checkpoints. `scripts/ablate_coarse.py`, 16 fixed edu val batches (the training eval batches; base matches the logged final bpb).
At stage s, `up` (everything below stage s, read back to stage s positions) is replaced before `z = up * STE(conf) + resid`. "shuffle" takes `up` from another row of the batch, a different document, at the same scale. The cell is the rise in bpb.

| checkpoint | stage 0: zero / shuffle | stage 1 (input = main network): zero / shuffle |
|---|---|---|
| lookahead 2-stage s0, s1, s2 | +2.01 / +1.35, +2.38 / +1.45, +2.66 / +1.14 | +0.30 / +0.043, +0.40 / +0.006, +0.17 / +0.011 |
| eoc 2-stage s0, s1, s2 | +3.18 / +0.58, +3.42 / +0.73, +3.15 / +0.26 | +0.17 / +0.001, +0.18 / +0.002, +0.06 / +0.006 |
| eoc_lag 2-stage s0 | +3.09 / +0.13 | +0.02 / +0.002 |
| lookahead 1-stage s0, s1 | +3.17 / +1.10, +3.56 / +1.03 (main) | |
| eoc 1-stage s0, s1 | +3.68 / +0.20, +3.61 / +0.18 (main) | |

Reading: in every 2-stage checkpoint the main network (8 layers, d 512, most of the parameters) carries almost no document-specific information to the stage-1 decoder. Its output is near-constant (lookahead 2-stage s0: mean-vector norm 30.9, per-position spread 11.7; the 1-stage main: 17.6 and 16.4). The 2-stage models are, in effect, a stage-0 + stage-1 local model. In 1-stage models, `lookahead` uses its main network (+1.0 bpb when shuffled) and `eoc` barely does (+0.2).
