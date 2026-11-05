# Agent Note: grow a hierarchy from a trained model (train.grow_from)

Status: implemented

## Problem

In every 2-stage model we trained from scratch on text, the main network ends mostly unused: the stage-1 fate is above 0 in 1 of 5 seeds of the reference layout, and the blog split was dead up to 12.58B bytes. The 1-stage H-Net uses its main network. A test of the "init race" reading needs a 2-stage model whose main network starts as a working model ([grown hierarchy keeps main](../../../../lab/experiments/refuted/2026-11-03-grown-hierarchy-keeps-main.md)).

## Decision

`train.grow_from` names an N-stage `final.pt`. `hnet/grow.py` maps its weights into the fresh (N+1)-stage model: embedding, head, stages 0 to N-1, and the main network keep their roles, the outer stage's pad vector moves to the new stage (the new stage must have the outer stage's width), and the new stage directly above the main network keeps its fresh init. Every source key must land exactly once, and every target key must come from the source or from the new stage; anything else raises. The trainer logs the source path, its sha256, step, wandb name, and the count of fresh keys under `init_from` in the full config. AdamW state starts fresh, with the normal warmup. Tests: `tests/test_grow.py`.

## Alternatives considered

- **A transplant script that writes an `init_from` file.** That needs the target model built outside the trainer from a second copy of the config. Doing it in the trainer builds the model once, from the run's own config.
- **Insert the new stage at the byte level (outside stage 0).** The main network would then see the same chunk rate as before, but the byte embedding and head would move to a fresh stage, and the trained stage 0 would read the outputs of an untrained encoder. Inserting above the main network keeps the trained byte path intact.
- **Carry the AdamW state over.** The new stage has no state, and the moments of the old weights reflect a different model. A fresh optimizer with warmup is the simpler, standard choice.

## Consequences

- The growth changes the main network's input rate (here from 1/3 to 1/9 of the bytes). The main network must adapt to coarser chunks after the growth.
- The 1-stage phase spends more FLOPs per byte than the 2-stage model (36.1M against 22.6M in the reference layout), so a grown run at matched bytes is not at matched FLOPs.
- On text, the growth keeps the main network's state: the one seed whose 1-stage main network was in use stayed in use at both stages (stage-1 shuffle cost 0.62), and the two dead seeds stayed dead ([grown hierarchy keeps main](../../../../lab/experiments/refuted/2026-11-03-grown-hierarchy-keeps-main.md)). The growth is only as good as the 1-stage model it starts from.
- With windowed-attention outer stages and a 1-stage main network in use in every seed, the growth keeps each seed's level of use and does not raise it: the end order of stage-1 shuffle costs (0.12, 0.20, 0.34) is the order of the phase-1 costs, and the same layout from scratch ends dead at stage 1 in every seed ([windowed growth keeps main](../../../../lab/experiments/refuted/2026-11-04-windowed-growth-keeps-main.md)). Growth is a way to keep a live main network, not a way to make one.
