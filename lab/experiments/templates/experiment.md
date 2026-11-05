# Experiment: <2-5 word descriptive name>

Status: planned

Date proposed: YYYY-MM-DD
Algorithm: (link to `../algorithms/…` or "none — standard method")
Owner: (human or agent session)

## Question

What will this run teach us? One paragraph. No solution, no predicted numbers here.

## Hypothesis

A falsifiable claim. Example: "Replacing attention with X does not increase validation loss by more than 2% at 10k steps under config Y."

## Predictions

Write these before any run. Behaviors against a control: direction, ordering, shape, or sign. A number only as a fixed anchor (uniform ln vocab, 0, an oracle ceiling). "The same" means inside the measured seed spread of the controls. No probabilities. At most three predictions. Commit this file while it still lives in `planned/` with no Results section.

Main prediction (only this one answers the Question):

- …

Other predictions (at most two):

- …

If supported, we do: …

If not supported, we do: …

Gates (the run is broken if one fails; gates are not predictions):

- …

## Method

Include pseudocode of the change under test, so a reader can follow the idea without opening the diff.

Protocol, data, controls, metrics, compute, seeds, stop conditions. Enough that someone else can repeat the run.

Amendments after the first commit go in a dated subsection. If an amendment invalidates Predictions, abandon this file and open a new planned experiment.

## Related

Links to Agent Notes, algorithm writeups, prior runs.
