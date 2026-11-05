# AGENTS.md — Experiments

Experiments are measured inquiry. They are not spikes (`lab/spikes/`), not campaigns (`lab/campaigns/`), not design decisions (Agent Notes), and not incident reports (postmortems).

The scientific method is the format. **Question, hypothesis and predictions are written and committed before any run.** Filling predictions after seeing results is a process failure; delete the file and start over if that happened.

A **campaign** may coordinate several experiment files. The campaign `program.md` must link those files. It must not copy their hypotheses or predictions. Lab rules: [lab/AGENTS.md](../AGENTS.md).

## Layout

```
lab/experiments/
  algorithms/     Writeups (and notebooks) on methods — not run records
  planned/        Hypothesis + predictions + method; no results
  successes/      Main prediction supported
  refuted/        Main prediction not supported, or can't tell (the folder keeps its name for the verifier; Status: refuted)
  mixed/          Closed to new records. One main prediction makes a mixed verdict impossible. Kept for the verifier and old records.
  results/        Data, figures, notebooks outputs; linked from the writeup
  templates/      Copy these; do not edit in place as a live record
```

Keep `planned/`, `successes/`, `mixed/`, and `refuted/` in this slice. Write new records to `planned/`, `successes/` and `refuted/` only.

Naming for run records: `yyyy-mm-dd-short-slug.md` in `planned/`, `successes/`, or `refuted/`.

Algorithm writeups: `algorithms/yyyy-mm-dd-short-slug.md` plus optional `algorithms/yyyy-mm-dd-short-slug.ipynb`.

Result artifacts: `results/yyyy-mm-dd-short-slug/` matching the run file's slug. Large binaries go in `ignored/experiment-artifacts/` with a pointer file in `results/`.

## Name every experiment

Every experiment, ablation, arm, task, and trial group gets a name of 2 to 5 words that says what it tests. Examples: "Language-directed object removal", "Bag-of-words reader control", "No-text reader floor". The filename slug, the `# Experiment:` title, and every table and report use that name.

A letter-number code (`V1`, `S16`, `R16`) is not a name. A code may follow the name as an alias, for example "No-text reader floor (V4)". A code never stands alone in a title, a table cell, a Verdict, or a link. Reason: a reader who sees `V4` must open another file to learn what it tests, and an agent that meets `V4` in a later session guesses.

A control or floor arm says so in its name. The Question says what the arm is a control for.

`scripts/verify_template.py` checks that each experiment title has 2 to 5 words.

## Write the idea down as pseudocode

Every Method section holds pseudocode of the change under test, in a fenced block. The verifier checks for the fence. A reader should follow the idea without opening the diff. If the method is new, the algorithm writeup holds the full pseudocode and Method links to it.

## How to run an experiment

This is the method children learn, used the same way here. Write one planned file per question. Commit it before the run.

1. **Ask a question.** One sentence: "Does changing X change Y?" One question per experiment.
2. **Make a guess (hypothesis).** "I think X will make Y better / worse / no different, because ___." The "because" names the mechanism.
3. **Say what you expect to see (prediction).** A behavior against a control, not a number to hit. Allowed forms: direction (A better / worse / same as B), ordering (A < B < C), shape (the K-curve slopes down / is flat), sign. A number is allowed only when it is a fixed anchor (uniform ln vocab, 0, an oracle ceiling). "The same" means inside the seed spread of the controls. You measure that spread from repeated control runs. You do not choose it. Write at most three predictions. ONE is the main prediction, and only it answers the question.
4. **Make it a fair test.** Change one thing. The control is the same model without the change, on the same rows. Run enough seeds to know what "no change" looks like. If no spread has been measured, measuring it is the first job.
5. **Write what each answer makes you do.** "If supported, we do A. If not, we do B." If both answers lead to the same next step, do not run it.
6. **Observe.** Report paired differences with their intervals, and the whole curve, not one cut point.
7. **Conclude,** with one of three words for the main prediction:
   - **supported**: the result matches it, outside noise.
   - **not supported**: the result is the opposite, or there is no effect where one was expected.
   - **can't tell**: the result is inside noise. The test was too small. This is not a verdict on the idea. Write the bigger test that would answer it.
8. **Next question.** What the result taught, and what you try next.

### Gates are not predictions

Finite loss, peak memory, head-off checks and token counts are GATES. If a gate fails, the run is broken: fix it and rerun. Gates never count toward a verdict. Speed is a measurement. Report it. It is not a pass/fail line unless speed is the question.

### Rules

- The verdict comes from the main prediction, never from a count such as "5 of 11".
- No invented thresholds. A number in a prediction that is not a fixed anchor or a measured noise band gets deleted.
- No probabilities on predictions. An audit of 51 predictions found stated probabilities 10 to 25 points too high in the middle range.
- Read the earlier filings on the same arm before you write a prediction.
- Never edit predictions after the run starts. Method changes go in a dated amendment.

### Filing

| Verdict on the main prediction | Folder |
|---|---|
| supported | `successes/` |
| not supported | `failures/` |
| can't tell | `failures/`, with the next planned experiment named |

A filing leads with the main prediction's word and the paired number behind it.

### Example (Block-Parcae, rewritten)

- **Question.** Can a loop that runs over chunks in series replace the token loop?
- **Hypothesis.** I think the coda can decode tokens from the chunk states alone, because the spiral's chunk levels carry most of its gain.
- **Main prediction.** Block-4 CE is the same as plain Parcae + pointer (inside seed spread), and its K-curve slopes down.
- **What each answer does.** If supported, build the series design out. If not, keep the token stream in the loop.
- **Result.** 0.25 nats worse, far outside noise. Not supported. (The old filing said "failure 5/11" and hid this.)

In this template the filing folder for "not supported" and "can't tell" is `refuted/`, and its `Status:` line is `refuted`. The Verdict section says which of the two words applies. The name `refuted/` is historical: a "can't tell" record there does not refute the idea.

## Lifecycle

1. If the method is new or non-obvious, write or update an algorithm note first.
2. Copy `templates/experiment.md` to `planned/yyyy-mm-dd-slug.md`. Fill Question, Hypothesis, Predictions, Method. Predictions are behaviors against a control, with one main prediction. Commit.
3. Run the experiment. Do not edit Predictions. If the method must change, write why under Method as an amendment dated after the original commit; if the predictions no longer apply, reject this run and open a new planned file.
4. Store artifacts under `results/yyyy-mm-dd-slug/`.
5. Fill Results, Verdict, and Updated hypothesis. Move the file to `successes/` or `refuted/`. Set `Status:` to match.
6. If the outcome changes shipped design, add or update an Agent Note in the same change.

## Status values

Must match the folder:

- `Status: planned`
- `Status: success`
- `Status: refuted` (not supported, or can't tell)
- `Status: mixed` (old records only)

`planned/` files must contain `## Predictions` and `## Method` with real content. They must not contain a `## Results` section (use the placeholder sentence in the template's comment only; the live file omits Results entirely).

`successes/` and `refuted/` must contain Predictions, Method, Results, Verdict, and Updated hypothesis.

## Verdict

Folder status is not the same as a hypothesis verdict. The Verdict section opens with the main prediction's word and the paired number behind it.

Hypothesis verdicts:

- **supported**: the main prediction matched the result, outside noise. File under `successes/`.
- **not supported**: the result was the opposite, or showed no effect where one was expected. File under `refuted/`.
- **can't tell**: the result sat inside noise. The test was too small. File under `refuted/` and name the next planned experiment. This does not kill the idea.

Do not file a "success" because the code ran. Success is about the claim. A crash is never a verdict: it is a failed gate. Fix and rerun.

## Trials vs experiments

A **trial** is one execution of one candidate. Trial outcomes are `accepted`, `rejected`, `invalid`, `inconclusive`, and `crashed`. Those words describe candidate handling. They are not hypothesis verdicts.

The campaign ledger is execution evidence. Experiment documents summarize at the hypothesis level. Do not paste trial JSONL into the experiment writeup.

Link the evaluator identity, the protocol, and the artifact pointers needed to interpret the result. Generated campaign reports live under `lab/campaigns/<slug>/reports/`. Those reports are not experiment verdicts.

## Forbidden

- Results in `planned/`
- Editing Predictions after artifacts exist
- An experiment writeup with no link to method or data
- Dumping unlabeled numbers in `results/` with no matching writeup
- Turning a `lab/spikes/` spike into an experiment by writing predictions after the fact
- Copying experiment predictions into campaign `program.md`
- Using trial outcomes (`accepted`, `rejected`, `invalid`, `inconclusive`, `crashed`) as hypothesis verdicts
- Predictions with an invented numeric threshold or a stated probability
- A verdict built from a count of predictions that held
