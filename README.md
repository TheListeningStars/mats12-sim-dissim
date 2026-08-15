# Truth-conflict axis vs. deception-probe transfer

MATS 12.0 application experiment. Full design in [`docs/PLAN.md`](docs/PLAN.md); working rules for the coding agent in [`CLAUDE.md`](CLAUDE.md).

**Question.** Existing work carves deception by pragmatic *scenario* (lying, jailbreak, roleplay, sycophancy). We carve it by a **measured, graded truth-conflict score `c`** — how far the assertion the model *actually makes* sits from its internally-represented truth — and test whether `c` predicts probe transfer *better than* scenario labels.

> **Labels come from measured behaviour, not the prompt condition.** The model ignores
> the instruction often (compliance: 53% for instructed lying, 50% for counterfactual),
> so labelling rows by which prompt they came from produces a *instruction detector*, not
> a deception probe — under that labelling, honest rows separated from
> dissimulation-rows-where-the-model-refused-to-lie at AUROC 1.000. `src/behavior.py`
> parses the verdict the model actually gave, and everything downstream is built on that.
> Run with `--label-source condition` to reproduce the artifact. See `results/LOG.md`.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login          # needed for gated Llama-3.1-8B; Qwen/Gemma-3 are open
```

GPU: a single 16GB card (Colab, or a rented A100/H100) is enough; 7–8B models run in 4-bit. Confirm you can extract residual-stream activations before doing anything else.

## Run

```bash
# smoke-test the whole pipeline on CPU with structured-random activations (no model,
# no GPU; outputs land in separate '<model>-synthetic' dirs and are never findings):
python scripts/run_all.py --dry-run --synthetic

# tiny-N run with the real model (still needs a GPU / model download):
python scripts/run_all.py --dry-run

# real run on the primary model:
python scripts/run_all.py --model qwen2.5-7b-instruct

# replication:
python scripts/run_all.py --model llama-3.1-8b-instruct --headline-only
```

Individual stages:

```bash
python -m src.data_build          # -> data/manifest.csv
python -m src.activations  --model qwen2.5-7b-instruct   # -> cache/
python -m src.truth_axis   --model qwen2.5-7b-instruct   # -> d_truth, c per example, validity plot
python -m src.probes       --model qwen2.5-7b-instruct   # -> probes/
python -m src.transfer     --model qwen2.5-7b-instruct   # -> results/transfer_matrix.*
python -m src.baselines    --model qwen2.5-7b-instruct   # -> results/horse_race.*
python -m src.geometry     --model qwen2.5-7b-instruct   # -> results/monotonicity.*
```

## Layout

```
docs/PLAN.md          # design of record — read first
CLAUDE.md             # agent guardrails
requirements.txt
src/
  config.py           # models, layers, paths, seed
  data_build.py       # build stratified dataset + manifest
  activations.py      # cache residual-stream activations
  behavior.py         # parse the verdict the model ACTUALLY gave; compliance diagnostics
  truth_axis.py       # fit d_truth (Marks-Tegmark), compute c, validity
  probes.py           # logistic-regression + diff-of-means probes
  transfer.py         # train-on-cell / test-on-all -> transfer matrix
  baselines.py        # random, within-class, style, behavioral, scenario horse race
  geometry.py         # direction cosines, c-vs-transfer monotonicity curve
scripts/run_all.py    # orchestrator
data/  cache/  probes/  results/   # generated (gitignored)
```

## What "done" means
`run_all.py` yields: transfer heatmap, nested-comparison table (H1), `c`-vs-transfer curve (H2), instrument-validity plot, populated `results/LOG.md`, replicated on a 2nd model. A clean negative result is a valid, submittable outcome.
