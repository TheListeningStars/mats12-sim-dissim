# CLAUDE.md — project context for Claude Code

## What this project is
A 16-hour MATS 12.0 application experiment. Read `docs/PLAN.md` first and treat it as the source of truth. The one-sentence goal: **test whether a measured "truth-conflict" axis organizes deception-probe transfer better than the existing scenario taxonomy.**

## Non-negotiable framing (do not drift)
- The contribution is a **measured, graded latent variable** (`truth-conflict score c`), NOT a validation of Baudrillard. Baudrillard is one line of motivation, nothing more. Never write results language that "confirms Baudrillard."
- The **primary result is a horse race**: does `c` add explained variance in transfer AUROC *beyond* scenario labels? (nested model comparison). If it doesn't, that is the finding — report it cleanly.
- Do **not** claim novelty for: probes collapsing OOD, deception heterogeneity, role-play shallowness, belief-modification confounds. These are prior work (see PLAN §2).

## Scientific guardrails (Neel's review criteria — enforce these)
1. **Verify every number before it's reported.** No metric enters a figure/write-up unchecked. Unverified results are the #1 disqualifier.
2. **Baselines are mandatory, not optional:** random-direction floor, within-class OOD, style-shift, behavioral-only, and the scenario-label model. A result without its baseline is incomplete.
3. **Instrument validity first.** Before using `c`, confirm it behaves (dissimulation cells score high, truth-valueless cells ≈ 0). If `c` doesn't track the construct, stop and say so.
4. **Cache activations; never recompute.** Seed everything (`config.SEED`). Log every hyperparameter to `results/run_meta.json`.
5. **Watch for the negative branch.** If cross-`c` transfer drop ≈ within-class OOD drop by the first full matrix, that's the negative result — flag it, don't chase it as a bug.
6. **Red-team positives.** Check length, template leakage, single-topic confounds before believing any positive.

## Conventions
- Python 3.11, `venv`, deps pinned in `requirements.txt`.
- All paths via `src/config.py`. Data manifest is a single CSV; activations cached as `.npy`/`.pt` keyed by `(model, layer, cell)`.
- Small, testable functions. Add a `--dry-run` / small-N mode to every script so the pipeline can be smoke-tested on CPU before burning GPU time.
- Keep a running lab log in `results/LOG.md` (what ran, what was found, what's suspect).

## Build order (mirrors PLAN §10)
1. `src/data_build.py` — highest-risk piece, do first.
2. `src/activations.py` — cache residual stream.
3. `src/truth_axis.py` — fit `d_truth`, compute `c`, validity check.
4. `src/probes.py` — LR + diff-of-means.
5. `src/transfer.py` — transfer matrix.
6. `src/baselines.py` — incl. scenario-label horse race.
7. `src/geometry.py` — cosines, monotonicity curve.
8. `scripts/run_all.py` — orchestrate; write figures to `results/`.

## Definition of done
`scripts/run_all.py` produces: the transfer heatmap, the nested-comparison table (H1), the `c`-vs-transfer monotonicity curve (H2), instrument-validity plot, and `results/LOG.md` populated — replicated on a second model for the headline cells.
