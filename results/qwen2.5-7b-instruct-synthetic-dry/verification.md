# Verification — `qwen2.5-7b-instruct-synthetic-dry`

Every number below was recomputed from the saved CSVs by `scripts/verify.py`, which imports nothing from the analysis modules. `pipeline` is what the pipeline wrote; `independent` is what fresh code gets. Rows marked NEW are checks the pipeline does not perform at all.

| quantity                                              | pipeline     | independent      | match           | note                                                                                                                   |
|:------------------------------------------------------|:-------------|:-----------------|:----------------|:-----------------------------------------------------------------------------------------------------------------------|
| horse race: R² M0 (scenario)                          | 0.078287     | 0.078287         | OK              |                                                                                                                        |
| horse race: R² M1 (c only)                            | 0.003958     | 0.003958         | OK              |                                                                                                                        |
| horse race: R² M2 (both)                              | 0.083320     | 0.083320         | OK              |                                                                                                                        |
| horse race: ΔR² (M2−M0)                               | 0.005033     | 0.005033         | OK              |                                                                                                                        |
| horse race: n off-diag pairs                          | 240.000000   | 240.000000       | OK              |                                                                                                                        |
| in-distribution (diagonal) AUROC                      | 0.570980     | 0.570980         | OK              |                                                                                                                        |
| H2 Spearman ρ                                         | -0.300000    | -0.300000        | OK              |                                                                                                                        |
| H2 n targets                                          | —            | 16               | NEW             | this is the true sample size for H2                                                                                    |
| cell-level permutation p for ΔR²                      | not computed | 0.828            | NEW             | n_cells=16, n_perm=5000; null mean ΔR²=0.0149, observed=0.0050                                                         |
| observed ΔR² vs its own null mean                     | —            | 0.0050 vs 0.0149 | BELOW NULL MEAN | if observed sits below the null mean, random cell->c assignment explains as much or more than the real assignment      |
| variance decomp: R² source cell alone                 | not computed | 0.093            | NEW             |                                                                                                                        |
| variance decomp: R² target cell alone                 | not computed | 0.123            | NEW             |                                                                                                                        |
| variance decomp: R² source + target                   | not computed | 0.215            | NEW             |                                                                                                                        |
| variance decomp: R² scenario src+tgt (=M0)            | not computed | 0.078            | NEW             |                                                                                                                        |
| belief-label vs ground-truth-label agreement, |t̂|<0.3 | not computed | 0.976            | NEW             | n=286; agreement falling toward 0.5 in low bins means the LABEL is noisy there, not that deception is harder to detect |
| belief-label vs ground-truth-label agreement, 0.3–0.6 | not computed | 1.000            | NEW             | n=213; agreement falling toward 0.5 in low bins means the LABEL is noisy there, not that deception is harder to detect |
| belief-label vs ground-truth-label agreement, >0.6    | not computed | 1.000            | NEW             | n=21; agreement falling toward 0.5 in low bins means the LABEL is noisy there, not that deception is harder to detect  |

All directly-comparable quantities agree with the pipeline.
