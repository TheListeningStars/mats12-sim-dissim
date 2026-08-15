# Verification — `qwen3.5-9b`

Every number below was recomputed from the saved CSVs by `scripts/verify.py`, which imports nothing from the analysis modules. `pipeline` is what the pipeline wrote; `independent` is what fresh code gets. Rows marked NEW are checks the pipeline does not perform at all.

| quantity                                              | pipeline     | independent      | match           | note                                                                                                                   |
|:------------------------------------------------------|:-------------|:-----------------|:----------------|:-----------------------------------------------------------------------------------------------------------------------|
| horse race: R² M0 (scenario)                          | 0.199919     | 0.199919         | OK              |                                                                                                                        |
| horse race: R² M1 (c only)                            | 0.018106     | 0.018106         | OK              |                                                                                                                        |
| horse race: R² M2 (both)                              | 0.289576     | 0.289576         | OK              |                                                                                                                        |
| horse race: ΔR² (M2−M0)                               | 0.089657     | 0.089657         | OK              |                                                                                                                        |
| horse race: n off-diag pairs                          | 240.000000   | 240.000000       | OK              |                                                                                                                        |
| in-distribution (diagonal) AUROC                      | 0.979276     | 0.979276         | OK              |                                                                                                                        |
| H2 Spearman ρ                                         | 0.570588     | 0.570588         | OK              |                                                                                                                        |
| H2 n targets                                          | —            | 16               | NEW             | this is the true sample size for H2                                                                                    |
| cell-level permutation p for ΔR²                      | not computed | 0.151            | NEW             | n_cells=16, n_perm=5000; null mean ΔR²=0.0458, observed=0.0897                                                         |
| observed ΔR² vs its own null mean                     | —            | 0.0897 vs 0.0458 | above null mean | if observed sits below the null mean, random cell->c assignment explains as much or more than the real assignment      |
| variance decomp: R² source cell alone                 | not computed | 0.067            | NEW             |                                                                                                                        |
| variance decomp: R² target cell alone                 | not computed | 0.687            | NEW             |                                                                                                                        |
| variance decomp: R² source + target                   | not computed | 0.740            | NEW             |                                                                                                                        |
| variance decomp: R² scenario src+tgt (=M0)            | not computed | 0.200            | NEW             |                                                                                                                        |
| belief-label vs ground-truth-label agreement, |t̂|<0.3 | not computed | 0.958            | NEW             | n=334; agreement falling toward 0.5 in low bins means the LABEL is noisy there, not that deception is harder to detect |
| belief-label vs ground-truth-label agreement, 0.3–0.6 | not computed | 0.899            | NEW             | n=455; agreement falling toward 0.5 in low bins means the LABEL is noisy there, not that deception is harder to detect |
| belief-label vs ground-truth-label agreement, >0.6    | not computed | 0.981            | NEW             | n=992; agreement falling toward 0.5 in low bins means the LABEL is noisy there, not that deception is harder to detect |

All directly-comparable quantities agree with the pipeline.
