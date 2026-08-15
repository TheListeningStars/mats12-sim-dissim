# Verification — `olmo-3-7b-instruct`

Every number below was recomputed from the saved CSVs by `scripts/verify.py`, which imports nothing from the analysis modules. `pipeline` is what the pipeline wrote; `independent` is what fresh code gets. Rows marked NEW are checks the pipeline does not perform at all.

| quantity                                              | pipeline     | independent      | match           | note                                                                                                                   |
|:------------------------------------------------------|:-------------|:-----------------|:----------------|:-----------------------------------------------------------------------------------------------------------------------|
| horse race: R² M0 (scenario)                          | 0.382691     | 0.382691         | OK              |                                                                                                                        |
| horse race: R² M1 (c only)                            | 0.247151     | 0.247151         | OK              |                                                                                                                        |
| horse race: R² M2 (both)                              | 0.403568     | 0.403568         | OK              |                                                                                                                        |
| horse race: ΔR² (M2−M0)                               | 0.020877     | 0.020877         | OK              |                                                                                                                        |
| horse race: n off-diag pairs                          | 156.000000   | 156.000000       | OK              |                                                                                                                        |
| in-distribution (diagonal) AUROC                      | 0.948049     | 0.948049         | OK              |                                                                                                                        |
| H2 Spearman ρ                                         | 0.280220     | 0.280220         | OK              |                                                                                                                        |
| H2 n targets                                          | —            | 13               | NEW             | this is the true sample size for H2                                                                                    |
| cell-level permutation p for ΔR²                      | not computed | 0.439            | NEW             | n_cells=13, n_perm=5000; null mean ΔR²=0.0226, observed=0.0209                                                         |
| observed ΔR² vs its own null mean                     | —            | 0.0209 vs 0.0226 | BELOW NULL MEAN | if observed sits below the null mean, random cell->c assignment explains as much or more than the real assignment      |
| variance decomp: R² source cell alone                 | not computed | 0.295            | NEW             |                                                                                                                        |
| variance decomp: R² target cell alone                 | not computed | 0.195            | NEW             |                                                                                                                        |
| variance decomp: R² source + target                   | not computed | 0.510            | NEW             |                                                                                                                        |
| variance decomp: R² scenario src+tgt (=M0)            | not computed | 0.383            | NEW             |                                                                                                                        |
| belief-label vs ground-truth-label agreement, |t̂|<0.3 | not computed | 0.665            | NEW             | n=224; agreement falling toward 0.5 in low bins means the LABEL is noisy there, not that deception is harder to detect |
| belief-label vs ground-truth-label agreement, 0.3–0.6 | not computed | 0.841            | NEW             | n=239; agreement falling toward 0.5 in low bins means the LABEL is noisy there, not that deception is harder to detect |
| belief-label vs ground-truth-label agreement, >0.6    | not computed | 0.966            | NEW             | n=770; agreement falling toward 0.5 in low bins means the LABEL is noisy there, not that deception is harder to detect |

All directly-comparable quantities agree with the pipeline.
