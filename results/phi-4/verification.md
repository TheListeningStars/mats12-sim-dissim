# Verification — `phi-4`

Every number below was recomputed from the saved CSVs by `scripts/verify.py`, which imports nothing from the analysis modules. `pipeline` is what the pipeline wrote; `independent` is what fresh code gets. Rows marked NEW are checks the pipeline does not perform at all.

| quantity                                              | pipeline     | independent      | match           | note                                                                                                                   |
|:------------------------------------------------------|:-------------|:-----------------|:----------------|:-----------------------------------------------------------------------------------------------------------------------|
| horse race: R² M0 (scenario)                          | 0.149134     | 0.149134         | OK              |                                                                                                                        |
| horse race: R² M1 (c only)                            | 0.032757     | 0.032757         | OK              |                                                                                                                        |
| horse race: R² M2 (both)                              | 0.227417     | 0.227417         | OK              |                                                                                                                        |
| horse race: ΔR² (M2−M0)                               | 0.078283     | 0.078283         | OK              |                                                                                                                        |
| horse race: n off-diag pairs                          | 56.000000    | 56.000000        | OK              |                                                                                                                        |
| in-distribution (diagonal) AUROC                      | 0.994712     | 0.994712         | OK              |                                                                                                                        |
| cell-level permutation p for ΔR²                      | not computed | 0.529            | NEW             | n_cells=8, n_perm=5000; null mean ΔR²=0.1153, observed=0.0783                                                          |
| observed ΔR² vs its own null mean                     | —            | 0.0783 vs 0.1153 | BELOW NULL MEAN | if observed sits below the null mean, random cell->c assignment explains as much or more than the real assignment      |
| variance decomp: R² source cell alone                 | not computed | 0.062            | NEW             |                                                                                                                        |
| variance decomp: R² target cell alone                 | not computed | 0.587            | NEW             |                                                                                                                        |
| variance decomp: R² source + target                   | not computed | 0.673            | NEW             |                                                                                                                        |
| variance decomp: R² scenario src+tgt (=M0)            | not computed | 0.149            | NEW             |                                                                                                                        |
| belief-label vs ground-truth-label agreement, |t̂|<0.3 | not computed | 0.863            | NEW             | n=204; agreement falling toward 0.5 in low bins means the LABEL is noisy there, not that deception is harder to detect |
| belief-label vs ground-truth-label agreement, 0.3–0.6 | not computed | 0.912            | NEW             | n=217; agreement falling toward 0.5 in low bins means the LABEL is noisy there, not that deception is harder to detect |
| belief-label vs ground-truth-label agreement, >0.6    | not computed | 0.978            | NEW             | n=546; agreement falling toward 0.5 in low bins means the LABEL is noisy there, not that deception is harder to detect |

All directly-comparable quantities agree with the pipeline.
