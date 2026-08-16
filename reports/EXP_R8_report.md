# EXP_R8 report: fixed safety-threshold stress

R8 evaluated the R5 held-out safety probabilities at fixed thresholds 0.3,
0.5, 0.7, and 0.9. No threshold was selected from held-out outcomes.

| threshold | flagged | unsafe precision | unsafe recall | accepted utility rate |
|---:|---:|---:|---:|---:|
| 0.3 | 29 | 0.241 | 0.583 | 0.898 |
| 0.5 | 25 | 0.280 | 0.583 | 0.893 |
| 0.7 | 24 | 0.292 | 0.583 | 0.893 |
| 0.9 | 20 | 0.350 | 0.583 | 0.890 |

All thresholds detect exactly 7 of the 12 unsafe candidates. Raising the
threshold improves precision only by reducing false positives; it does not
improve recall. The current limitation is therefore safety-head separation,
not threshold tuning. Stove is the only task with unsafe positives and shows
the same fixed-recall behavior.

After EXP_R8 completion, C: was checked once: 1,180.10 GB free of 1,862.02
GB. E: was not inspected.
