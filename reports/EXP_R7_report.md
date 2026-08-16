# EXP_R7 report: calibration and safety-threshold audit

## Protocol

R7 audits the held-out probabilities from R5 with a fixed 0.5 threshold. No
threshold was selected from held-out outcomes. The audit covers 540 candidates
and keeps the independent confirmation set unopened.

## Overall calibration

| output | Brier | ECE (10 bins) | precision @ 0.5 | recall @ 0.5 |
|---|---:|---:|---:|---:|
| success probability | 0.148 | 0.143 | 0.909 | 0.907 |
| unsafe probability | 0.038 | 0.042 | 0.280 | 0.583 |
| factorized utility | 0.150 | 0.144 | 0.906 | 0.903 |
| scalar utility | 0.176 | 0.180 | 0.906 | 0.860 |

Only 12 of 540 candidates were unsafe. Therefore the unsafe threshold
precision/recall has high uncertainty; task subsets with zero unsafe support
are explicitly reported as undefined (`null`), not as zero.

The stove task is the main calibration risk: unsafe Brier=0.114, ECE=0.120,
and the fixed threshold has precision=0.28 and recall=0.583. This supports
keeping the safety head but does not justify deployment without a larger,
safety-enriched development cohort.

## Resource audit

After EXP_R7 completion, C: was checked once: 1,180.10 GB free of 1,862.02
GB. E: was not inspected.
