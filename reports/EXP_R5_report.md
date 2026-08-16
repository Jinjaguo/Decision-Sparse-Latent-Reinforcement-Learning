# EXP_R5 report: leave-one-demo-out robustness

## Protocol

R5 refits the three R4 baselines ten times, holding out one complete demo
across all three tasks in each fold. The held-out demo is excluded from model
fitting and feature normalization. The corrected R3 pre-outcome schema and
unique-hash assertions remain active. The experiment uses no simulator and no
untouched confirmation data; MLP fitting ran on the RTX 4090.

## Aggregate result

Across 60 held-out branches and 540 candidates:

| baseline | tie-aware top-1 utility | tie-aware top-3 utility | mAP |
|---|---:|---:|---:|
| retrieval-only | 0.750 | 0.828 | 0.862 |
| scalar utility | 0.894 | 0.881 | 0.915 |
| factorized heads | 0.933 | 0.906 | 0.948 |

All 60 branches had positive utility support. The factorized model improves
over scalar by 0.039 top-1 utility and 0.033 mAP in this cross-demo audit;
its bootstrap intervals are stored in
`runs/exp_r5_s1_leave_one_demo_out_20260816_r2/metrics.json`.

## Interpretation

The fixed R4 split favored the scalar model, but leave-one-demo-out testing
reverses that ordering and gives factorization the stronger result. This is
evidence that separate success and safety heads may improve distributional
robustness, not proof of causal action-consequence factorization. The next
experiment should perform taskwise/routewise ablations and inspect whether
the gain comes specifically from safety-heavy candidates or from a generic
regularization effect.

## Audit note

The first R5 launch failed because its direct script entry point did not add
the repository root before importing the R4 helper module. The corrected
`..._r2` run is valid; the failed attempt produced no scientific output.
