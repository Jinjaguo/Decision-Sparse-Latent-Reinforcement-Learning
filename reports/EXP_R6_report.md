# EXP_R6 report: factorized-action ablations

## Protocol

R6 repeats the ten leave-one-demo-out folds from R5 and compares six models:
scalar utility, full factorized success/unsafe heads, factorized without route
identity, factorized without action chunks, success-only, and safety-only. All
models use only the corrected R3 pre-outcome fields. The independent
confirmation set remains unopened; training ran offline on the RTX 4090.

## Aggregate result

| variant | tie-aware top-1 utility | tie-aware top-3 utility | mAP |
|---|---:|---:|---:|
| scalar full | 0.894 | 0.881 | 0.915 |
| factorized full | 0.933 | 0.906 | 0.948 |
| factorized without route | 0.933 | 0.894 | 0.927 |
| factorized without action | 0.925 | 0.897 | 0.926 |
| success-only | 0.919 | 0.897 | 0.929 |
| safety-only | 0.915 | 0.915 | 0.944 |

## Interpretation

The full factorized model is strongest on aggregate mAP and ties the best
top-1 result. Removing route identity leaves top-1 unchanged but lowers mAP;
removing action chunks lowers both top-1 and mAP. Success-only and safety-only
controls are weaker than the joint factorized score. This supports a
factorized state/action consequence interface rather than a pure route-prior
explanation, although the study is still limited to the development tasks and
offline labels.

Taskwise metrics and bootstrap intervals are stored in
`runs/exp_r6_s1_factorized_ablations_20260816_r1/metrics.json`.

## Resource audit

After EXP_R6 completion, C: was checked once: 1,180.10 GB free of 1,862.02
GB. E: was not inspected.
