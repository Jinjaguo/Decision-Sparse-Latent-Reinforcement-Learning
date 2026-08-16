# EXP_R4 report: offline factorized-action baselines

## Protocol

R4 consumes only the final valid R3 run
`runs/exp_r3_s1_instrumented_candidate_matrix_20260816_r4`. It checks that
the pre-outcome table has no post-action fields and that its row hashes are
unique before fitting. The fixed demo-level split is demos 0--5 for training,
6--7 for validation, and 8--9 for test. The untouched independent
confirmation set remains unopened.

The three baselines are a label-free retrieval score, a scalar MLP predicting
success-and-not-safety utility, and a factorized MLP with separate success and
unsafe heads. The factorized score is
`P(success) * (1 - P(unsafe))`. The offline models ran on the RTX 4090 via
CUDA; no simulator was imported.

## Test result

The test split contains 12 branches and 108 candidates:

| baseline | tie-aware top-1 utility | tie-aware top-3 utility | mAP |
|---|---:|---:|---:|
| retrieval-only | 0.917 | 0.917 | 0.960 |
| scalar utility | 1.000 | 0.972 | 0.990 |
| factorized heads | 0.917 | 0.972 | 0.980 |

All 12 test branches had positive utility support. Bootstrap intervals are
stored in `runs/exp_r4_s1_offline_factorized_baselines_20260816_r3/metrics.json`.

## Interpretation

The scalar state/action model improves over the label-free retrieval score on
this development split. The factorized model does not improve over the
scalar model in top-1 or mAP, so EXP_R4 does not establish a factorization
advantage. The test set is only 12 branches and the validation/test outcomes
are highly favorable; this is an offline development result, not a final
confirmation. The next step should stress calibration, leave-one-demo-out
robustness, and factorized ablations before any untouched confirmation.

## Audit note

The first R4 attempt failed immediately because object state vectors had
task-dependent lengths. The implementation now pads object positions and
quaternions to fixed widths and asserts finite, fixed-dimensional features.
The failed run is retained; only the corrected `..._r3` run is valid.
