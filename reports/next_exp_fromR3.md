# Next experiment after EXP_R3: EXP_R4 offline factorized-action baselines

Use only
`runs/exp_r3_s1_instrumented_candidate_matrix_20260816_r3`.

## Question

Does factorizing action consequences into admissible state/action channels
improve candidate ranking over retrieval-only and scalar-consequence
baselines without using post-action labels at decision time?

## Required baselines

1. Retrieval-only route prior, with no outcome-derived route prior.
2. Scalar-consequence model using only admissible pre-action fields.
3. Factorized model with separate action, safety, and task-consequence heads;
   outcome labels are training targets only.

Use deterministic demo-level train/validation/test splits, target-demo
exclusion, tie-aware top-k and average-precision metrics, and bootstrap
confidence intervals over demos. The candidate table must be checked for
post-action names and hashes before fitting. Fit only on branches whose
pre-action candidate row is valid; retain zero-step and unsafe labels in the
outcome join.

## Stopping and safety gates

- A leakage assertion or duplicate pre-outcome hash invalidates the run.
- A split with no positive/negative support is reported as undefined rather
  than silently imputed.
- Do not use the untouched independent confirmation set.
- Offline fitting may use multiple CPU cores or GPU after checking the
  available backend; no simulator parallelism is needed.
