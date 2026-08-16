# Next experiment after EXP_R6: EXP_R7 calibration and decision-threshold audit

R6 supports the full factorized model, but ranking metrics alone do not show
whether its safety head is calibrated enough for deployment. EXP_R7 should
audit probability calibration and threshold selection using the same
leave-one-demo-out folds.

## Plan

- Save success and unsafe probabilities for every held-out candidate.
- Evaluate reliability bins, Brier score, expected calibration error, and
  precision/recall at safety thresholds selected only on training folds.
- Compare full factorized, scalar, success-only, and safety-only models.
- Report calibration separately for all candidates, already-successful
  zero-step candidates, and candidates with valid pre-action force.

No threshold may be selected from held-out outcomes. Do not open the untouched
confirmation set until calibration and threshold behavior are frozen.
