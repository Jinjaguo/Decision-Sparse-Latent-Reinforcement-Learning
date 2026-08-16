# Next experiment after EXP_R7: EXP_R8 safety-enriched calibration stress

R7 finds acceptable aggregate calibration but weak unsafe recall/precision on
the stove subset. EXP_R8 should stress the safety head without changing the
decision interface.

- Reweight or stratify only the training folds to increase unsafe support;
  held-out folds remain untouched.
- Compare fixed 0.5, training-fold-selected thresholds, and conservative
  thresholds with explicit precision/recall tradeoffs.
- Report calibration by task and pre-action force-validity subset, with null
  metrics for absent support.
- Keep all zero-step and unsafe candidates; do not open independent
  confirmation until the threshold protocol is frozen.
