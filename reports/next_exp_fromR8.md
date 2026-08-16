# Next experiment after EXP_R8: EXP_R9 safety-ranking diagnostic

Threshold stress shows that the safety head misses the same unsafe cases at
all fixed thresholds. EXP_R9 should inspect ranking rather than thresholding:

- report unsafe-candidate rank, unsafe-vs-safe score separation, and
  per-task leave-one-demo-out safety average precision;
- compare the current safety head with a force-aware safety-only head using
  pre-action force and contact inputs only;
- preserve the fixed threshold results as a negative control;
- do not claim deployment safety or open untouched confirmation while unsafe
  ranking remains weak.
