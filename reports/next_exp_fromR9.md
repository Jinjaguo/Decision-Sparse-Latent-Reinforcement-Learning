# Next experiment after EXP_R9: EXP_R10 specialist route complementarity

R9 suggests the factorized unsafe head has a modest ranking advantage, but
unsafe support is too concentrated. EXP_R10 should test whether the named
routes have complementary success/safety profiles rather than merely
different outcome rates.

- Compute routewise success, unsafe, and utility distributions only within
  training folds, then evaluate held-out route selection.
- Compare factorized scores with route-blind and fixed-route controls.
- Report pairwise specialist complementarity, taskwise uncertainty, and
  whether any route is dominated across all tasks.
- Keep route priors label-free at decision time and do not open untouched
  confirmation.
