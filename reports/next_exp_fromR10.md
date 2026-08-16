# Next experiment after EXP_R10: EXP_R11 freeze and independent confirmation

The development mainline now has: a matched-state candidate substrate,
cross-demo robustness, factorized ablations, calibration stress, safety
ranking, and route complementarity. Freeze the factorized protocol before
opening the untouched independent confirmation set.

The confirmation run must use the frozen pre-action feature schema, fixed
model/seed/split policy, no threshold retuning, no route-prior fitting, and
the same zero-determinism gate. It must produce a separate immutable run and
report only pre-registered aggregate and taskwise metrics. If confirmation
fails or calibration is unsafe, preserve the negative result and do not tune
on it.
