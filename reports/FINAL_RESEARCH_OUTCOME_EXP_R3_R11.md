# Final research outcome: EXP_R3--EXP_R11

The corrected campaign establishes an auditable chain from same-state
candidate instrumentation to frozen independent confirmation:

1. EXP_R3 produced a valid 60-branch, 540-candidate pre-outcome matrix with
   exact matched-zero determinism. Two earlier R3 runs were invalidated by
   code-review findings and retained as evidence rather than mixed into the
   analysis.
2. EXP_R4 showed scalar state/action ranking was strong on the fixed split,
   but EXP_R5 leave-one-demo-out evaluation favored factorized success/unsafe
   heads (top-1 0.933, mAP 0.948).
3. EXP_R6 ablations showed the factorized result was not explained only by
   route identity: removing route identity lowered mAP, and removing action
   chunks lowered both top-1 and mAP.
4. EXP_R7--R9 found good aggregate probability error but weak sparse unsafe
   recall; threshold changes did not improve recall, while the factorized
   unsafe head modestly improved unsafe ranking over a safety-only control.
5. EXP_R10 found complementary route profiles and a modest route-aware
   selection benefit without fitting an outcome-derived route prior.
6. EXP_R11 froze the model before opening the independent demo_28--29 cohort.
   On 24 confirmation branches, frozen selection achieved 0.917 safe success
   versus 0.625 for the fixed D control, with a passed zero gate and no
   future-action leakage.

## Defensible conclusion

Factorized action consequences provide a useful intermediate decision
interface for success-preserving recovery and show cross-demo and independent
confirmation benefit. The evidence is narrower than a universal safety claim:
unsafe support is sparse, safety calibration is task-concentrated, and oracle
headroom remains. The next responsible step is a larger safety-enriched
confirmation cohort, not additional tuning on the consumed confirmation set.
