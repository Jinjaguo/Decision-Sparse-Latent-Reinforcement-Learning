# Next experiment after EXP_R5: EXP_R6 factorized ablation and safety stress

R5 supports factorized heads under cross-demo evaluation, while R4 did not.
EXP_R6 should isolate the source of this reversal without opening untouched
confirmation data.

## Plan

- Keep the same ten leave-one-demo-out folds and compare: scalar utility,
  factorized success/unsafe heads, success-only head, safety-only head, and a
  factorized model without route identity.
- Report per-task, per-route, and safety-heavy subset metrics, using only
  pre-action inputs and branch bootstrap intervals.
- Preserve all zero-step and unsafe labels; report undefined subsets instead
  of silently dropping them.
- Add permutation tests for route identity and action chunks to distinguish
  factorization from generic capacity.

## Decision rule

If factorized gains persist without route identity and concentrate on unsafe
or force-sensitive candidates, freeze the factorized protocol for later
confirmation. If gains disappear under the ablations, treat R5 as a capacity
or route-identity artifact and stop expanding the factorized claim.
