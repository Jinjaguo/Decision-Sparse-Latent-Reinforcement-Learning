# Next experiment after EXP_R4: EXP_R5 leave-one-demo-out robustness

R4 shows a promising scalar state/action baseline but no factorization gain.
Before expanding the model or opening the untouched confirmation set, test
whether this ranking survives demo-level distribution changes.

## Plan

- Run deterministic leave-one-demo-out evaluation over the ten development
  demos, keeping all candidates from the held-out demo entirely out of model
  fitting and feature normalization.
- Compare retrieval-only, scalar, and factorized scores with the same
  tie-aware top-k and mAP metrics.
- Report per-task and per-demo results, bootstrap over branches, and undefined
  metrics when a fold lacks positive or negative support.
- Keep the post-action-field assertion, hash uniqueness assertion, and target
  demo exclusion active.

## Decision rule

If the scalar advantage is stable and factorization remains neutral, freeze
the scalar baseline and investigate whether a factorized inductive bias helps
only on safety-heavy or force-sensitive subsets. If performance collapses
under leave-one-demo-out evaluation, stop model expansion and treat R4 as
overfitting evidence.
