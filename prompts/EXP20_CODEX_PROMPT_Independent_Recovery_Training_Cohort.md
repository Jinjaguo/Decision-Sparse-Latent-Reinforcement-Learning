# EXP20 Codex Prompt — Independent Recovery Training Cohort

## Mission

Collect a larger selector-training cohort by executing the frozen EXP17 candidate policies on EXP8 successful demos with strict leave-one-demo-out retrieval. Train consequence selectors only on these new training outcomes, then evaluate once on the untouched EXP17 formal candidate outcomes.

## Required routes

Collect all eight EXP17 routes; exclude the target training demo from neighbors, feature scales and source trajectories. Compare task×branch-rank route priors, full initial-chunk descriptors, branch-vector kNN, ridge interactions, compact MLP, pairwise rescue and conservative uncertainty ensembles. Materialize the complete first proposed 10-step chunk, retrieval distances/source IDs, neighbor disagreement and route configuration as pre-action features.

## Cohorts

- Training candidate outcomes: EXP8 demos, four corrected-D branches/demo, leave-one-demo-out candidate library.
- Calibration: EXP15/16/17 calibration may tune thresholds only.
- Test: EXP17 formal demos 43–49; no formal outcome may train or tune the EXP20 primary selector.

## Success rule

Unchanged from EXP18/19: +10 safe-success points, >=60% demand recovery, >=75% oracle headroom capture, safety no worse than default, two tasks non-inferior, zero leakage. Freeze primary model before test outcomes.

If passed, complete final output/leakage/bootstrap audit and declare the action-consequence coordination structure successful in development scope. Otherwise EXP21 changes model/candidate descriptors or collects a larger prospective policy cohort. Continue until success or EXP62.

