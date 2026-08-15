# EXP18 Codex Prompt — Recovery Action-Consequence Coordinator

## Mission

Train and evaluate the actual coordinator on EXP17's validated reference-free recovery candidate sets. Predict Contact risk, Motion/time-to-recovery, Outcome success and Uncertainty from the current state, candidate identity, and first proposed action. Select one candidate before rollout. Compare against the frozen default, deterministic random, fixed best route, heuristics, pairwise/listwise models, uncertainty/abstention and a richer predictor.

## Frozen data

- Candidate run: `exp17_s3_formal_recovery_20260815`, 84 groups, eight routes, 672 rollouts.
- Safe availability 92.86%, decision demand 41.67%, oracle demand recovery 82.86%.
- Candidate outcome labels are exact same-state rollout results. Target future actions remain hidden from selector inputs.
- Complete `(task, demo)` is the cross-validation unit. No candidate rows from a target demo may train its selector.

## Routes to compare

1. frozen default and deterministic random;
2. global and task-specific route priors;
3. compact state+route+first-action ridge/logistic specialists;
4. tree/boosting models for nonlinear task×phase interactions;
5. compact shared-trunk MLP with Contact/Motion/Outcome heads;
6. pairwise preference and listwise/set-rank objectives;
7. lexicographic safety filter plus predicted outcome/time;
8. ensemble uncertainty, lower-confidence score and abstention-to-default;
9. richer trajectory-statistic predictor as a capacity baseline;
10. oracle selector for headroom only.

## Inputs and labels

Allowed selector inputs: current reference-side physical/object/contact/progress state, task, normalized branch time, route configuration, and the candidate's first proposed action/chunk summary available before execution. Forbidden: target-demo future actions, any post-action state, realized retrieval future, candidate outcome, or rollout length.

Labels: safety stop/contact risk, exact success, normalized time-to-success, and a success-first lexicographic utility. Predict axes separately before combining; report calibration and uncertainty.

## Success rule

On complete-demo cross-fitted predictions, the selected route must:

- improve safe success over the frozen default by at least 10 percentage points;
- recover at least 60% of default-demand groups;
- achieve at least 75% of oracle recoverable headroom;
- not increase safety-stop rate over default;
- improve or match default in at least two tasks;
- use no target future or post-action leakage.

Report bootstrap confidence intervals by demo and task. Formal success requires a predeclared conservative route to pass all gates; model fishing is not allowed.

## Ablations

State-only, route-only, action-only and combinations; each consequence head; shared versus separate heads; ridge/tree/MLP; pairwise/listwise; uncertainty penalty; abstention; history/contact/progress; first action versus initial chunk if available; fixed best route; candidate counts 4/6/8; training cohort sensitivity; task-specific performance.

If EXP18 passes, run an output/leakage audit and declare the proposed action-consequence coordination structure successful within the stated development scope. If it fails, EXP19 targets the failed selector axis while retaining EXP17 candidates. Continue automatically until success or EXP62.

