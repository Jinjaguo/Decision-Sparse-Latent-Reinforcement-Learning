# EXP22 Codex Prompt — Action-Consequence Progress State Machine

## Mission

EXP21 showed that sequential fallback improves aggregate safe success, but fixed-time switching captures only 41.4% of oracle headroom. EXP22 must coordinate action chunks from their predicted task consequences and switch only when observed consequences indicate stagnation, regression, or approaching force risk. Target-demo future actions remain inaccessible to every candidate and coordinator.

## Required implementation routes

Implement and compare several genuinely different mechanisms, not threshold-only variants:

1. Goal-consequence retrieval: among locally plausible independent-demo chunks, score the recorded successor state against that demo's terminal goal state, then aggregate the best consequences.
2. Progress-consequence retrieval: trade off current-state compatibility with normalized demonstrated phase advancement under monotone memory.
3. Response-alignment retrieval: rank empirical state deltas by alignment with the current-to-goal direction and use a robust medoid action.
4. Online progress state machines: monitor retrieved phase, task-object movement, contact transitions, action response, and force margin; switch among complementary mechanisms when progress stalls.
5. Diverse periodic portfolio: cycle goal-directed, smooth, response-aligned, and conservative modes to test whether broad coverage matters more than the learned stall trigger.
6. Task-specialized state machine: allow Drawer, Bowl, and Stove to use different mechanism order, while freezing the order from calibration only.
7. Ablate fixed consequence policies, online switching, task specialization, force preemption, progress memory, robust aggregation, horizon, and candidate diversity.

Search over consequence weights, candidate shortlist sizes, stall windows, minimum dwell, mode order, 220–280 step budgets, and two-to-four mode portfolios during calibration. Select using safe success, default-demand recovery, oracle-headroom capture proxy, safety stops, per-task robustness, switching frequency, and validity. Prefer the smallest structure within one calibration success of the best.

## Formal gate

Freeze one primary before formal outcomes. On the untouched same-state formal branches it must improve safe success by at least 10 points over the 140-step default, recover at least 60% of default-demand groups, capture at least 75% of the EXP17 oracle headroom, not worsen safety-stop rate, and be non-inferior on at least two tasks. Audit source hashes, target-future isolation, complete denominators, finite states, and deterministic protocol lock.

If the gate passes, run full tests and a final leakage/reproducibility audit. If it fails, write the evidence, generate a broad EXP23 prompt from the observed bottleneck, and continue autonomously until success or EXP62.

