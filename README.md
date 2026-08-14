# Decision-Sparse Latent Reinforcement Learning

This repository implements the staged experiments specified in [PROJECT.md](PROJECT.md).
The active scope is **EXP1 only**: deterministic LIBERO replay and causal q-space
decision-criticality mapping. Later latent-RL phases are intentionally out of scope.

## Current stage

Stage E3 completed with a failed deterministic-replay hard gate. E4, E5, and the
q-perturbation sweep are blocked pending simulator/data reconciliation. See
[experiments/exp1_decision_sparsity/EXP1.md](experiments/exp1_decision_sparsity/EXP1.md)
and the append-only [research log](research_log/README.md).

Detailed completed-experiment report: [exp1_report.md](exp1_report.md).

Each completed experiment must produce a detailed root-level `exp{id}_report.md`
before it is considered handed off.

## Safety gates

1. Audit the exact environment, repositories, task API, and dataset schema.
2. Pass deterministic demonstration replay.
3. Pass arbitrary-time branch restoration.
4. Only then run the smallest arm-q perturbation smoke test.

No full sweep is launched automatically.
