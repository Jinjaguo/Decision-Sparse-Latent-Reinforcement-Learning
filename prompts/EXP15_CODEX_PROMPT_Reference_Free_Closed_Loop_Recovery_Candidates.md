# EXP15 Codex Prompt — Reference-Free Closed-Loop Recovery Candidates

## Mission

Repair the evaluation and proposal architecture exposed by EXP12–14. The successful target-demo continuation is an expert upper bound and often already maximizes the success-first quality. It must no longer be a selectable default that every candidate is required to beat. EXP15 builds reference-free candidate macro-policies from other successful demos, runs them with feedback/replanning, and asks whether candidate sets contain safe recovery choices relative to a realistic non-oracle default.

Do not train the final consequence coordinator yet. First establish cross-task decision demand and recoverability.

## Frozen evidence and proposal revision

- EXP12: only 10/84 local candidates beat the expert nominal by 0.05, all Stove.
- EXP13 formal: 12/84, all Stove, despite eight local action-space families and 100% candidate validity.
- EXP14 calibration: object-centric open-loop proposals still gave Drawer 0/8 and Bowl 0/8, while terminal failure rose to 46.4% even without clipping.
- For Drawer/Bowl, a changed 10-step state followed by the original open-loop suffix is frequently inconsistent with the new contact/grasp state.

Therefore EXP15 changes two linked assumptions:

1. the target-demo future remains an evaluation oracle/quality ceiling, never an available candidate or proposal input;
2. candidates are feedback macro-policies that emit/replan short chunks from observed state, rather than fixed chunks followed blindly by the target suffix.

This is a proposal correction based on accumulated evidence, not a relaxation of a failed threshold.

## Primary questions

1. Can leave-demo-out feedback candidates recover exact task success across Drawer, Bowl, and Stove without seeing the target future actions?
2. Do candidate sets create genuine decision demand: a frozen default fails or is materially worse, while another candidate safely recovers?
3. Are consequence cards sufficient to describe the resulting spread for a later selector?

## Candidate interface

Every candidate implements the same callable interface:

`candidate(observed physical/object/contact state, recent history) -> next 10-step action chunk`

Execute the first short chunk, observe the new state, and replan until success, a frozen horizon, or a safety stop. Exact gripper signs and OSC limits remain enforced. Candidate identity and parameters remain fixed during a rollout; target outcomes never update the candidate.

## Routes to implement and compare

### R1 — Object-centric nearest-state behavior policy

Index all boundaries from independent successful training demos. Compare physical-state, EEF-to-object, object-to-goal, contact-aware, and task-progress distances. At each replan retrieve leave-demo-out neighbors and emit their next action chunks.

### R2 — Monotone trajectory retrieval

Retrieve a source demo and maintain a monotone source index so the candidate cannot jump backward between unrelated phases. Compare nearest trajectory, diverse top-k trajectories, and soft alignment. Retarget translation commands using current-versus-source object-frame error.

### R3 — Weighted local behavior cloning

Fit compact ridge/MLP/mixture regressors on independent training boundaries, using task, physical state, object geometry, contact flags, progress and short history. Predict the next 10 actions or temporal basis coefficients. Use whole-demo splits and report action imitation only as a diagnostic, never as task success.

### R4 — Feedback task skills

Implement state-triggered approach/contact/manipulate/settle skills. Learn canonical EEF-to-object waypoints and transition guards from training demos, then close the loop on current geometry. Drawer must pull only after handle acquisition; Bowl must grasp, lift, transport, lower and release; Stove must approach and actuate the button.

### R5 — Recovery and perturbation-conditioned policies

Create candidates specialized for stalled progress, lost contact, overshoot, object-goal error and unsafe-force risk. Disturbance labels are generated on training rollouts only. Compare specialist routing with a shared recovery policy.

### R6 — Short-horizon learned consequence search

Build a finite action primitive pool at each replan and use training-fold consequence prediction to choose several diverse chunks. Compare expected quality, lower-confidence-bound, lexicographic safety, and abstaining search. The simulator target outcome never enters search.

### R7 — Policy/library ensemble candidates

Treat independently trained/retrieved policies as separate candidates. Include deterministic default, diverse library members, conservative policy, progress-seeking policy and risk-averse policy. This source-level diversity is the candidate set for later consequence coordination.

### R8 — Restricted hierarchical compositions

Compose at most two audited components, such as monotone retrieval plus feedback correction or behavior clone plus recovery specialist. Compare no feedback, one replan, and repeated replanning to isolate the value of closed-loop execution.

## Calibration and formal cohorts

Use independent EXP8 references (roughly demos 30/31–40) for all fitting and libraries. Calibration may use episodes 41–42 for route authorization. Formal development uses episodes 43–49 with all target-demo future actions hidden from candidates. Reuse exact corrected-D branch states, but evaluate full feedback rollout from each branch.

Freeze candidate routes, distance scales, horizons, safety stops, default policy, and all thresholds before target outcomes.

## Evaluation baselines

- Expert target continuation: evaluation-only upper bound, never selectable.
- Frozen non-oracle default: one predeclared reference-free policy.
- Random source candidate.
- Open-loop version of the same policy/library candidate.
- No-replan, one-replan, and repeated-replan variants.

## Primary metrics

- exact task success and time-to-success;
- reference-relative quality gap, but not “expert improvement” as a solvability requirement;
- safe-candidate availability per group;
- decision-demand rate: default failure/material deficit with another safe recovery;
- oracle recovery rate among selectable candidates;
- candidate validity, clipping, safety stops, force/contact tails;
- task/demo/phase replication;
- candidate source contribution and diversity;
- target-future leakage audit.

## Success rule

The recovery-candidate module passes only if:

- at least 70% of all formal groups contain a selectable candidate with exact success and bounded safety cost;
- at least two of three tasks individually reach 60% safe-candidate availability;
- at least 30% of all groups exhibit decision demand relative to the frozen non-oracle default;
- among demand groups, selectable oracle recovery is at least 60%;
- at least 90% of proposed chunks are executable without clipping/non-finite state;
- target-demo future actions are absent from every candidate input, retrieval index, fit, and search update.

Report expert-relative quality honestly, but do not require a non-oracle policy to outperform the expert demonstration.

## Required ablations

- R1–R8 separately;
- expert suffix versus hidden expert suffix;
- open-loop versus one-replan versus repeated-replan;
- physical versus object versus contact/progress context;
- nearest-one versus weighted-k versus monotone retrieval;
- shared versus task-specific policy;
- direct actions versus basis/chunk prediction;
- recovery specialists on/off;
- uncertainty/risk penalty on/off;
- chunk length 5/10/20 and rollout horizon sensitivity;
- candidate count 4/8/12;
- default choice sensitivity, while the primary default remains frozen.

## Outputs and next step

Create `experiments/exp15_closed_loop_recovery/`, immutable runs, tests, leakage manifests, raw trajectories, safety audits, `reports/exp15_report.md`, `reports/next_exp_from15.md`, and the next multi-route prompt.

If EXP15 passes, EXP16 trains the consequence coordinator and compares it against the frozen default, random, heuristic, richer-future and oracle selectors on the same demand groups. If candidate availability fails in one task, EXP16 targets that task with a stronger policy source or longer-horizon planner. If all reference-free policies fail, EXP16 imports or trains a successful external policy under a separate data/provenance audit. Continue automatically; stop only when the complete project succeeds or EXP62 has been completed and the next experiment would be EXP63.
