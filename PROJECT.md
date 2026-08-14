# Decision-Sparse Latent RL for Contact-Rich Manipulation

**Working project document — living research specification**

Last updated: 2026-08-13

This document defines the scientific direction, implementation boundaries, project structure, claim ladder, experiment sequence, and research-log protocol for the Decision-Sparse Latent RL project. It is intentionally a living document. The method and claims may change as evidence accumulates, but every change to the scientific claim must be justified by recorded experimental evidence.

---

## Core scientific story

The central hypothesis is:

> **Manipulation is densely controlled but sparsely decided.**

A robot may need a controller to run at 20–100 Hz, but not every control timestep deserves an independent learned decision. Large portions of a manipulation trajectory consist of free-space approach, transit, retreat, or other motion that can be executed reliably by interpolation, inverse kinematics, trajectory optimization, or a conventional low-level controller. The learned policy should concentrate its capacity and exploration on the relatively small set of transitions where small configuration changes can produce large changes in final task outcome.

These transitions are expected to be especially common around **contact establishment, gripper closure, contact-mode changes, insertion alignment, release, and other dynamics-sensitive events**.

The long-term goal is therefore to replace dense-time RL,

\[
s_0 \rightarrow a_0 \rightarrow s_1 \rightarrow a_1 \rightarrow \cdots \rightarrow s_H
\]

with an event-driven formulation,

\[
s_{\tau_0} \Rightarrow s_{\tau_1} \Rightarrow \cdots \Rightarrow s_{\tau_K},
\qquad K \ll H,
\]

where learned decisions occur only at outcome-sensitive events and the motion between events is delegated to a low-level connector.

The later-stage formulation will place these event-level decisions in a **q-grounded latent space**. A latent action should represent a desired future physical configuration rather than an arbitrary internal neural hidden state:

\[
z_k = E_q(q_{\tau_k}), \qquad
z_{k+1}^{*} \sim \pi_\theta(z_k, c_k),
\]

followed by a physically feasible q-space projection and low-level execution.

The project must not assume this story is true. The first experiment is designed specifically to falsify or support the most basic premise: **are manipulation outcomes actually concentrated in a small subset of timesteps?**

---

## What the project is trying to establish

The strongest eventual scientific statement would be:

> **Decision sparsity is a structural property of contact-rich manipulation: a small fraction of trajectory timesteps carries most of the outcome sensitivity, and restricting learned decisions to those transitions can preserve or improve task success while substantially reducing effective decision horizon and RL interaction cost.**

This is an end-state claim, not the starting claim.

The project must earn this statement through progressively stronger experiments. Early experiments should use weaker language such as **outcome sensitivity is temporally non-uniform under controlled counterfactual interventions**.

---

## Research questions

### RQ-A — Does decision sparsity exist?

Along a successful manipulation trajectory, does a small feasible perturbation to robot configuration have approximately equal effect at all timesteps, or is the effect concentrated into a small number of temporal regions?

Operationally, define a local counterfactual criticality score

\[
C_\epsilon(t)
=
P(Y=1\mid \text{unperturbed continuation at }t)
-
\mathbb{E}_{\delta q \sim \mathcal B_\epsilon}
P(Y=1\mid q_t+\delta q,\text{ matched continuation}),
\]

where \(Y\) is final task success.

If the baseline continuation is deterministically successful, this reduces to

\[
C_\epsilon(t) = 1 - P(Y=1\mid q_t+\delta q).
\]

The initial experiment uses q-space directly. No latent representation is needed to test this question.

### RQ-B — Are critical regions physically meaningful?

If temporal criticality exists, do its peaks align with identifiable physical events such as contact establishment, gripper closure, contact-mode changes, insertion alignment, or release?

This question must be tested after RQ-A. Correlation with contact does not prove RQ-A, and RQ-A does not require contact labels.

### RQ-C — Can dense-time decision making be removed without losing performance?

Given an oracle set of critical regions, can a policy make learned decisions only at those regions while a conventional connector handles the intervening trajectory?

This converts the dense MDP into an event-driven semi-Markov decision process.

### RQ-D — Can the event detector be learned?

Can the critical regions be predicted from observations available at inference time rather than from hindsight perturbation experiments?

### RQ-E — Does a q-grounded latent decision space improve learning?

Can an event-level policy act in a latent configuration space

\[
z = E_q(q)
\]

while maintaining a reliable connection to physically executable configurations?

### RQ-F — Does success/failure visitation improve event-level RL?

Can a success-visitation process reward be defined on event transitions rather than dense low-level steps, and does it improve sample efficiency over sparse outcome reward alone?

---

## Claim ladder

The project should never jump directly to the strongest story. Claims should tighten only when the corresponding evidence exists.

### Claim level A — Temporal non-uniformity

Allowed after Exp1 if counterfactual outcome sensitivity varies substantially across time and this result is robust across demonstrations, perturbation directions, and random seeds.

Safe wording:

> **Manipulation trajectories exhibit strongly non-uniform local outcome sensitivity under q-space counterfactual perturbations.**

Do not yet claim that RL should be sparse.

### Claim level B — Decision sparsity

Allowed only if a small fraction of timesteps consistently explains a large fraction of total measured sensitivity and this concentration is stronger than uniform or time-shuffled controls.

Safe wording:

> **Outcome sensitivity is temporally concentrated, supporting a decision-sparse view of manipulation.**

Do not yet claim that contact is the cause.

### Claim level C — Physical-event concentration

Allowed only after criticality peaks are shown to align with verified physical events across multiple task families.

Safe wording:

> **Decision-critical regions are concentrated near contact-sensitive and dynamics-sensitive transitions.**

Do not generalize from one grasp task to manipulation as a whole.

### Claim level D — Sparse decisions preserve control performance

Allowed only after an event-driven policy or oracle event policy achieves comparable task success to a dense-time policy while using substantially fewer learned decision points.

Safe wording:

> **Dense low-level control can be retained while learned decision making is temporally sparsified.**

### Claim level E — Learned event detection

Allowed only when event timing no longer depends on oracle demonstration annotations or post-hoc perturbation sweeps.

Safe wording:

> **Decision-critical events can be predicted online from available robot observations.**

### Claim level F — Q-grounded latent RL

Allowed only after the latent mapping and q-space projection have been quantitatively validated.

Safe wording:

> **Event-level decisions can be optimized in a q-grounded latent configuration space while remaining physically executable.**

### Claim level G — General manipulation principle

Allowed only after the phenomenon and algorithm survive multiple task families, multiple initial states, multiple objects, and preferably at least one real-robot validation.

Safe wording:

> **Decision sparsity is a reusable structural prior for contact-rich manipulation.**

This strongest claim should be removed if cross-task or real-world evidence is not available.

---

## Project sequence

### Phase 0 — Reproducibility and simulator audit

The first engineering task is to establish a reproducible LIBERO installation, inspect the exact dataset structure, enumerate available tasks, verify exact package versions, and verify deterministic state restoration and demonstration replay.

No latent model should be installed during this phase unless required for an existing dataset tool.

### Phase 1 — Exp1: Causal decision-criticality mapping in q-space

Use successful LIBERO demonstrations and exact MuJoCo simulator states to branch counterfactual rollouts from many time indices.

At time \(t\), restore the full simulator state, perturb only the Panda arm joint configuration by a controlled small amount, forward the simulator, and continue using the same recorded downstream demonstration actions.

Measure final success probability as a function of perturbation time and perturbation size.

This phase asks only whether trajectory timesteps have unequal causal sensitivity under a matched continuation protocol.

### Phase 2 — Physical-event alignment

Add reliable physical event metadata.

Candidate event signals include gripper closure, robot-object contact, contact-count changes, contact-pair changes, and task-state transitions. The exact MuJoCo geom/body identifiers must be discovered from the runtime model and logged. Do not hardcode guessed names.

Test whether criticality peaks concentrate near these events more strongly than matched non-event windows.

### Phase 3 — Event-driven control without latent RL

Construct an oracle event-driven controller first.

The purpose is to isolate temporal abstraction from representation learning. If oracle event sparsification does not preserve performance, adding latent representations will not solve the core problem.

Compare dense decision making, fixed action chunking, uniform keyframes, semantic keyframes, and oracle critical events under matched environment interaction budgets.

### Phase 4 — Q-grounded representation

Only after Phase 3 succeeds, introduce

\[
E_q:q\rightarrow z.
\]

The representation may come from a pretrained robot model or from a separately validated configuration representation. The mapping must be empirically checked for local smoothness, neighborhood preservation, recoverability, and physical ambiguity.

A latent should never be assumed to correspond to a valid configuration.

Implement a q-space projection

\[
q^*
=
\arg\min_{q\in \mathcal Q_{\mathrm{feasible}}}
\|E_q(q)-z^*\|^2
+
\alpha\|q-q_{\mathrm{current}}\|^2
\]

subject to explicitly implemented joint and collision constraints.

### Phase 5 — Event-level RL

The event-driven state transition becomes a semi-Markov decision process.

The actor selects a next event-level goal. The connector executes the intermediate dense control. Duration \(\Delta_k\) must be recorded because event transitions have unequal lengths.

Compare dense RL with event-driven RL under matched environment interaction budgets.

### Phase 6 — Event-level success visitation matching

Adapt the idea of Success Visitation Matching to event transitions.

A possible event tuple is

\[
\xi_k =
(z_k, z^*_{k+1}, z_{k+1}, \Delta_k).
\]

Successful and failed event transitions form positive and negative visitation sets. A discriminator provides a process reward at the event level.

This is a later-stage module, not part of Exp1.

### Phase 7 — Real-robot validation

Only reproduce the smallest established phenomenon first: perturbation sensitivity around a small number of physically meaningful events.

Do not begin real-robot RL until the simulation result is stable.

---

## Repository strategy

### Repositories to clone now

Create the project as its own repository. External projects should remain under `third_party/` and should not be edited unless a documented patch is unavoidable.

Use LIBERO as the first benchmark:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
git -C third_party/LIBERO rev-parse HEAD
```

Record the exact commit SHA in the research log before running experiments.

LIBERO currently pins `robosuite==1.4.0` in its `requirements.txt`. For source inspection only, it is useful to clone robosuite and checkout the exact tag:

```bash
git clone https://github.com/ARISE-Initiative/robosuite.git third_party/robosuite-src
git -C third_party/robosuite-src checkout v1.4.0
git -C third_party/robosuite-src rev-parse HEAD
```

Do **not** install the cloned robosuite source in editable mode unless there is a documented reason. LIBERO's own dependency specification should define the runtime environment for Exp1.

### Repositories to defer

Do not clone OpenPI for Exp1.

OpenPI becomes relevant only when the project reaches q-grounded latent representation experiments. At that point use the official repository:

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git third_party/openpi
```

Do not clone LeRobot for Exp1. It can become useful later for common policy infrastructure, RL baselines, dataset conversion, or real-robot integration.

Do not attempt to clone a Success Visitation Matching implementation yet. As checked on 2026-08-13, the official SVM project page still labels the code as **coming soon**. If code is released later, record the release date and exact commit before integrating it. Until then, any SVM implementation must be derived from the paper and kept as an independent project module.

CALVIN should be treated as a later cross-benchmark test rather than the first causal experiment. Its public dataset is useful for long continuous robot trajectories, but Exp1 benefits more directly from LIBERO's stored full MuJoCo states and explicit state-restoration API.

---

## Verified LIBERO interfaces relevant to Exp1

The current LIBERO repository exposes the following methods in `libero/libero/envs/env_wrapper.py`:

```text
ControlEnv.get_sim_state()
ControlEnv.set_state(mujoco_state)
ControlEnv.set_init_state(init_state)
ControlEnv.regenerate_obs_from_state(mujoco_state)
ControlEnv.check_success()
```

`regenerate_obs_from_state()` sets a flattened MuJoCo state, calls `sim.forward()`, updates task state and observables, and returns regenerated observations.

The dataset construction script stores, per episode:

```text
states
actions
robot_states
rewards
dones
obs/joint_states
obs/gripper_states
obs/ee_states
```

During dataset construction, the code reads exact observation keys:

```text
robot0_joint_pos
robot0_gripper_qpos
robot0_eef_pos
robot0_eef_quat
```

The project must still verify these identifiers at runtime before using them. If a version mismatch changes them, stop and update the experiment specification rather than guessing replacements.

In robosuite v1.4.0, the robot implementation builds arm joint qpos indices in:

```text
self._ref_joint_pos_indexes
```

and arm joint qvel indices in:

```text
self._ref_joint_vel_indexes
```

These are private implementation attributes. Exp1 may use them only after verifying the installed runtime is robosuite v1.4.0 and asserting the attributes exist. If the installed package differs, inspect its source and document the exact alternative before changing code.

---

## Recommended project layout

```text
decision-sparse-latent-rl/
├── PROJECT.md
├── README.md
├── pyproject.toml
├── .gitignore
├── third_party/
│   ├── LIBERO/
│   └── robosuite-src/
├── src/
│   └── decision_sparse_rl/
│       ├── envs/
│       ├── interventions/
│       ├── metrics/
│       ├── logging/
│       └── utils/
├── scripts/
│   └── exp1/
├── experiments/
│   └── exp1_decision_sparsity/
│       ├── EXP1.md
│       ├── configs/
│       ├── manifests/
│       └── README.md
├── research_log/
│   ├── README.md
│   └── YYYY-MM-DD.md
├── runs/
│   └── <run_id>/
├── results/
│   └── exp1/
└── paper/
    ├── claims.md
    ├── figures/
    └── evidence_table.md
```

The `third_party/` repositories should be excluded from the main project Git history or added as submodules only after the workflow is stable.

---

## Research-log protocol

**Every experiment, including failed setup attempts, must produce a research-log entry.**

The log is append-only. Do not delete an old interpretation because a later experiment contradicted it.

Every meaningful run must record:

```text
Timestamp
Research question
Hypothesis
Run ID
Git commit SHA of this project
Git commit SHA of every external repository used
Python version
CUDA / MuJoCo / robosuite / LIBERO versions
Machine / GPU identifier
Dataset path or dataset identifier
Task suite
Task ID
Task name
Demonstration IDs
Random seeds
Exact command
Exact config file
Code changes since previous run
Expected result before seeing the result
Observed metrics
Generated artifacts / plots
Unexpected behavior
Failed checks
Interpretation
Alternative explanations
Claim impact
Next experiment
```

Do not write only “worked” or “failed.” Preserve enough information that the run can be reconstructed later.

Every run directory must contain at least:

```text
config_resolved.yaml
command.txt
environment.txt
git_state.txt
stdout.log
stderr.log
metrics.json
artifacts/
```

If the working tree is dirty, save:

```text
git_diff.patch
```

before the experiment begins.

The `paper/evidence_table.md` file should connect every paper-level statement to the experiments that support or contradict it.

Example:

```text
Claim:
Outcome sensitivity is temporally concentrated.

Supporting runs:
exp1_taskA_seed0_...
exp1_taskA_seed1_...
exp1_taskB_seed0_...

Contradicting runs:
...

Current status:
supported / mixed / unsupported

Allowed paper wording:
...
```

---

## Rules for Codex

Codex is an implementation and research assistant, not an authority on the scientific conclusion.

Before editing code, Codex must read:

```text
PROJECT.md
experiments/exp1_decision_sparsity/EXP1.md
the latest research_log entry
```

Codex must never guess a repository path, API name, observation key, simulator field, joint index, dataset schema, task identifier, or configuration key.

If an identifier is needed, Codex must inspect the exact checked-out source, runtime object, dataset file, log, or configuration file first. If the value cannot be established from available evidence, Codex must stop that specific subtask and report exactly what information is missing.

Codex must not silently upgrade dependencies.

Codex must not edit files under `third_party/` unless explicitly instructed. Any required patch must first be documented in the research log and kept as a patch file.

Codex must run smoke tests before long experiments.

Codex must not implement later phases before the current phase's gate is evaluated.

Codex must preserve negative results.

Codex must keep experiment configs immutable after a run starts. A changed configuration receives a new run ID.

Codex must report all failed assertions and replay divergence.

Codex must not reinterpret a failed experiment as supporting the hypothesis.

---

## Master Codex prompt

Use the following as the initial project-level prompt for Codex.

```text
You are the research implementation engineer for the project "Decision-Sparse Latent RL for Contact-Rich Manipulation."

Your first responsibility is reproducibility and falsifiability, not maximizing positive results.

Read PROJECT.md and experiments/exp1_decision_sparsity/EXP1.md in full before making changes. Also read the latest entry under research_log/ if one exists.

The current scientific hypothesis is that manipulation is densely controlled but sparsely decided: only a small subset of trajectory timesteps may have high causal sensitivity to robot configuration. This hypothesis is NOT assumed true.

For now, work ONLY on Exp1. Do not implement latent RL, OpenPI integration, SVM, reward learning, event-level RL, or real-robot code.

Use the official LIBERO repository as the simulation benchmark. Treat external repositories as read-only dependencies under third_party/.

Critical rules:
- Never guess API names, observation keys, task names, file paths, joint indexes, state-vector layouts, or configuration fields.
- Before using any such identifier, inspect the exact checked-out source code, installed runtime object, HDF5 structure, or configuration file and record what you found.
- Do not silently upgrade or replace LIBERO dependencies.
- Record exact repository commit SHAs and package versions.
- Every setup attempt and every experiment run must append a research_log entry.
- Every run must have a unique run_id and preserve its resolved config, exact command, environment/version information, stdout/stderr, metrics, and artifacts.
- Never overwrite results from an existing run.
- Preserve negative and contradictory results.
- Do not change scientific claims based on intuition. Update claim status only after a completed experiment and record the supporting evidence.
- Keep third_party repositories unmodified. If a patch becomes necessary, stop and explain why before patching.
- Prefer small, testable modules with deterministic unit or integration tests.
- Make atomic Git commits with descriptive messages.

Start with the Exp1 environment/replay audit:
1. Create the project directory structure specified in PROJECT.md if it does not exist.
2. Verify the LIBERO checkout and record its exact commit SHA.
3. Verify the installed Python, MuJoCo, robosuite, LIBERO, numpy, h5py, and CUDA versions.
4. Enumerate LIBERO task suites and tasks programmatically and save a manifest with exact suite names, task IDs, task names, and language instructions.
5. Inspect a demonstration HDF5 file programmatically and save its exact group/dataset schema, shapes, and dtypes.
6. Verify the exact simulator-state restoration API from the checked-out LIBERO source.
7. Verify the exact arm joint qpos indices from the installed robosuite source/runtime; do not assume them.
8. Implement a deterministic replay smoke test before any perturbation experiment.
9. Stop after producing the replay-audit report if replay/state restoration is not sufficiently reproducible.

After each completed step, append to research_log with:
- what was attempted,
- exact command,
- result,
- anomalies,
- interpretation,
- next action.

At the end of each Codex session, summarize:
- files changed,
- commands executed,
- tests passed/failed,
- experiment results,
- unresolved issues,
- current claim status,
- the single next experiment that should be run.

Do not proceed to the q-perturbation sweep until the deterministic replay gate in EXP1.md is satisfied.
```

---

## What success would mean scientifically

If only Exp1 succeeds, the project can already answer:

> **Are all manipulation timesteps equally sensitive to configuration errors?**

If Exp1 and event alignment succeed, it can answer:

> **Where in a manipulation trajectory does configuration precision matter most, and are those regions associated with physical interaction events?**

If event-driven control succeeds, it can answer:

> **Does a robot-learning policy need to make a learned decision at every control step?**

If latent RL succeeds, it can answer:

> **Can the reduced set of task-critical decisions be represented and optimized as physically grounded latent configuration transitions?**

If success-visitation RL also succeeds, it can answer:

> **Can sparse task outcomes be converted into useful event-level process rewards without returning to dense-time policy learning?**

The final paper should be built around the strongest question that the evidence actually answers, even if that is narrower than the original proposal.

