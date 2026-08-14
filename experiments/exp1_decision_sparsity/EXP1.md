# EXP1 — Causal Decision-Sparsity Mapping in LIBERO

**Experiment status:** first experiment to implement  
**Purpose:** test the core hypothesis before implementing latent RL  
**Benchmark:** LIBERO  
**Primary variable:** time-indexed q-space counterfactual outcome sensitivity

---

## Scientific question

The first experiment asks:

> **Along a successful manipulation trajectory, are all timesteps equally important to final success, or is outcome sensitivity concentrated in a small subset of timesteps?**

This experiment deliberately avoids latent representations and RL.

If the basic decision-sparsity phenomenon does not appear in q-space under controlled counterfactual testing, there is no justification for building a more complicated latent RL system around it.

---

## Exp1 claim boundary

The first version of Exp1 measures sensitivity under a **matched demonstration continuation**.

At a selected trajectory timestep \(t\):

\[
x_t \rightarrow q_t+\delta q
\]

is introduced while preserving the rest of the simulator state as closely as possible, and the remaining recorded demonstration actions are replayed.

Therefore the first valid claim is:

> **Local q perturbations have time-dependent effects on success under matched demonstration continuation.**

This protocol does **not** initially prove that a closed-loop policy would exhibit the same sensitivity.

If the result is positive, a later Exp1 extension must repeat the intervention under a closed-loop policy or dynamically realized perturbation before using stronger causal-control language.

---

## Why LIBERO is the first benchmark

LIBERO demonstration data and environment wrappers provide exactly the capabilities needed for causal branching.

The current repository stores full flattened MuJoCo simulator states in the episode dataset and provides environment methods that can restore those states.

Verified methods in `libero/libero/envs/env_wrapper.py` include:

```text
get_sim_state()
set_state(mujoco_state)
set_init_state(init_state)
regenerate_obs_from_state(mujoco_state)
check_success()
```

The current dataset-generation script stores:

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

and accesses the following runtime observation keys:

```text
robot0_joint_pos
robot0_gripper_qpos
robot0_eef_pos
robot0_eef_quat
```

These names must be re-verified against the exact checkout/runtime before use.

---

## Experiment overview

For each successful demonstration:

\[
\tau =
(x_0,a_0,x_1,a_1,\ldots,x_T),
\]

choose a set of branch times \(t\).

For each \(t\), restore the exact recorded simulator state \(x_t\).

Generate a feasible small arm-joint perturbation

\[
q_t' = q_t + \delta q.
\]

Insert \(q_t'\) into the simulator, call the required forward/update operations, then replay the recorded actions from \(a_t\) onward.

Repeat this with multiple perturbation directions and magnitudes.

Record terminal task success.

Define

\[
C_\epsilon(t)
=
P(Y=1\mid x_t,\text{baseline continuation})
-
\mathbb E_{\delta q}
P(Y=1\mid x_t+\delta q,\text{same continuation}).
\]

For a deterministic successful baseline,

\[
C_\epsilon(t)
=
1-P(Y=1\mid x_t+\delta q).
\]

Plot \(C_\epsilon(t)\) over the trajectory.

The key empirical question is whether the curve is approximately flat or contains sparse high-sensitivity regions.

---

## Stage E0 — Environment and repository audit

Before running any scientific experiment, create a reproducibility record.

The audit must record:

```text
Project Git SHA
LIBERO Git SHA
robosuite version
robosuite source location
MuJoCo version
Python version
numpy version
h5py version
CUDA version
GPU model
operating system
LIBERO dataset root
```

Do not continue if the runtime robosuite version differs from LIBERO's expected dependency without explicitly documenting and resolving the mismatch.

LIBERO's current `requirements.txt` pins:

```text
robosuite==1.4.0
```

For source reference, robosuite v1.4.0 defines robot arm joint qpos indices in:

```text
self._ref_joint_pos_indexes
```

and qvel indices in:

```text
self._ref_joint_vel_indexes
```

Because these are private attributes, runtime assertions are mandatory before use.

---

## Stage E1 — Enumerate tasks; do not guess task identifiers

Implement:

```text
scripts/exp1/enumerate_tasks.py
```

The script must programmatically query LIBERO's benchmark API and save:

```text
experiments/exp1_decision_sparsity/manifests/tasks.json
```

For every available task used by Exp1, record at least:

```text
suite
task_id
task.name
task.language
task.problem_folder
task.bddl_file
```

Do not type task names manually into experiment code before this manifest exists.

Initial task selection should span different physical structures rather than maximizing task count.

The desired categories are:

```text
pick / place or grasp-dependent task
articulated-object task such as drawer manipulation
contact / switch / stove-like interaction task
```

Select exact tasks only from the generated manifest. Record the selection and rationale in the research log before looking at perturbation results.

---

## Stage E2 — Audit demonstration files

Implement:

```text
scripts/exp1/audit_dataset.py
```

For each selected task, inspect the exact HDF5 file and save a schema manifest.

Record:

```text
all groups and dataset names
shape
dtype
episode count
episode lengths
presence of states
presence of actions
presence of proprioceptive observations
metadata attributes
```

The script must not assume the HDF5 layout beyond what it verifies at runtime.

Save the result as:

```text
experiments/exp1_decision_sparsity/manifests/dataset_schema_<task>.json
```

---

## Stage E3 — Deterministic replay audit

This stage is a hard gate.

Implement:

```text
scripts/exp1/replay_demo.py
```

For a sampled demonstration:

1. Restore the recorded initial simulator state.
2. Replay the recorded action sequence.
3. After every action, compare the current flattened simulator state with the corresponding recorded state.
4. Record the state-space error at every timestep.
5. Record final task success using the environment's verified success checker.
6. Repeat across multiple demonstrations and initial conditions.

Do not use only a binary pass/fail. Save the complete replay divergence curve.

LIBERO's own dataset-generation code computes:

\[
\|x^{recorded}_{j+1}-x^{replay}_{j+1}\|_2
\]

and emits a warning when it exceeds `0.01`. Exp1 should record the raw distribution rather than blindly adopting that threshold.

The research log must contain:

```text
median replay error
95th percentile replay error
maximum replay error
final success rate
failure examples
whether divergence accumulates over time
```

### Replay gate

Do not proceed to perturbation experiments until:

- unperturbed demonstrations replay with high success;
- restoring the initial state is reproducible;
- state divergence is understood and documented;
- any systematic action/state indexing issue has been resolved.

If replay is unreliable, Exp1 stops here and the next task is simulator/data reconciliation.

---

## Stage E4 — Arbitrary-time branch restoration audit

Implement:

```text
scripts/exp1/validate_branch_restore.py
```

The goal is to verify that branching directly from a recorded intermediate state is equivalent, within measured tolerance, to reaching that state by replaying from the beginning.

For sampled timesteps \(t\):

**Path A**

```text
restore initial state
replay actions 0 ... t-1
take action a_t
```

**Path B**

```text
restore recorded state[t]
take action a_t
```

Compare:

```text
simulator state after a_t
robot joint positions
gripper state
end-effector pose
task success state
next observation
```

Save the discrepancy distribution.

If direct branch restoration produces systematically different dynamics, do not run the counterfactual sweep until the cause is understood.

---

## Stage E5 — Exact q-index verification

Do not infer joint positions from the flattened MuJoCo state by manually guessing offsets.

At runtime, inspect:

```text
env.robots[0]
```

and verify the installed robosuite source.

For robosuite v1.4.0, the verified arm joint qpos index attribute is:

```text
env.robots[0]._ref_joint_pos_indexes
```

Before using it, assert:

```text
runtime robosuite version == expected version
attribute exists
index count equals Panda arm DoF
values refer to qpos entries whose runtime values match the observed arm joint positions
```

Record the verified index vector in the run metadata.

For Exp1 v1, perturb only Panda arm joints.

Do not perturb gripper joints until arm-only experiments are stable.

---

## Stage E6 — Perturbation construction

The intervention must be local and reproducible.

Let the arm configuration be:

\[
q_t\in\mathbb R^7.
\]

Sample a random direction:

\[
u \sim \mathcal N(0,I),
\qquad
u \leftarrow \frac{u}{\|u\|_2}.
\]

Scale the perturbation relative to each joint's legal range rather than assuming a universal angular scale.

A normalized perturbation may be constructed from the joint-range vector \(r\):

\[
\delta q =
\epsilon \,
\mathrm{diag}(r)\,
u.
\]

The initial perturbation magnitude grid should be defined in a config file and frozen before the full sweep.

A reasonable pilot is to use several small normalized magnitudes covering roughly sub-percent to a few percent of joint range. The exact values selected for the full run must be recorded before inspecting the full experiment.

Reject or clip interventions that violate verified joint limits.

Do not silently alter invalid perturbations. Record the rejection rate.

---

## Stage E7 — Counterfactual rollout protocol

Implement a reusable function with the conceptual interface:

```text
branch_rollout(
    recorded_state,
    recorded_actions_after_t,
    arm_q_perturbation,
    seed,
)
```

The actual Python signature must be derived from the checked source and project code. Do not copy this conceptual signature blindly.

For every branch:

```text
restore full state[t]
read verified arm q
apply perturbation to arm qpos
preserve all non-intervened simulator state entries
call sim.forward()
regenerate/update observables using the verified LIBERO mechanism
replay actions t ... T-1
evaluate final task success
save trajectory diagnostics
```

The branch result must include:

```text
task
demo_id
t
trajectory_length
epsilon
direction_seed
delta_q
q_before
q_after
baseline_success
perturbed_success
terminal success signal
replay state divergence
minimum joint-limit margin
exception / invalid reason
```

---

## Stage E8 — Smoke experiment

Before a full sweep, use a tiny run.

Suggested smoke scope:

```text
one task
one demonstration
approximately 8–12 branch timesteps covering the trajectory
one perturbation magnitude
4 perturbation directions per timestep
```

The exact branch indices must include early, middle, and late portions and should not be manually chosen after watching outcome results.

The smoke experiment is for validating code and output integrity, not for supporting a scientific claim.

Check:

```text
all branches start from the intended recorded state
q perturbations have the requested magnitude
non-arm state is unchanged at intervention time
unperturbed branch succeeds
success checker behaves correctly
outputs are deterministic when seed is repeated
run metadata is complete
```

---

## Stage E9 — Pilot sensitivity map

After the smoke test passes, run a pilot with:

```text
at least three demonstrations from one task
dense or near-dense temporal sampling
multiple perturbation directions
multiple perturbation magnitudes
```

Generate for each demonstration:

\[
C_\epsilon(t).
\]

Produce:

```text
criticality_vs_time.png
success_probability_vs_time.png
criticality_heatmap_time_x_epsilon.png
trajectory_event_overlay.png
```

Do not average across demonstrations before visually inspecting per-demonstration curves.

A positive pilot result should show reproducible temporal structure rather than isolated single-sample failures.

---

## Stage E10 — Full Exp1

The full run should include multiple task families and multiple demonstrations per task.

Do not set the final sample count based on a desired p-value after seeing results.

Before the full run, write a frozen config containing:

```text
selected tasks
selected demonstration IDs or selection rule
number of branch timesteps
temporal sampling rule
epsilon grid
number of perturbation directions
random seeds
invalid-perturbation handling
primary metrics
secondary metrics
plot definitions
statistical tests
```

Commit the config before launching the sweep.

---

## Primary metrics

### Time-indexed criticality

\[
C_\epsilon(t)
=
P_{base}(Y=1\mid t)
-
P_{perturbed}(Y=1\mid t,\epsilon).
\]

### Sensitivity concentration curve

Sort timesteps by measured criticality and compute the fraction of total criticality explained by the top fraction of timesteps.

The core plot is:

```text
x-axis: fraction of timesteps selected
y-axis: fraction of total measured criticality captured
```

Compare with a uniform diagonal baseline.

### Top-fraction concentration

Pre-register summaries such as:

```text
fraction of total sensitivity captured by top 10% of timesteps
fraction captured by top 20%
fraction captured by top 30%
```

### Temporal sparsity statistic

Report at least one distributional concentration measure in addition to the top-k curve, for example normalized entropy or Gini coefficient.

The exact implementation must be documented and unit-tested.

---

## Secondary diagnostics

Record:

```text
joint velocity
end-effector speed
gripper state
gripper closing rate
action magnitude
distance to joint limits
MuJoCo contact count
contact-pair changes
task-state changes
```

Contact analysis must not hardcode guessed geom names.

Inspect `sim.data.contact`, map geom IDs to names using the active MuJoCo model, save the raw contact names, and only then define robot-object contact classes.

The contact classifier itself must have a small validation report with manually inspected examples.

---

## Required controls

### Unperturbed branch control

At every sampled \(t\), branch from the recorded state with:

\[
\delta q=0.
\]

This controls for branch-restoration artifacts.

### Time-shuffled control

Shuffle the association between measured criticality values and trajectory time to estimate the null level for apparent temporal clustering.

### Matched perturbation-size control

Every compared timestep must use the same perturbation magnitude distribution.

### Joint-direction control

Verify that a critical time is not produced only by one pathological perturbation direction.

### State-restoration error control

Test whether measured criticality correlates with branch-restoration divergence. If it does, the result may be a simulator artifact.

---

## First physical-event analysis

Only after a robust criticality curve exists, overlay physical signals.

Candidate events:

```text
gripper begins closing
gripper reaches a closed configuration
first robot-object contact
contact-pair set changes
object begins moving
task predicate changes
```

Exact event definitions must be derived from verified runtime signals.

Compare critical windows against matched non-critical windows using event distance:

\[
d(t,\mathcal E)=\min_{e\in\mathcal E}|t-e|.
\]

A useful result would show that high-criticality timesteps are substantially closer to verified contact or dynamics events than randomly selected matched timesteps.

This supports Claim level C only after replication across task families.

---

## Important confound: teleport intervention

Direct qpos intervention is useful because it isolates local configuration sensitivity, but it creates a state that may not be dynamically reachable by the original controller history.

Therefore Exp1 must distinguish:

```text
Exp1-A: simulator-state q intervention
Exp1-B: dynamically realized perturbation
```

Exp1-A is the fast causal map.

If Exp1-A is positive, Exp1-B should introduce a brief control perturbation before the critical time so that the modified configuration arises through simulator dynamics rather than qpos teleportation.

Only agreement between Exp1-A and Exp1-B justifies a stronger physical-control interpretation.

---

## Important confound: open-loop demonstration continuation

Replaying the remaining demonstration actions means the continuation policy does not actively recover from perturbations.

Therefore the first result measures sensitivity under a fixed continuation.

If Exp1-A is positive, repeat a subset of experiments using a verified closed-loop policy.

The claim ladder is:

```text
fixed-continuation sensitivity
→ dynamically realized perturbation sensitivity
→ closed-loop decision sensitivity
```

Do not skip these distinctions in the paper.

---

## Go / no-go criteria

### Strong pass

Proceed to event-driven control if the following pattern appears across multiple tasks:

```text
criticality is strongly non-uniform over time
the concentration curve is substantially above uniform
the structure replicates across demonstrations
the result survives multiple perturbation magnitudes
the result is not explained by replay/restoration error
critical regions show meaningful physical-event alignment
```

### Partial pass

If criticality is non-uniform but not strongly sparse, revise the story toward:

> **adaptive temporal resolution for manipulation**

rather than claiming strong decision sparsity.

### Fail

If criticality is approximately uniform after controls, or if apparent peaks disappear under restoration controls or dynamically realized perturbations, do not proceed with the original decision-sparse claim.

Possible pivots should be recorded rather than hidden.

---

## Required output files

The Exp1 code should generate machine-readable outputs first and paper figures second.

Expected structure:

```text
runs/<run_id>/
├── config_resolved.yaml
├── command.txt
├── environment.txt
├── git_state.txt
├── stdout.log
├── stderr.log
├── metrics.json
├── branches.parquet
└── artifacts/
    ├── criticality_vs_time.png
    ├── criticality_heatmap_time_x_epsilon.png
    ├── concentration_curve.png
    ├── replay_error.png
    └── contact_overlay.png
```

Never make the only copy of a result a PNG.

The raw branch table is the source of truth.

---

## Research-log entry template

Append one entry after every setup step or experiment.

```markdown
## <timestamp> — <run_id or setup step>

**Question**

What exact uncertainty is this step testing?

**Pre-run expectation**

What outcome was expected before seeing the result?

**Code state**

Project SHA:
LIBERO SHA:
robosuite version:
Working tree clean/dirty:

**Environment**

Machine:
GPU:
Python:
MuJoCo:
CUDA:

**Data**

Suite:
Task ID:
Task name:
Demo IDs:

**Command**

```bash
<exact command>
```

**Configuration**

Config path:
Seed(s):
Perturbation settings:

**Result**

Primary metrics:
Secondary diagnostics:
Generated files:

**Failures / anomalies**

Record every warning, exception, replay divergence, invalid branch, or unexpected observation.

**Interpretation**

What does the result support?
What does it fail to support?
What alternative explanations remain?

**Claim impact**

Current strongest allowed claim:
Any claim that must be weakened:

**Next experiment**

One concrete next experiment and why it is the highest-value next step.
```

---

## Exp1 Codex prompt

Use this prompt after the repository has been created and both `PROJECT.md` and this file are present.

```text
Work only on EXP1 for the Decision-Sparse Latent RL project.

Read PROJECT.md and this EXP1.md completely before editing code. Read the latest research_log entry if present.

The experiment is intended to test whether q-space outcome sensitivity is temporally concentrated along successful LIBERO manipulation demonstrations.

Do NOT implement RL, latent representations, OpenPI, SVM, reward models, or real-robot code.

Your immediate task is to build the reproducibility and deterministic-replay foundation.

Non-negotiable rules:
- Never guess any API name, observation key, task identifier, dataset path, HDF5 key, MuJoCo state index, joint index, or configuration field.
- Inspect the exact checked-out LIBERO source, installed robosuite source/runtime, HDF5 files, and runtime objects before using identifiers.
- Record exact source locations for any private API used.
- Do not silently upgrade dependencies.
- Do not edit third_party code.
- Create a unique run_id for every run.
- Never overwrite a previous run.
- Append a research_log entry after every meaningful step, including failures.
- Preserve complete raw metrics and branch-level results.
- Use fixed random seeds and record them.
- Run smoke tests before any large sweep.
- Stop if the deterministic replay gate fails.
- Do not modify the scientific claim to fit the observed result.

Implement in this order:

A. Create or verify the project structure.

B. Implement scripts/exp1/audit_environment.py.
It must record project SHA, LIBERO SHA, Python, MuJoCo, robosuite, numpy, h5py, CUDA, GPU, and dataset root.

C. Implement scripts/exp1/enumerate_tasks.py.
Use the actual LIBERO benchmark API.
Save exact suite, task_id, task.name, task.language, task.problem_folder, and task.bddl_file fields to a manifest.
Do not hardcode task names.

D. Implement scripts/exp1/audit_dataset.py.
Inspect the real HDF5 file and save its full group/dataset schema with shapes and dtypes.

E. Implement scripts/exp1/replay_demo.py.
Restore a recorded initial state and replay recorded actions.
At every step, compare the runtime flattened simulator state with the next recorded state.
Save the full divergence curve and final success.
Do not use a single hardcoded tolerance as the only output.

F. Implement scripts/exp1/validate_branch_restore.py.
Compare reaching an intermediate state by normal replay against directly restoring the corresponding recorded state and taking the same next action.
Save state, q, EE, gripper, and observation discrepancies.

G. Verify the exact Panda arm qpos indices from the installed robosuite runtime and source.
For robosuite v1.4.0 the upstream source uses _ref_joint_pos_indexes, but you must assert the installed version and verify values against observed joint positions before relying on it.

H. Only if replay and branch restoration are reliable, implement the smallest q-perturbation smoke test.
Perturb arm joints only.
Keep all non-intervened state unchanged.
Use a tiny number of branch times and random directions.
Save every branch result.

After each step:
- append the research log,
- run tests,
- make an atomic Git commit,
- summarize changed files, exact commands, results, failures, and the next step.

At the end of this session, do not automatically launch the full sweep.
Report whether the replay gate and branch-restoration gate pass, with quantitative evidence.
```

---

## What Exp1 can answer if successful

A successful Exp1 provides evidence for a specific and important statement:

> **The final outcome of a manipulation trajectory is not equally sensitive to configuration error at every timestep.**

If the concentration is strong, Exp1 additionally supports:

> **A minority of timesteps accounts for a disproportionate fraction of measured outcome sensitivity.**

If the physical-event overlay is also robust:

> **These high-sensitivity regions concentrate near contact-sensitive or dynamics-sensitive transitions.**

Exp1 alone does **not** establish that latent RL is superior, that an event-driven policy is optimal, or that decision sparsity generalizes to all manipulation.

Those are later experiments.

