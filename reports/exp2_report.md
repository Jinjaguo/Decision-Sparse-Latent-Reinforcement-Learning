# EXP2 Report: Full-State Matched-Twin Simulator Reconciliation

## 1. Report record

| Item | Value |
|---|---|
| Experiment | EXP2 |
| Title | Full-State Matched-Twin Simulator/Data Reconciliation |
| Execution date | 2026-08-14 (America/New_York) |
| Final status | **Completed; R2, corrected R4 Condition D, and R5 passed** |
| Selected restore condition | **D: `mjSTATE_INTEGRATION` plus explicit controller, robot, environment-timing, and Panda-gripper state** |
| Zero-twin gate | **Passed: all evaluated errors were exactly `0.0`** |
| R5 measurability gate | **Passed: 48/48 interventions exceeded the zero-noise criterion** |
| Protocol | [`EXP2.md`](../experiments/exp2_simulator_reconciliation/EXP2.md) |
| Controlling execution prompt | [`EXP2_CODEX_PROMPT_Simulator_Reconciliation.md`](../prompts/EXP2_CODEX_PROMPT_Simulator_Reconciliation.md) |
| Project thesis | Manipulation is densely controlled but may be sparsely decided. |

## 2. Executive conclusion

EXP2 established a valid same-runtime counterfactual branching substrate for the
tested LIBERO / robosuite stack. The public LIBERO 79-element flattened state used
in EXP1 was not enough for historical replay. In EXP2, the smallest tested snapshot
that passed the preregistered zero-twin gate was:

1. MuJoCo `mjSTATE_INTEGRATION` (390 elements in this model/runtime);
2. explicit Operational Space Controller state;
3. explicit robot recent buffers;
4. environment step bookkeeping; and
5. `env.robots[0].gripper.current_action`, the mutable Panda-gripper action
   integrator.

The final corrected Condition D run covered all 9 demonstrations, all 12 frozen
branch times per demonstration, and 3 repeats per branch, for 324 matched zero-twin
pairs and 17,121 paired continuation steps. Integration, qpos, qvel, controller,
EEF, and terminal object-pose errors all had median, P95, and maximum equal to
`0.0`. Final-success agreement was 100%. Error remained exactly zero for every
task, for contact and non-contact branches, and for early, middle, and late phases.

R5 was therefore legally reached. Its fixed smoke design evaluated 48 q-only
interventions: 3 tasks, one demonstration per task, 4 frozen temporal quantiles,
2 seeded directions, and both signs. Every zero control remained exact, every
intervention preserved non-arm integration components exactly, all states were
finite, and all perturbed q values stayed inside runtime-derived Panda joint
limits. All 48 effects exceeded 10 times their corresponding zero-noise P99; the
global zero-noise P99 was `0.0`, the median maximum-future integration-state effect
was `49.824963778807955`, the maximum was `4405.331419219012`, and 5 interventions
flipped final task success.

This validates measurement, not the main scientific hypothesis. EXP2 permits the
statement that local q interventions are measurable above the matched-twin
simulator noise floor. It does **not** establish temporal decision sparsity, contact
alignment, event-driven control, or the value of a latent action representation.

## 3. Scope and preregistered boundaries

EXP2 was a reconciliation experiment. It deliberately did not implement latent
representation learning, SVMs, event-level RL, OpenPI integration, a full temporal
q sweep, or real-robot execution. The task and demonstration selection was carried
forward unchanged from EXP1:

| Task ID | Task | Demonstrations |
|---:|---|---|
| 0 | `open_the_middle_drawer_of_the_cabinet` | `demo_0`, `demo_1`, `demo_2` |
| 7 | `turn_on_the_stove` | `demo_0`, `demo_1`, `demo_2` |
| 8 | `put_the_bowl_on_the_plate` | `demo_0`, `demo_1`, `demo_2` |

The R4 gate was frozen before the formal comparisons:

| Criterion | Threshold |
|---|---:|
| Demonstrations | all 9 |
| Branches per demonstration | 12 |
| Repeats per branch | at least 3 |
| Comparisons per condition | at least 324 |
| Final-success agreement | 100% |
| Integration L2 median | at most `1e-10` |
| Integration L2 P95 | at most `1e-8` |
| Integration L2 maximum | at most `1e-6` |
| Terminal object-pose L2 P95 | at most `1e-6` |
| Systematic task/contact/time spikes | none |

These thresholds were not relaxed. The corrected D result passed the original
thresholds with an observed numerical floor of exactly zero.

## 4. Runtime and source audit (R0)

### 4.1 Fixed runtime

| Component | Audited value |
|---|---|
| Conda environment | `libero-exp1` |
| Python | 3.8.20 |
| NumPy | 1.22.4 |
| h5py | 3.11.0 |
| PyArrow | 17.0.0 |
| robosuite | 1.4.0 |
| MuJoCo | 3.2.3 |
| LIBERO source revision | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| robosuite reference revision | `fbee5844ff5632f5b5698e204ec5357ca50be0df` |
| GPU available | NVIDIA GeForce RTX 4090, 24,564 MiB |
| Physics execution | CPU MuJoCo path; GPU was not used for simulator stepping |

GPU availability did not justify changing the physics backend during a simulator
reconciliation experiment. The same CPU MuJoCo path was retained across reference,
zero-twin, and perturbation continuations. The GPU remains available for later
representation-learning or policy-training experiments.

### 4.2 Model and state dimensions

The representative audited model reported:

| Field | Value |
|---|---:|
| `nq` | 41 |
| `nv` | 37 |
| `na` | 0 |
| `nu` | 9 |
| `nbody` | 38 |
| `nmocap` | 0 |
| `nuserdata` | 0 |
| Legacy flattened state | 79 |
| `mjSTATE_FULLPHYSICS` | 79 |
| `mjSTATE_INTEGRATION` | 390 |

The installed binding exposed `mj_stateSize`, `mj_getState`, `mj_setState`, and
`mj_forward`. It did not expose `mj_copyData`. The installed `MjData` object did
support independent `copy.deepcopy`, which was used only for diagnostic Condition
E, not as the final serialized controller evidence.

The runtime-reported integration components were:

```text
mjSTATE_TIME
mjSTATE_QPOS
mjSTATE_QVEL
mjSTATE_ACT
mjSTATE_WARMSTART
mjSTATE_CTRL
mjSTATE_QFRC_APPLIED
mjSTATE_XFRC_APPLIED
mjSTATE_EQ_ACTIVE
mjSTATE_MOCAP_POS
mjSTATE_MOCAP_QUAT
mjSTATE_USERDATA
mjSTATE_PLUGIN
```

The full machine-readable audit is in
[`runtime_state_schema.json`](../experiments/exp2_simulator_reconciliation/manifests/runtime_state_schema.json).

### 4.3 Explicit Python runtime state

The source and runtime object graph showed that the next policy step also depends
on mutable Python-side state. The final Condition D serialization covers:

- OSC references and goals: `initial_joint`, `initial_ee_pos`,
  `initial_ee_ori_mat`, `goal_pos`, `goal_ori`, `relative_ori`, `ori_ref`,
  `new_update`, and the classified controller caches;
- robot outputs and buffers: torques, recent qpos/actions/torques, recent EEF
  force-torque, pose, velocity, acceleration, and the velocity ring buffer;
- environment timing: timestep, current time, and done status; and
- Panda gripper history: `env.robots[0].gripper.current_action`.

The exact object paths, types, shapes, dtypes, runtime-change observations, and
source references are recorded in
[`controller_state_schema.json`](../experiments/exp2_simulator_reconciliation/manifests/controller_state_schema.json).

## 5. Snapshot boundary and implementation (R1)

The final boundary is not an informal “before action” point. It is defined as:

- boundary `t=0`: after XML reset, simulator reset, recorded initial-state write,
  post-processing, forced observable update, controller initial-joint/EEF reference
  synchronization, and `controller.new_update=True`;
- boundary `t>0`: immediately after `ControlEnv.step(actions[t-1])` returns and
  before `ControlEnv.step(actions[t])` is called; and
- continuation from boundary `t`: call `ControlEnv.step(actions[t])` next.

The call-chain audit observed 25 inner simulation steps for an outer policy action.
The trace and source locations are frozen in
[`policy_step_boundary.json`](../experiments/exp2_simulator_reconciliation/manifests/policy_step_boundary.json).

The snapshot utilities provide explicit capture, restoration, validation,
serialization, and deserialization for legacy, FULLPHYSICS, and INTEGRATION state,
plus explicit Python runtime state. Restoration intentionally does not insert an
extra `mj_forward`: the next normal robosuite step performs its own required update.

Final R1 run: `exp2_r1_gripper_20260814T011302`.

| R1 result | Value |
|---|---:|
| Audited boundaries | 4 |
| Boundary trace events | 778 |
| Observed inner simulation steps | 25 |
| Maximum INTEGRATION round-trip L2 | `0.0` |
| Maximum controller/runtime round-trip error | `0.0` |
| Gate | **pass** |

An earlier diagnostic called `mj_forward` immediately after restoration. That
modified derived/warm-start quantities and was inconsistent with the audited outer
step boundary. Removing that extra operation produced exact round trips. The failed
diagnostic was preserved rather than deleted.

## 6. Same-runtime local references (R2)

Final R2 run: `exp2_r2_gripper_refs_20260814T011336`.

For each frozen public episode, the experiment loaded the stored XML, initialized
from the verified first state, executed the recorded actions, and saved a local
snapshot at every audited policy boundary. Each boundary contains legacy,
FULLPHYSICS, INTEGRATION, controller/robot/gripper state, action indexing, Panda arm
q, EEF pose, task status, and raw runtime-derived contacts.

| R2 criterion | Result |
|---|---:|
| Reference rollouts | 9 |
| Successful final outcomes | 9/9 |
| Finite snapshots | all |
| Consistent shapes | all |
| Maximum INTEGRATION round-trip L2 | `0.0` |
| Maximum explicit runtime-state round-trip error | `0.0` |
| Unclassified required mutable fields | 0 |
| Gate | **pass** |

The public HDF5 trajectory was retained only as a diagnostic. It continued to show
nonzero differences, particularly in bowl-on-plate trajectories, confirming that
EXP2 did not erase or reinterpret the negative EXP1 evidence.

One earlier R2 implementation interleaved restore diagnostics with the reference
rollout and thereby mutated Python controller history. It failed and was replaced
by a clean two-phase procedure: generate the immutable reference first, then run
diagnostics separately. Later audits also synchronized OSC null-space initialization
and added the missing gripper integrator. All intermediate runs remain preserved.

## 7. Frozen branch selection (R3)

The final branch manifest contains 108 branches: 9 trajectories times 12 branches.
For each trajectory it selected 10 temporal quantiles (5% through 95%), the first
verified gripper-command sign change, and the first maximum contact-count change.
Duplicates use the nearest unused valid index, choosing the lower index on ties.

The three drawer demonstrations contain a constant gripper command of `-1`, so no
sign change exists. The preregistered deterministic fallback uses the 50% temporal
point. This fallback is recorded explicitly and was not selected using q outcomes.

The final manifest is
[`branch_times_reconciliation.json`](../experiments/exp2_simulator_reconciliation/manifests/branch_times_reconciliation.json)
and was committed before corrected formal validation.

## 8. Restore-condition experiment (R4)

### 8.1 Initial formal A–D comparison

Run: `exp2_r4_formal_20260814T002437`.

This run evaluated 1,296 pairs (324 per condition), 68,484 paired steps, and 890,292
component rows. All four conditions failed in this initial run. The results below
are retained because they show the failure progression; the initial D result is
superseded by the corrected D validation in Section 8.4.

| Condition | Integration L2 median | P95 | Max | Final success agreement gate | Overall |
|---|---:|---:|---:|---|---|
| A: legacy, no controller | `3.873e-10` | `53.1652` | `3054.3088` | fail | fail |
| B: FULLPHYSICS, no controller | `3.854e-10` | `54.3039` | `3054.3088` | fail | fail |
| C: INTEGRATION, no controller | `0.0` | `54.3039` | `3054.3088` | fail | fail |
| D: INTEGRATION + then-classified controller/robot | `0.0` | `46.6820` | `3054.3088` | fail | fail |

Selected additional global statistics from the same initial run:

| Condition | qpos P95 | qvel P95 | Controller P95 | EEF-position P95 | Terminal object-pose P95 |
|---|---:|---:|---:|---:|---:|
| A | `0.0652293` | `0.0937812` | `131.0` | `0.0112896` | `0.2022633` |
| B | `0.0652293` | `0.0937812` | `131.0` | `0.0112896` | `0.2022633` |
| C | `0.0656238` | `0.0940594` | `131.0` | `0.0112896` | `0.2022633` |
| D, incomplete schema | `0.0224012` | `0.0428395` | `29.0743` | `0.00624490` | `0.1728352` |

The median of C and incomplete D was zero, but their tails were catastrophic. This
is precisely why the gate uses P95, maximum, task/contact/time strata, and outcome
agreement rather than only the mean or median.

### 8.2 Diagnostic Condition E

Because the initial D failed, the protocol required Condition E. Run
`exp2_r4_condition_e_20260814T010207` used an independent same-process deep copy of
the complete `mujoco.MjData` plus the then-classified controller/robot state.

It still failed: integration L2 median was `0.0`, P95 was `44.40608641901275`, and
maximum was `3054.308753850878`; terminal object-pose P95 was
`0.1728352435383957`. Drawer was exact, while stove and bowl-on-plate diverged.
Because full MuJoCo data did not repair the task-specific pattern, the missing state
had to be Python-side rather than an omitted MuJoCo array.

### 8.3 Root cause

Source inspection identified the missing mutable variable:

```text
env.robots[0].gripper.current_action
```

`PandaGripper.format_action` updates this value by a fixed increment during the
internal control loop. It is therefore an action integrator, not a static model
parameter or a derived observation. Its omission explains the task pattern:

- drawer commands remained at one sign and saturated, masking the missing history;
- stove and bowl demonstrations changed gripper direction, so restoring a default
  `current_action` changed subsequent actuator commands and caused divergence; and
- copying all MuJoCo data could not capture this Python object field.

The field was added to the explicit schema, capture/restore implementation, R1
round-trip audit, R2 references, and frozen branch manifest before revalidation.

### 8.4 Corrected formal Condition D

Run: `exp2_r4_condition_d_gripper_20260814T011457`.

| Gate metric | Observed | Required | Status |
|---|---:|---:|---|
| Demonstrations | 9 | 9 | pass |
| Branches per demonstration | 12 | 12 | pass |
| Repeats per branch | 3 | at least 3 | pass |
| Zero-twin pairs | 324 | at least 324 | pass |
| Paired continuation steps | 17,121 | retained raw | pass |
| Component rows | 222,573 | retained raw | pass |
| Final-success agreement | 100% | 100% | pass |
| Integration L2 median | `0.0` | at most `1e-10` | pass |
| Integration L2 P95 | `0.0` | at most `1e-8` | pass |
| Integration L2 maximum | `0.0` | at most `1e-6` | pass |
| Terminal object-pose L2 P95 | `0.0` | at most `1e-6` | pass |
| All integration values finite | yes | yes | pass |
| Systematic stratum spikes | none | none | pass |

All reported medians, P95 values, maxima, and bootstrap 95% confidence intervals
for integration, qpos, qvel, controller, EEF, and terminal object pose were exactly
`0.0`. The same was true independently for all three tasks, contact/non-contact
branches, and early/middle/late phases.

The selected minimal tested representation is therefore Condition D. Condition C
did not pass, and the decisive state absent from C is Python-side gripper history.
Controller/runtime history was necessary in the tested robosuite 1.4.0 stack.

### 8.5 Prefix-replay diagnostic

Reconstructing a branch only by replaying the public prefix in a freshly initialized
environment did not reproduce the saved local boundary exactly: the maximum
diagnostic prefix INTEGRATION L2 error was `2167.4243431321693` and legacy-state L2
was `1.2156529760717303`. This diagnostic is not the zero-twin gate: both formal
twins are restored from the identical serialized local snapshot. It does show that
public-prefix reconstruction is not an interchangeable substitute for the final D
snapshot and remains an unresolved portability/history issue.

## 9. Minimal q-perturbation measurability smoke test (R5)

Run: `exp2_r5_q_smoke_20260814T012633`.

R5 used `demo_0` for each task and the 15%, 35%, 65%, and 85% branch quantiles from
the already frozen R3 manifest. At each of 12 branch points, it generated two PCG64
normal unit directions with seed `20260814` and tested both signs. The fixed
perturbation was:

```text
delta_q[j] = sign * 0.005 * verified_joint_range[j] * unit_direction[j]
```

Panda arm qpos indices and joint ranges were discovered from the active model; no
hand-written joint offsets or limits were used. Each intervention changed Panda arm
q only and preserved all other integration components. A normal subsequent
`env.step()` recomputed controller-dependent quantities; no unregistered controller
target change was inserted.

| R5 metric | Result |
|---|---:|
| Total interventions | 48 |
| Interventions per task | 16 |
| Signs | 24 negative, 24 positive |
| Per-step rows | 2,488 |
| All zero controls pass | yes |
| Global zero-noise P99 | `0.0` |
| Maximum zero-control noise | `0.0` |
| Maximum non-arm component Linf at intervention | `0.0` |
| All q values inside verified limits | yes |
| All post-intervention states finite | yes |
| Effects above 10x corresponding zero-noise P99 | 48/48 |
| Minimum maximum-future integration L2 effect | `32.83229142722571` |
| Median maximum-future integration L2 effect | `49.824963778807955` |
| Approximate empirical P95 | `2858.992845159297` |
| Maximum maximum-future integration L2 effect | `4405.331419219012` |
| Final-success flips | 5/48 |
| Contact-divergence steps summed over interventions | 904 |
| Task-predicate-divergence steps summed over interventions | 130 |
| Maximum terminal object-pose L2 | `0.4943669282463729` |
| R5 gate | **pass** |

These magnitudes are not estimates of a population effect and must not be read as a
temporal-importance ranking. The R5 branch set was intentionally tiny and selected
only to test separability from numerical restoration noise.

## 10. Artifact inventory

### Final reference run

[`runs/exp2_r2_gripper_refs_20260814T011336`](../runs/exp2_r2_gripper_refs_20260814T011336)
contains the 9 reference trajectories, per-boundary simulator arrays, explicit
runtime snapshots, round-trip validations, the reference manifest, run config,
command, environment, Git state, stdout, stderr, metrics, and failure record.

### Full A–D comparison

[`runs/exp2_r4_formal_20260814T002437`](../runs/exp2_r4_formal_20260814T002437)
contains 1,296 pair rows, 68,484 step rows, 890,292 component rows, failure examples,
prefix diagnostics, and the four required restore-condition plots.

### Corrected passing D validation

[`runs/exp2_r4_condition_d_gripper_20260814T011457`](../runs/exp2_r4_condition_d_gripper_20260814T011457)
contains 324 pair rows, 17,121 step rows, 222,573 component rows, zero failure
examples, prefix diagnostics, and regenerated plots for the selected condition.

### R5 smoke test

[`runs/exp2_r5_q_smoke_20260814T012633`](../runs/exp2_r5_q_smoke_20260814T012633)
contains 2,488 per-step rows, 48 intervention summaries, the exact joint-limit
manifest, failure examples, and `perturbation_effect_vs_noise.png`.

All tabular artifacts are actual Zstandard-compressed Parquet files written with
PyArrow, not renamed CSV or JSON files.

## 11. What was observed, inferred, and not resolved

### Direct observations

- Public historical replay retained 9/9 final success but failed state-level
  reproduction in EXP1.
- R1 exact snapshot round trips are possible at the audited policy boundary.
- All 9 local references succeeded and passed immediate round-trip checks.
- Conditions A, B, and C failed the R4 gate.
- Incomplete D and diagnostic E failed in a task-specific way.
- Adding the explicit Panda gripper integrator made corrected D exactly identical
  across every tested zero-twin step and stratum.
- The fixed R5 q perturbations were measurably separated from the exact zero floor.

### Evidence-supported inferences

- MuJoCo integration state alone is insufficient for policy-step branching in this
  exact software stack.
- Additional Python-side controller/robot/gripper runtime history is causal for the
  next transition.
- `PandaGripper.current_action` was the decisive omitted field in the initially
  tested D/E schema.
- The corrected D snapshot is a valid substrate for same-runtime matched branching
  in the tested 3-task × 3-demonstration pilot.

### Unresolved questions

- Which subset of the other explicitly restored controller and robot fields is
  mathematically minimal was not exhaustively ablated after the gripper fix.
- Public-prefix replay still does not recreate the saved local branch state exactly.
- The corrected gate has not yet been tested on all LIBERO tasks, all available
  demonstrations, another OS, another MuJoCo/robosuite version, or another machine.
- R5 does not determine whether sensitivity is concentrated at a small number of
  times, whether it aligns with contact, or whether effects generalize across
  demonstrations and directions.

## 12. Claim update

### Strongest allowed claims

> Accurate policy-step branching in the tested LIBERO / robosuite 1.4.0 / MuJoCo
> 3.2.3 stack requires MuJoCo integration state plus additional explicitly captured
> controller, robot, environment-timing, and Panda-gripper runtime state.

> With that corrected state, same-runtime matched zero branches were identical at
> the measured numerical precision across 324 pairs and 17,121 continuation steps.

> In the preregistered EXP2 smoke test, local Panda-arm q interventions were
> measurable above the matched-twin simulator noise floor.

### Claims that remain forbidden

- decision sparsity exists;
- only a small set of timesteps controls manipulation outcomes;
- critical timesteps align with contact or gripper events;
- the 5 observed success flips establish a temporal bottleneck;
- event-triggered or latent-action RL is justified;
- the result generalizes beyond the tested stack and pilot tasks.

## 13. Tests and dependency validation

Before EXP2 modifications, the repository test suite had 14 passing tests. After
the full implementation and final R5 run:

```text
25 passed, 0 failed, 0 skipped
pip check: No broken requirements found.
```

Coverage added for simulator snapshot capture/restore and serialization, explicit
controller/runtime round trips, deterministic branch selection, q-intervention
masking and non-arm preservation, and non-overwriting run directories.

## 14. Principal execution commands

```powershell
'C:\Users\Guoji\anaconda3\envs\libero-exp1\python.exe' scripts\exp2\audit_runtime_state.py --run-id exp2_r0_gripper_audit_20260814T011233

'C:\Users\Guoji\anaconda3\envs\libero-exp1\python.exe' scripts\exp2\compare_restore_conditions.py --run-id exp2_r1_gripper_20260814T011302

'C:\Users\Guoji\anaconda3\envs\libero-exp1\python.exe' scripts\exp2\generate_local_reference.py --run-id exp2_r2_gripper_refs_20260814T011336

'C:\Users\Guoji\anaconda3\envs\libero-exp1\python.exe' scripts\exp2\validate_zero_twins.py --run-id exp2_r4_formal_20260814T002437 --reference-run runs\exp2_r2_nullspace_refs_20260814T002227 --repeats 3

'C:\Users\Guoji\anaconda3\envs\libero-exp1\python.exe' scripts\exp2\validate_zero_twins.py --run-id exp2_r4_condition_e_20260814T010207 --reference-run runs\exp2_r2_nullspace_refs_20260814T002227 --repeats 3 --condition-e-only

'C:\Users\Guoji\anaconda3\envs\libero-exp1\python.exe' scripts\exp2\validate_zero_twins.py --run-id exp2_r4_condition_d_gripper_20260814T011457 --reference-run runs\exp2_r2_gripper_refs_20260814T011336 --repeats 3 --condition-d-only

'C:\Users\Guoji\anaconda3\envs\libero-exp1\python.exe' scripts\exp2\perturbation_smoke.py --run-id exp2_r5_q_smoke_20260814T012633 --zero-twin-run runs\exp2_r4_condition_d_gripper_20260814T011457 --reference-run runs\exp2_r2_gripper_refs_20260814T011336

'C:\Users\Guoji\anaconda3\envs\libero-exp1\python.exe' -m pytest -q
'C:\Users\Guoji\anaconda3\envs\libero-exp1\python.exe' -m pip check
```

Every formal run directory also contains its literal command, resolved config,
environment, Git state, stdout, stderr, and metrics.

## 15. Initial A–D statistics by task

The following tables complete the requested global and per-task reporting for the
initial formal A–D run. Bracketed ranges are bootstrap 95% confidence intervals
using seed 0 and 1,000 resamples, matching the formal summarizer. They were computed
from the immutable raw Parquet rows. “D” here means the explicitly labeled
**incomplete-schema** D run; the corrected D statistics are all zero globally and
for every task, as reported in Section 8.4.

### 15.1 INTEGRATION-state L2

| Condition | Scope | Median [95% CI] | P95 [95% CI] | Maximum |
|---|---|---:|---:|---:|
| A | global | 3.873e-10 [2.969e-10, 5.543e-10] | 53.17 [46.68, 64.4] | 3054 |
| A | drawer | 9.064e-12 [8.692e-12, 9.433e-12] | 1.052e-09 [8.518e-10, 1.313e-09] | 53.17 |
| A | stove | 0.8504 [0.6619, 1.093] | 28.9 [27.09, 30.25] | 104.9 |
| A | bowl | 0.08894 [0.05077, 0.1736] | 234 [199.8, 259.7] | 3054 |
| B | global | 3.854e-10 [2.915e-10, 5.529e-10] | 54.3 [47.18, 64.56] | 3054 |
| B | drawer | 8.926e-12 [8.522e-12, 9.234e-12] | 1.027e-09 [8.159e-10, 1.313e-09] | 53.17 |
| B | stove | 0.9581 [0.7177, 1.109] | 29.22 [27.22, 30.59] | 104.9 |
| B | bowl | 0.08894 [0.05077, 0.1736] | 234 [199.8, 259.7] | 3054 |
| C | global | 0 [0, 0] | 54.3 [47.18, 64.56] | 3054 |
| C | drawer | 0 [0, 0] | 0 [0, 0] | 53.17 |
| C | stove | 0.9581 [0.7177, 1.109] | 28.43 [26.99, 29.79] | 104.9 |
| C | bowl | 0.08894 [0.05077, 0.1736] | 234 [199.8, 259.7] | 3054 |
| D | global | 0 [0, 0] | 46.68 [39.64, 56.7] | 3054 |
| D | drawer | 0 [0, 0] | 0 [0, 0] | 0 |
| D | stove | 0.6426 [0.5715, 0.7284] | 26.94 [25.84, 27.63] | 87.76 |
| D | bowl | 0.02285 [0.0167, 0.04137] | 225.8 [191.6, 243.5] | 3054 |

### 15.2 qpos L2

| Condition | Scope | Median [95% CI] | P95 [95% CI] | Maximum |
|---|---|---:|---:|---:|
| A | global | 2.707e-12 [2.358e-12, 3.421e-12] | 0.06523 [0.05748, 0.07228] | 0.6074 |
| A | drawer | 1.043e-13 [9.66e-14, 1.169e-13] | 4.113e-12 [3.919e-12, 4.355e-12] | 0.2131 |
| A | stove | 0.0004172 [0.0003968, 0.0004554] | 0.2164 [0.2119, 0.223] | 0.6074 |
| A | bowl | 4.91e-05 [4.538e-05, 5.284e-05] | 0.06709 [0.0638, 0.07391] | 0.2558 |
| B | global | 2.684e-12 [2.332e-12, 3.414e-12] | 0.06523 [0.05748, 0.07228] | 0.6074 |
| B | drawer | 1.03e-13 [9.599e-14, 1.093e-13] | 4.022e-12 [3.826e-12, 4.307e-12] | 0.2131 |
| B | stove | 0.0004331 [0.0004087, 0.0004629] | 0.2164 [0.2119, 0.223] | 0.6074 |
| B | bowl | 4.91e-05 [4.538e-05, 5.284e-05] | 0.06709 [0.0638, 0.07391] | 0.2558 |
| C | global | 0 [0, 0] | 0.06562 [0.05747, 0.07265] | 0.6074 |
| C | drawer | 0 [0, 0] | 0 [0, 0] | 0.2131 |
| C | stove | 0.0004331 [0.0004087, 0.0004629] | 0.2161 [0.2118, 0.2228] | 0.6074 |
| C | bowl | 4.91e-05 [4.538e-05, 5.284e-05] | 0.06709 [0.0638, 0.07391] | 0.2558 |
| D | global | 0 [0, 0] | 0.0224 [0.02124, 0.02343] | 0.4836 |
| D | drawer | 0 [0, 0] | 0 [0, 0] | 0 |
| D | stove | 0.0004305 [0.0004062, 0.0004593] | 0.05667 [0.04656, 0.07024] | 0.4836 |
| D | bowl | 4.131e-05 [3.665e-05, 4.344e-05] | 0.02868 [0.0267, 0.03005] | 0.09808 |

### 15.3 qvel L2

| Condition | Scope | Median [95% CI] | P95 [95% CI] | Maximum |
|---|---|---:|---:|---:|
| A | global | 5.597e-12 [4.513e-12, 7.087e-12] | 0.09378 [0.08512, 0.09843] | 2.733 |
| A | drawer | 1.335e-13 [1.168e-13, 1.445e-13] | 1.07e-11 [9.362e-12, 1.205e-11] | 1.22 |
| A | stove | 0.0009583 [0.0007331, 0.001438] | 0.3196 [0.2966, 0.343] | 2.733 |
| A | bowl | 0.0001205 [6.775e-05, 0.0002528] | 0.06823 [0.06171, 0.07528] | 0.6638 |
| B | global | 5.565e-12 [4.51e-12, 7.053e-12] | 0.09378 [0.08511, 0.09796] | 2.733 |
| B | drawer | 1.298e-13 [1.119e-13, 1.42e-13] | 1.039e-11 [9.288e-12, 1.158e-11] | 1.22 |
| B | stove | 0.0009951 [0.0008006, 0.001467] | 0.3196 [0.2966, 0.343] | 2.733 |
| B | bowl | 0.0001205 [6.775e-05, 0.0002528] | 0.06823 [0.06171, 0.07528] | 0.6638 |
| C | global | 0 [0, 0] | 0.09406 [0.08446, 0.09862] | 2.733 |
| C | drawer | 0 [0, 0] | 0 [0, 0] | 1.22 |
| C | stove | 0.0009951 [0.0008006, 0.001467] | 0.319 [0.299, 0.3428] | 2.733 |
| C | bowl | 0.0001205 [6.775e-05, 0.0002528] | 0.06823 [0.06171, 0.07528] | 0.6638 |
| D | global | 0 [0, 0] | 0.04284 [0.03964, 0.04742] | 0.4222 |
| D | drawer | 0 [0, 0] | 0 [0, 0] | 0 |
| D | stove | 0.0008479 [0.0006861, 0.000961] | 0.1481 [0.1288, 0.191] | 0.4222 |
| D | bowl | 2.765e-05 [2.128e-05, 5.016e-05] | 0.05263 [0.04793, 0.06097] | 0.3843 |

### 15.4 Controller maximum field L2

| Condition | Scope | Median [95% CI] | P95 [95% CI] | Maximum |
|---|---|---:|---:|---:|
| A | global | 70 [70, 72] | 131 [131, 131] | 357.9 |
| A | drawer | 90 [90, 90] | 143 [143, 143] | 224.7 |
| A | stove | 63 [63, 63.51] | 136 [121.1, 144.7] | 357.9 |
| A | bowl | 59 [58, 59] | 87 [86, 87] | 106.3 |
| B | global | 70 [70, 72] | 131 [131, 131] | 357.9 |
| B | drawer | 90 [90, 90] | 143 [143, 143] | 224.7 |
| B | stove | 63 [63, 63.51] | 137.5 [130.3, 147.2] | 357.9 |
| B | bowl | 59 [58, 59] | 87 [86, 87] | 106.3 |
| C | global | 70 [70, 72] | 131 [131, 131] | 357.9 |
| C | drawer | 90 [90, 90] | 143 [143, 143] | 224.7 |
| C | stove | 63 [63, 63] | 130.8 [115.7, 137.5] | 357.9 |
| C | bowl | 59 [58, 59] | 87 [86, 87] | 106.3 |
| D | global | 0 [0, 0] | 29.07 [26.2, 31.79] | 357.9 |
| D | drawer | 0 [0, 0] | 0 [0, 0] | 0 |
| D | stove | 0.9747 [0.5056, 1.704] | 81.57 [80.2, 97.35] | 357.9 |
| D | bowl | 0.005559 [0.003319, 0.008049] | 15.56 [14.42, 16.43] | 106.3 |

### 15.5 EEF-position L2

| Condition | Scope | Median [95% CI] | P95 [95% CI] | Maximum |
|---|---|---:|---:|---:|
| A | global | 2.257e-13 [1.896e-13, 2.807e-13] | 0.01129 [0.009578, 0.01172] | 0.03883 |
| A | drawer | 6.716e-15 [6.064e-15, 7.072e-15] | 3.474e-13 [2.947e-13, 3.838e-13] | 0.01107 |
| A | stove | 8.677e-05 [7.902e-05, 9.451e-05] | 0.02049 [0.01996, 0.02098] | 0.03147 |
| A | bowl | 1.975e-05 [1.923e-05, 2.034e-05] | 0.01219 [0.01192, 0.01254] | 0.03883 |
| B | global | 2.257e-13 [1.816e-13, 2.817e-13] | 0.01129 [0.009578, 0.01172] | 0.03883 |
| B | drawer | 6.659e-15 [6.049e-15, 6.98e-15] | 3.481e-13 [2.983e-13, 3.885e-13] | 0.01107 |
| B | stove | 8.188e-05 [7.88e-05, 9.389e-05] | 0.02049 [0.01996, 0.02098] | 0.03147 |
| B | bowl | 1.975e-05 [1.923e-05, 2.034e-05] | 0.01219 [0.01192, 0.01254] | 0.03883 |
| C | global | 0 [0, 0] | 0.01129 [0.009578, 0.01172] | 0.03883 |
| C | drawer | 0 [0, 0] | 0 [0, 0] | 0.01107 |
| C | stove | 8.188e-05 [7.88e-05, 9.389e-05] | 0.02049 [0.01996, 0.02098] | 0.03147 |
| C | bowl | 1.975e-05 [1.923e-05, 2.034e-05] | 0.01219 [0.01192, 0.01254] | 0.03883 |
| D | global | 0 [0, 0] | 0.006245 [0.004408, 0.007177] | 0.03883 |
| D | drawer | 0 [0, 0] | 0 [0, 0] | 0 |
| D | stove | 8.013e-05 [7.852e-05, 8.611e-05] | 0.01436 [0.0109, 0.01671] | 0.03147 |
| D | bowl | 1.729e-05 [1.469e-05, 1.793e-05] | 0.01218 [0.01187, 0.01253] | 0.03883 |

### 15.6 Terminal object-pose L2

| Condition | Scope | Median [95% CI] | P95 [95% CI] | Maximum |
|---|---|---:|---:|---:|
| A | global | 5.836e-09 [4.761e-12, 0.0006709] | 0.2023 [0.1623, 0.275] | 0.3725 |
| A | drawer | 1.966e-12 [1.7e-12, 2.871e-12] | 8.921e-12 [5.65e-12, 0.1263] | 0.1263 |
| A | stove | 0.001307 [0.000479, 0.005673] | 0.2639 [0.1172, 0.3725] | 0.3725 |
| A | bowl | 0.02635 [0.003992, 0.03654] | 0.2657 [0.192, 0.2767] | 0.2767 |
| B | global | 5.836e-09 [4.696e-12, 0.0008526] | 0.2023 [0.1623, 0.275] | 0.3725 |
| B | drawer | 1.966e-12 [1.7e-12, 2.911e-12] | 8.502e-12 [5.339e-12, 0.1263] | 0.1263 |
| B | stove | 0.001307 [0.0006198, 0.005673] | 0.2639 [0.1172, 0.3725] | 0.3725 |
| B | bowl | 0.02635 [0.003992, 0.03654] | 0.2657 [0.192, 0.2767] | 0.2767 |
| C | global | 0 [0, 0.0008526] | 0.2023 [0.1623, 0.2749] | 0.3725 |
| C | drawer | 0 [0, 0] | 0 [0, 0.1263] | 0.1263 |
| C | stove | 0.001307 [0.0006198, 0.005673] | 0.2628 [0.1175, 0.3725] | 0.3725 |
| C | bowl | 0.02635 [0.003992, 0.03654] | 0.2657 [0.192, 0.2767] | 0.2767 |
| D | global | 0 [0, 1.858e-05] | 0.1728 [0.1547, 0.2485] | 0.3725 |
| D | drawer | 0 [0, 0] | 0 [0, 0] | 0 |
| D | stove | 0.001307 [0.0006198, 0.002664] | 0.2377 [0.03337, 0.3725] | 0.3725 |
| D | bowl | 0.02424 [0.003815, 0.03393] | 0.2323 [0.1691, 0.2749] | 0.2749 |

## 16. Direct answers required by the protocol

| Question | Answer |
|---|---|
| Which restoration condition passed? | Corrected Condition D. |
| What exact state was required? | `mjSTATE_INTEGRATION` plus explicit OSC, robot-buffer, environment-timing, and Panda-gripper `current_action` state. |
| Was controller history necessary? | Yes, in the tested stack; C failed and corrected D passed. |
| What was the zero-twin noise floor? | Exactly `0.0` for all reported zero-twin metrics. |
| Did the result depend on contact? | No observed restoration dependence; both contact and non-contact strata were exactly zero after correction. |
| Is matched branching valid? | Yes, for the tested same-runtime serialized-snapshot protocol and pilot scope. |
| Was q measurability tested? | Yes, in the preregistered 48-intervention smoke test. |
| Was the effect separable from noise? | Yes; 48/48 exceeded 10x their zero-noise P99, with exact zero controls. |
| What comes next? | A fully preregistered, time-indexed q-criticality experiment using corrected D snapshots; see [`next_exp_from2.md`](next_exp_from2.md). |
