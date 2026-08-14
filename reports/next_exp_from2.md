# Next Experiment After EXP2: Time-Indexed Causal Criticality

## 1. Recommendation

| Item | Recommendation |
|---|---|
| Source experiment | EXP2 |
| Proposed experiment | **EXP3 — Preregistered Time-Indexed q-Criticality in LIBERO** |
| Priority | Highest-value next scientific step |
| Core question | Is perturbation sensitivity concentrated at a small subset of policy-step boundaries? |
| Required branching substrate | Corrected EXP2 Condition D |
| Scope | Causal measurement; no latent RL training yet |
| GPU use | Optional for analysis; MuJoCo branching remains on the validated CPU path |

EXP2 repaired the measurement substrate. The next experiment should now return
directly to the project thesis—“manipulation is densely controlled but may be
sparsely decided”—with a full, outcome-blind temporal intervention design. It
should not jump to latent actions or event-triggered RL before temporal concentration
has been demonstrated and replicated.

## 2. Why this is now justified

EXP2 established three prerequisites:

1. the corrected serialized snapshot produces exact zero twins across 324 pairs;
2. q-only interventions preserve all non-arm integration components at the branch
   and yield finite, joint-limit-valid continuations; and
3. the fixed 0.5%-of-joint-range intervention is strongly separated from the exact
   zero-control floor in a 48-intervention smoke test.

The unresolved scientific question is not whether an intervention is measurable.
It is whether the distribution of intervention effects over time is sparse,
reproducible, and meaningfully associated with task transitions rather than merely
with longer remaining horizons or chaotic contact dynamics.

## 3. Research questions

### EXP3-RQ1 — Temporal concentration

For a fixed, preregistered q intervention distribution, is causal effect mass
concentrated at a small fraction of policy-step boundaries?

### EXP3-RQ2 — Reproducibility

Do high-criticality regions reproduce across demonstrations of the same task and
across perturbation directions and signs?

### EXP3-RQ3 — Event alignment

Are high-criticality regions enriched near independently detected contact changes,
gripper-command changes, or task-predicate transitions?

### EXP3-RQ4 — Outcome relevance

Do continuous trajectory effects near a candidate critical region predict terminal
object-pose changes, task-predicate divergence, or final success flips?

## 4. Hypotheses and falsification criteria

### H3.1 — Concentrated criticality

A minority of frozen time points accounts for a majority of normalized causal
effect mass within each task.

This hypothesis should be considered unsupported if effect mass is approximately
uniform after correcting for continuation length and task scale, or if apparent
concentration is driven by one demonstration or one direction.

### H3.2 — Within-task reproducibility

High-criticality temporal regions have positive rank agreement across the three
demonstrations of a task after mapping time to normalized task progress.

This hypothesis fails if rankings are unstable across demonstrations or change
qualitatively under perturbation sign/direction.

### H3.3 — Event enrichment

Criticality is enriched in preregistered windows around independently defined
contact, gripper, or predicate transitions relative to outcome-blind temporal
controls.

This hypothesis fails if enrichment confidence intervals include the null or if the
association disappears when controlling for phase and remaining horizon.

### H3.4 — Substrate validity

Every paired zero control continues to satisfy the EXP2 gate, and every q
intervention continues to preserve non-arm state at the intervention boundary.

Any failure of H3.4 invalidates the affected causal comparison and stops EXP3 for
diagnosis; it must not be repaired by silently filtering difficult branches.

## 5. Frozen experimental unit and coverage

Retain the same pilot tasks and all three demonstrations per task:

```text
open_the_middle_drawer_of_the_cabinet: demo_0, demo_1, demo_2
turn_on_the_stove: demo_0, demo_1, demo_2
put_the_bowl_on_the_plate: demo_0, demo_1, demo_2
```

Use all 12 already frozen R3 branches per demonstration. Do not select times using
R5 effects. The base design is therefore 108 outcome-blind branch points.

Recommended directional coverage:

- 4 PCG64-seeded unit directions per branch;
- both signs for each direction;
- the same fixed 0.5%-of-runtime-joint-range epsilon used by R5; and
- one freshly validated matched zero control for every intervention pair.

This yields 864 q interventions (`108 × 4 × 2`) and their matched controls. If the
runtime cost is too high, reduce directions in a written preregistration before any
outcome is inspected; do not reduce branches or demonstrations based on results.

The R5 seed `20260814` has already produced observed effects. EXP3 should use a new,
frozen master seed and record the generated direction vector for every branch.

## 6. Mandatory state and boundary protocol

Use the exact EXP2 passing substrate:

```text
mjSTATE_INTEGRATION
+ explicit OperationalSpaceController state
+ explicit SingleArm buffers
+ environment timestep / cur_time / done
+ PandaGripper.current_action
```

Use the boundary frozen in
[`policy_step_boundary.json`](../experiments/exp2_simulator_reconciliation/manifests/policy_step_boundary.json):
after `ControlEnv.step(actions[t-1])` returns and immediately before
`ControlEnv.step(actions[t])`.

At every branch:

1. restore the identical corrected D snapshot for zero and perturbation arms;
2. verify an immediate capture/restore checksum;
3. run two zero continuations at minimum to confirm the local noise floor;
4. modify only runtime-discovered Panda arm qpos indices;
5. verify every non-arm INTEGRATION component has Linf at most `1e-12`;
6. verify perturbed q against runtime-derived joint limits;
7. continue with the identical recorded future actions; and
8. save complete per-step raw data for zero and perturbation continuations.

Do not insert an extra `mj_forward` at restoration. Do not change controller targets
to “help” the perturbed state. The next ordinary `env.step()` should handle the
same update path validated in EXP2.

## 7. Outcome variables

### 7.1 Primary causal effect

Use a fixed-horizon, paired response so that early branches do not automatically
receive larger scores merely because more future steps remain. A recommended
primary response is the area under a normalized state/task divergence curve over a
preregistered horizon `H`, capped by the remaining trajectory length.

The precise state components and normalization constants must be frozen from the
reference data before reading EXP3 intervention outcomes. At minimum, retain
separate q, qvel, EEF, object-pose, contact, and predicate channels rather than
collapsing them immediately into one opaque score.

### 7.2 Secondary outcomes

- maximum future INTEGRATION, q, qvel, and EEF divergence;
- terminal object-pose difference;
- contact-sequence divergence;
- task-predicate divergence and first divergence time;
- recovery time under a preregistered definition;
- terminal task success and success flip;
- paired standardized effect relative to the matched zero floor.

Success flip should remain secondary. It is scientifically meaningful but coarse,
and EXP2 observed only 5 flips in 48 smoke interventions.

### 7.3 Task-progress coordinate

Report both normalized policy time and an independently derived task-progress
coordinate. Task progress may use task predicates only after their exact runtime
implementation and directionality have been audited. Do not invent generic phase
labels that are not grounded in task state.

## 8. Measuring “decision sparsity”

Preregister at least two complementary concentration measures:

1. **Top-k effect mass:** fraction of absolute paired effect contributed by the top
   10%, 20%, and 30% of branch times within task/demonstration.
2. **Concentration index:** Gini coefficient or normalized entropy of nonnegative
   effect magnitudes over time.

Also report full temporal curves. A single Gini value can hide whether high values
are reproducible, located at trajectory boundaries, or driven by a single outlier.

Recommended inference unit: demonstration, not individual simulator step. Use a
hierarchical or cluster bootstrap that resamples demonstrations and then directions
within a demonstration. Avoid treating thousands of continuation steps as
independent samples.

Before the formal run, freeze what constitutes meaningful concentration. For
example, a preregistration could require all of the following:

- median top-20% effect mass above a stated threshold;
- bootstrap lower confidence bound above the uniform-time null;
- positive cross-demonstration rank agreement within at least two of three tasks;
- robustness to direction/sign aggregation; and
- no single demonstration contributing more than a stated share of the conclusion.

The numerical thresholds should be chosen from scientific effect-size reasoning or
an outcome-blind calibration, not tuned on EXP3 results.

## 9. Event-alignment analysis

Event alignment must be secondary to temporal concentration and use events defined
without q outcomes. Freeze event detectors from the local reference trajectories:

- first and large changes in raw contact-pair sets;
- gripper-command sign changes or the documented no-event state;
- task-predicate transitions;
- object support/grasp transitions only if exact runtime predicates can be audited.

Define symmetric event windows in policy steps or normalized time before the run.
Compare criticality inside and outside windows while conditioning on early/middle/
late phase and remaining horizon. Preserve tasks such as drawer demonstrations that
have no gripper sign change; do not force a semantic event label onto the 50%
fallback point.

## 10. Gates and stop conditions

### Measurement-validity gate

For every zero pair:

```text
final-success agreement = 100%
all INTEGRATION values finite
INTEGRATION L2 median <= 1e-10
INTEGRATION L2 P95 <= 1e-8
INTEGRATION L2 max <= 1e-6
terminal object-pose L2 P95 <= 1e-6
```

For every intervention:

```text
non-arm INTEGRATION Linf at intervention <= 1e-12
all q values within audited joint limits
all future states finite
no result-dependent branch removal
```

If any validity failure is systematic, stop the full run and diagnose it. Do not
loosen the EXP2 thresholds after seeing criticality outcomes.

### Scientific decision rule

- If temporal concentration and replication both pass, the next experiment may
  test whether an event-conditioned latent policy preserves performance with fewer
  high-level decisions.
- If concentration appears but does not replicate, expand demonstrations and
  directions before changing the method.
- If effects are broad and uniform, report negative evidence for the current q
  intervention definition; do not claim decision sparsity.
- If only terminal success is insensitive while continuous measures are structured,
  distinguish robustness of task completion from state-level criticality.
- If all fixed-epsilon effects saturate, specify a separate multi-epsilon experiment;
  do not tune epsilon inside EXP3.

## 11. Required artifacts

Create a separate, immutable EXP3 structure and preserve at least:

```text
experiments/exp3_time_indexed_criticality/
  EXP3.md
  configs/
  manifests/

runs/<run_id>/
  config_resolved.yaml
  command.txt
  environment.txt
  git_state.txt
  stdout.log
  stderr.log
  metrics.json
  artifacts/
    zero_controls.parquet
    interventions.parquet
    per_step_effects.parquet
    direction_manifest.json
    event_manifest.json
    failure_examples.json
    plots/
```

Required plots should show raw per-demonstration temporal curves, direction/sign
variability, concentration curves, event-window comparisons, and success/predicate
outcomes. Plotting only a task-averaged curve would hide the replication question.

## 12. Compute plan

MuJoCo physics and robosuite control should remain on the validated CPU execution
path. Parallelize only across independent process-level branch jobs if determinism
and immutable run-directory handling are revalidated. Avoid sharing a mutable
environment between workers.

The RTX 4090 may be used for later learned representations, bootstrap acceleration
implemented in a tested tensor backend, or policy training. Moving the simulator to
an unvalidated GPU physics implementation would create a new reconciliation problem
and is outside EXP3.

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Earlier branches have longer continuations | Use a preregistered fixed-horizon primary response and report terminal outcomes separately. |
| Contact chaos dominates raw state L2 | Report component/task metrics and replication, not only aggregate INTEGRATION norm. |
| Direction choice drives rankings | Use multiple seeded directions, both signs, and direction-level uncertainty. |
| One demonstration drives concentration | Treat demonstration as the inference unit and report leave-one-demo-out sensitivity. |
| R5 informed branch choice | Use all already frozen R3 branches; do not select using R5. |
| Snapshot schema regresses | Re-run corrected D zero controls in every formal job. |
| Public-prefix mismatch is confused with twin error | Branch only from corrected local D snapshots and retain prefix replay as a separate diagnostic. |

## 14. Claims EXP3 may unlock

If the preregistered concentration and replication tests pass, a narrowly scoped
claim may become defensible:

> Under fixed small q interventions and the tested LIBERO pilot, causal outcome or
> trajectory sensitivity is temporally concentrated rather than uniform.

Contact alignment requires its separate enrichment test. Event-driven latent RL is
still a method hypothesis even after concentration is observed; it should be tested
in a later ablation against dense-control and fixed-rate latent baselines.

## 15. Single highest-value next action

Write and commit the EXP3 preregistration—especially the fixed-horizon primary
effect, concentration thresholds, direction seed/count, and inference procedure—
before inspecting any additional q-intervention outcomes. Then execute the complete
108-branch design using the exact corrected Condition D snapshot implementation.
