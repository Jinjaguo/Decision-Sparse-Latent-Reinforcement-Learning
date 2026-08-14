# Next Experiment from EXP3: Replicated, Direction-Resolved, Progress-Aligned q Criticality

**Proposed experiment ID:** EXP4  
**Status:** proposal only; do not execute automatically  
**Source result:** EXP3 formal classification = partial support  
**Primary purpose:** determine whether EXP3's temporal non-uniformity becomes
replicable after increasing demonstration/direction coverage and aligning physical
task progress rather than normalized clock time.

## 1. Decision from EXP3

EXP3 produced a clear fork:

- concentration exceeded the uniform null: median top-20 mass 0.4493, hierarchical
  95% CI `[0.3579, 0.5861]` versus 0.25;
- but no task reached the frozen cross-demo rank threshold;
- only 2/9 demonstrations were stable across direction and sign;
- contact and gripper events were not enriched;
- the task-predicate event did not survive adjustment or FDR;
- 73.38% of individual interventions exceeded the effect threshold, so the result
  was not a rare-effect regime;
- 58 success flips occurred, but primary criticality was only weakly associated with
  terminal object effect and predicate divergence.

The appropriate branch of the preregistered decision tree is therefore:

> **Partial concentration with poor replication → expand demonstrations and
> directions, and test outcome-blind progress alignment before sparse control.**

EXP4 should not train a sparse-decision policy. Its job is to decide whether the
replication failure reflects inadequate sampling/direction resolution, clock-time
misalignment, or genuinely demonstration-specific sensitivity.

## 2. Main research question

Does a direction-resolved q-sensitivity profile become reproducible across held-out
demonstrations when branch times are aligned by audited physical task progress rather
than only by normalized action index?

The experiment has three nested questions:

1. Is the moderate effect concentration independently replicated on demonstrations
   not used by EXP3?
2. Is low direction robustness caused by using only four random projections in a
   seven-dimensional q space?
3. Does task-progress alignment improve cross-demo rank agreement without selecting
   landmarks from intervention outcomes?

## 3. Hypotheses

### H1: independent temporal non-uniformity replication

On held-out demonstrations 3–9, demonstration-median top-20 effect mass will remain
above its uniform null with a cluster-aware 95% CI lower bound above the null.

### H2: direction-resolved sensitivity

The temporal ranking of a direction-aggregated local sensitivity estimate will be
more stable than individual random-direction curves. Sign-paired central differences
will be more consistent than one-sided effects.

### H3: progress alignment

Cross-demo rank agreement after frozen physical-progress alignment will exceed the
agreement obtained with normalized clock time in at least two of three tasks.

### H4: task-specificity

If progress alignment helps bowl placement but not drawer/stove articulation, the
result should be reported as task-specific rather than averaged into a universal
critical-time claim.

### H5: event mechanism remains unsupported unless independently replicated

Contact-count and gripper-sign events should remain secondary. No event may become a
control trigger unless it passes adjusted, FDR-controlled inference in held-out demos.

## 4. Experimental scope

### 4.1 Demonstrations

Use the next seven successful demonstrations per task, indices 3–9, provided that
each reference trajectory passes the exact same-runtime Condition-D gate.

| Quantity | Proposed value |
|---|---:|
| Tasks | 3 |
| New demos per task | 7 |
| Total new demos | 21 |
| Branches per demo | 12 |
| Total branches | 252 |

EXP3 demos 0–2 remain a fixed development/legacy cohort. EXP4 demos 3–9 form the
primary held-out replication cohort. Do not combine them for the primary confirmatory
claim. A pooled 30-demo analysis may be secondary after the held-out result is fixed.

If fewer than seven additional successful dataset demonstrations exist for a task,
stop and report the actual availability before changing the sample design. Do not
silently substitute tasks or failed trajectories.

### 4.2 Directions

Use eight frozen directions per branch, both signs:

- seven directions should form a reproducible orthonormal basis in the audited
  joint-range-scaled q coordinates;
- one additional PCG64 random direction should serve as an out-of-basis check;
- randomize basis order per branch from a new frozen seed;
- retain both signs for antithetic pairing.

This yields:

```text
252 branches × 8 directions × 2 signs = 4,032 interventions
```

Why structured directions: EXP3's median sign-pair rho was 0.633, whereas its
direction-only rho was 0.378. A complete scaled basis can estimate whether temporal
criticality is carried by one or several joint-space directions and avoids treating
four random projections as a full local property.

### 4.3 Epsilon

Keep the primary epsilon at the EXP3 value:

```text
0.005 × verified joint range
```

EXP3 did not meet its saturation rule, so a multi-epsilon sweep is not the next main
question. A very small calibration subset at 0.0025 and 0.01 may be run only after
the primary 0.005 result, under a separately frozen secondary manifest, to inspect
local linearity. It must not replace or tune the primary epsilon.

## 5. Outcome-blind progress alignment

Progress variables must be defined from unperturbed references and task source code
before any EXP4 intervention outcomes are read.

### 5.1 Drawer progress

Audit the exact middle-drawer joint used by `WoodenCabinet.is_open`. Define progress
from the reference joint state relative to its reference start and successful terminal
state. Check monotonicity and retain the raw non-monotone sequence.

Candidate progress coordinate:

```text
p_drawer(t) = clip((drawer_joint(t) - drawer_joint(0)) /
                         (drawer_joint(T) - drawer_joint(0)), 0, 1)
```

Do not infer the joint by string matching alone; resolve it through the active LIBERO
object state and MuJoCo joint IDs.

### 5.2 Stove progress

Audit the exact stove-button joint consumed by `FlatStove.turn_on`. Define progress
using the reference joint coordinate and its successful terminal direction, with the
same source-level identifier audit as the drawer.

### 5.3 Bowl-on-plate progress

Use an outcome-blind vector of physical progress features from the reference:

- bowl-to-plate planar distance;
- bowl bottom relative to plate top height;
- bowl vertical lift above its initial height;
- gripper-to-bowl distance;
- exact `On(bowl, plate)` predicate state.

A scalar progress coordinate may be produced by a frozen piecewise phase map:

1. reach;
2. grasp/lift;
3. transport;
4. place/release.

Phase boundaries must be defined by audited physical thresholds or reference-only
change points. They may not be selected to maximize q effects.

### 5.4 Alignment methods

Predeclare two analyses:

1. **normalized-time baseline:** the exact EXP3 ten-quantile comparison;
2. **physical-progress alignment:** interpolate the direction-aggregated sensitivity
   curve onto a fixed 21-point progress grid from 0 to 1.

Optionally add reference-only monotone dynamic time warping as a secondary analysis.
Its distance features, regularization window, and treatment of non-monotone progress
must be frozen before intervention outcomes.

## 6. Intervention and restore protocol

Use the corrected EXP2 Condition D without modification:

1. restore `mjSTATE_INTEGRATION` at the pre-policy branch boundary;
2. restore the audited controller, robot buffers, environment timing, and
   `PandaGripper.current_action`;
3. verify two zero continuations per branch;
4. apply only the seven Panda arm-q values;
5. do not insert an extra `mj_forward` before the normal next policy step;
6. replay the recorded remaining action suffix;
7. record the same separate physical, contact, predicate, and integration channels.

Hard intervention gates remain:

- non-arm integration L-infinity change <= `1e-12`;
- exact audited joint limits;
- all states finite;
- no branch removal;
- both signs and all directions present;
- matched zero controls passing at every branch.

## 7. Branch selection

Retain twelve branches per demo to preserve comparability with EXP3, but freeze them
in two parallel coordinate systems:

- ten normalized-time quantiles plus the same contact/gripper audit slots;
- ten physical-progress quantiles plus two independently frozen task-semantic
  landmarks.

To keep the primary intervention budget at 12 branches, use the normalized-time set
for the confirmatory replication and treat progress alignment initially as a mapping
of those same outcomes. Only if reference-only coverage shows that normalized-time
branches leave large progress gaps should a separately preregistered second branch
set be added.

Duplicate resolution must remain nearest unused action index, with the lower index
chosen at equal distance. Record all replacements.

## 8. Metrics

### 8.1 Preserve EXP3 metrics

Keep all EXP3 channels and its remaining-horizon mean so EXP4 remains directly
comparable. Report each component separately.

### 8.2 Direction-resolved local sensitivity

For each basis direction `d_j`, compute:

```text
even effect_j = (effect(+epsilon d_j) + effect(-epsilon d_j)) / 2
odd effect_j  = abs(effect(+epsilon d_j) - effect(-epsilon d_j)) / 2
```

The even term measures sign-symmetric amplification; the odd term diagnoses local
asymmetry/nonlinearity. Aggregate the seven basis even effects with both:

- median across directions;
- root-mean-square across directions.

The primary EXP4 branch metric should be selected and frozen after reference/zero
audits but before intervention outcomes. Recommended primary: RMS of sign-paired even
effects across the seven basis directions, duration-normalized over the remaining
horizon. The extra random direction is a held-out projection diagnostic.

### 8.3 Concentration and replication

Retain:

- top-10/20/30 mass;
- Gini;
- normalized entropy;
- cross-demo Spearman;
- direction/sign robustness;
- LODO;
- hierarchical task/demo/direction/sign bootstrap.

Add:

- intraclass correlation of aligned curve values across demos;
- top-k set overlap (Jaccard) after normalized-time and progress alignment;
- improvement in cross-demo rho from clock time to progress;
- variance components attributable to task, demo, branch time/progress, direction,
  sign, and residual.

## 9. Statistical plan

### 9.1 Primary inference unit

Demonstration remains the inference unit. Never treat the 4,032 interventions or
their future steps as independent samples.

### 9.2 Held-out primary test

On demos 3–9 only:

1. calculate one concentration summary per demo;
2. resample tasks, demos within task, and directions/signs within branch;
3. report the hierarchical 95% CI for median top-20 mass;
4. compare against the exact uniform null implied by the number of evaluated points.

### 9.3 Alignment comparison

For every task and demo pair, calculate Spearman rho under normalized time and
physical progress. Use paired demo-pair differences and a task/demo-clustered
permutation test. Report effect size and CI, not only a p-value.

### 9.4 Event analysis

Keep events secondary. Use the same controls as the corrected EXP3 analysis:

- remaining horizon;
- early/middle/late phase;
- demo fixed effects, which control task;
- within-demo label permutation;
- BH FDR across event families.

### 9.5 Multiple comparisons

One primary endpoint should govern the go/no-go decision. Apply BH FDR 0.05 to the
secondary alignment, event, component, and task-specific families. Predeclare which
family each hypothesis belongs to.

## 10. Proposed decision rule

Freeze exact numbers before full outcomes. A reasonable starting rule is:

### Strong replicated progress-aligned sparsity

Require all:

1. held-out median top-20 mass >= 0.50;
2. hierarchical 95% CI lower bound above the uniform null;
3. progress alignment increases median cross-demo rho by at least 0.15;
4. at least two tasks have progress-aligned median cross-demo rho >= 0.50;
5. at least 70% of held-out demos have direction-resolved robustness >= 0.50;
6. LODO minimum top-20 mass >= 0.45;
7. the held-out random direction agrees with the basis aggregate at rho >= 0.50.

If all pass, the project may proceed to an oracle sparse-decision control experiment.

### Replicated non-uniformity without aligned sparse times

If concentration replicates but alignment/robustness fails, conclude that q
sensitivity is non-uniform yet trajectory-specific. Move to adaptive temporal
allocation or state-conditional criticality, not a universal event schedule.

### Progress alignment succeeds only by task

If only one task passes, develop task-conditional progress models and forbid a
task-general sparsity claim.

### Uniform or broad sensitivity

If the held-out concentration CI overlaps the uniform null, record a negative
replication. If sensitivity remains broad, investigate adaptive dense-to-sparse
control budgets rather than fixed sparse decision points.

### Direction instability persists

If a complete basis still yields low direction agreement, replace scalar time
criticality with an anisotropic state-time sensitivity object, such as the spectrum
of a local effect Gram matrix. Do not average away the direction dependence.

## 11. Hard gates and stop conditions

Stop before formal intervention outcomes if any of the following occurs:

1. fewer than the preregistered successful held-out demos are available;
2. corrected Condition D fails on any selected task/demo stratum;
3. a task progress variable cannot be tied to an exact runtime identifier and source
   predicate;
4. zero twins exceed the exact/declared noise gate;
5. any structured direction/sign violates joint limits;
6. progress alignment or primary metric is not committed before outcomes;
7. dry-run Parquet schemas do not round-trip;
8. any branch would need to be removed after observing an effect.

If a gate fails, create a failure report and a new immutable run. Do not patch a run
in place.

## 12. Required manifests

Before full execution, freeze and hash:

1. `task_demo_manifest.json`;
2. `reference_validation_manifest.json`;
3. `branch_manifest.json`;
4. `progress_channel_schema.json`;
5. `progress_alignment_spec.json`;
6. `direction_basis_manifest.json`;
7. `effect_channel_schema.json`;
8. `effect_normalization.json`;
9. `primary_metric_spec.json`;
10. `event_manifest.json`;
11. `statistical_analysis_plan.json`;
12. `scientific_decision_rule.json`.

Each must record schema version, UTC freeze time, project SHA, source runs, source
file hashes, seed derivation, and an outcome-blind declaration.

## 13. Required artifacts and plots

Minimum formal artifacts:

- `zero_controls.parquet`;
- `interventions.parquet`;
- `per_step_effects.parquet`;
- `branch_summary.parquet`;
- `direction_resolved_summary.parquet`;
- `progress_aligned_curves.parquet`;
- `replication_summary.parquet`;
- `variance_components.json`;
- `scientific_decision.json`;
- `failure_examples.json`;
- all frozen manifests and hashes.

Recommended plots:

1. held-out criticality versus normalized time per demo;
2. held-out criticality versus physical progress per demo;
3. normalized-time versus progress-aligned cross-demo rho;
4. direction-basis heatmap by time/progress;
5. even versus odd sign-paired effects;
6. held-out random-direction agreement;
7. concentration curves for EXP3 versus EXP4 held-out demos;
8. task/demo/direction variance components;
9. top-k overlap matrices;
10. LODO and leave-one-task-out summaries;
11. event enrichment with adjusted CIs;
12. terminal object/predicate/success relevance.

## 14. Compute estimate

EXP3's 864 interventions produced 45,656 continuation rows in roughly several minutes
on the local validated CPU simulator path. EXP4's proposed 4,032 interventions are
about 4.67 times larger, before zero controls. Expected storage remains modest, but
the run should checkpoint by trajectory and merge only after every shard passes its
gate.

GPU use is optional for post-processing, bootstrap acceleration, or later learned
models. It should not replace the validated CPU MuJoCo path unless a separate
determinism audit proves equivalence.

## 15. Failure interpretations

### More demos erase concentration

Interpret EXP3 as a small-sample positive fluctuation. Record a negative replication
and do not proceed to sparse control.

### Progress alignment does not improve rank agreement

Interpret criticality as trajectory-specific, direction-specific, or dependent on
unobserved policy state. Consider state-conditional adaptive scheduling rather than
global phase landmarks.

### Basis aggregate is stable but individual directions are not

Use the direction-aggregated operator norm as the causal object. Do not claim that
every q perturbation shares the same critical times.

### Only bowl placement replicates

The likely mechanism is contact-rich transport/place sensitivity rather than a
universal manipulation principle. Continue with a bowl-family task set and report the
restricted scope.

### qvel continues to dominate

Audit the reference-only qvel normalizer and report component-specific results. A
predeclared alternate scale may be used as sensitivity analysis, but the primary
normalization cannot be selected from intervention outcomes.

## 16. Relationship to the research main line

EXP4 remains directly on the decision-sparsity main line. It tests whether a reliable
small set of causally important decision regions exists before investing in control
or representation learning.

The sequence should be:

```text
EXP2 exact branching
  → EXP3 pilot temporal non-uniformity with weak replication
  → EXP4 held-out, direction-resolved, progress-aligned replication
  → only if strong: oracle sparse-decision control
  → only after oracle benefit: learned sparse/event policy or latent RL
```

This ordering protects the project from building a learned event mechanism around a
pilot effect that may not generalize.

## 17. Single recommended next action

Audit demonstrations 3–9 for the same three tasks and create the outcome-blind
progress-channel schema. Do not generate any new q outcomes until the exact drawer
joint, stove joint, bowl/plate geometry, structured direction basis, alignment method,
primary metric, and decision rule are frozen and committed.
