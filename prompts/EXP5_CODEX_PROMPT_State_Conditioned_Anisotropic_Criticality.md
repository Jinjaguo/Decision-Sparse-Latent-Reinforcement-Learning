# EXP5 Codex Prompt — Cross-Fitted State-Conditioned Anisotropic q Criticality

**Project:** Decision-Sparse Latent RL for Contact-Rich Manipulation  
**Experiment:** EXP5 — Cross-Fitted State-Conditioned Anisotropic Criticality  
**Date:** 2026-08-14  
**Status:** Ready for preregistration; do not execute before protocol freeze  
**Primary scientific purpose:** determine whether reproducible local q-sensitivity geometry emerges after conditioning on audited physical state and perturbation radius, rather than forcing demonstrations onto one scalar time/progress coordinate  
**Development cohort:** EXP4 demos 3–9  
**Confirmatory cohort:** new demos 10–19  
**Simulation backend:** exact corrected-D CPU MuJoCo / robosuite path validated in EXP2–EXP4  
**GPU requirement:** use RTX 4090 for eligible matching, operator, SVD/eigendecomposition, bootstrap, and permutation workloads after CPU/GPU equivalence audit  
**Forbidden in EXP5:** sparse-time policy, latent RL, OpenPI, SVM, learned sparse scheduler, real-robot RL

## Protocol amendment — 2026-08-14 confirmatory reference completion

The original exact-index stop rule is superseded by this prospective completion
rule, authorized after the first reference audit and before any EXP5 q-intervention
outcome existed:

```text
1. retain every successful demos 10–19 same-runtime reference;
2. for any task with fewer than 10 qualified references, scan unused dataset demos
   from demo_20 in strictly increasing index order;
3. accept a candidate only when final success, finite-state, model, snapshot
   round-trip, and controller round-trip gates all pass;
4. stop scanning that task immediately when it reaches 10 qualified references;
5. preserve and report every rejected candidate;
6. do not reuse demos 3–9 or select candidates using q-intervention, criticality,
   operator, event, or terminal-perturbation outcomes.
```

The formal confirmatory cohort is eligibility-conditioned rather than the original
fixed-index cohort. The original drawer demo17 failure remains immutable negative
substrate evidence. This amendment does not authorize repeated-until-success
execution of the same trajectory, silent deletion, or threshold changes. Every
downstream manifest must record the replacement mapping and selection limitation.

---

## 1. Required reading

Before changing code, read completely:

```text
PROJECT.md
experiments/exp1_decision_sparsity/EXP1.md
experiments/exp2_simulator_reconciliation/EXP2.md
experiments/exp3_time_indexed_criticality/EXP3.md
experiments/exp4_replicated_progress_criticality/EXP4.md
reports/exp1_report.md
reports/exp2_report.md
reports/exp3_report.md
reports/exp4_report.md
reports/next_exp_from4.md
paper/evidence_table.md
the latest research_log entry
```

Also inspect the exact passing EXP2–EXP4 snapshot, perturbation, GPU-analysis, and manifest implementations.

Before creating new code, search the repository for reusable modules. Do not duplicate the corrected snapshot substrate.

Never guess file paths, task IDs, runtime object paths, MuJoCo joint/body IDs, controller fields, progress/state features, dataset keys, joint ranges, GPU devices, or manifest schemas. Discover them from the current repository, runtime, frozen manifests, or source.

---

## 2. EXP4 result that defines EXP5

EXP4 formal classification:

```text
replicated_nonuniformity_without_aligned_sparse_times
```

Held-out execution:

```text
3 tasks
7 new demos per task
21 held-out demos
252 branches
4,032 q interventions
211,808 future-step records
```

Corrected Condition D remained exact:

```text
756 matched pairs
39,714 continuation steps
all reported restoration errors = 0
```

Matched-zero:

```text
252 / 252 branches passed
median = 0
P95 = 0
max = 0
```

Independent replication:

```text
held-out median top-20 effect mass = 0.4507
95% hierarchical bootstrap CI = [0.4360, 0.5560]
uniform null = 0.25
```

This closely reproduces EXP3:

```text
EXP3 top-20 median = 0.4493
```

Therefore temporal non-uniformity is independently replicated.

The stronger sparse-time story did not replicate:

```text
physical-progress alignment improvement:
median delta rho = 0.0437
95% CI = [-0.0333, 0.1749]
p = 0.2549

direction-robust demos = 5 / 21
held-out random-direction agreement = 0.4336
```

The full q basis showed strong anisotropy:

```text
median leading-eigenvalue share = 0.7549
drawer = 0.8447
bowl = 0.6652
stove = 0.7736
```

but dominant directions changed with state and trajectory.

Variance decomposition:

```text
87.44% residual/interactions
5.99% progress position
4.76% demo-within-task
1.60% task
0.17% marginal direction
0.03% marginal sign
```

EXP4 also found 417 terminal success flips, but only weak association between the primary scalar criticality and terminal object displacement / predicate divergence.

No contact, gripper, or predicate event family survived adjusted FDR analysis.

The scientific conclusion entering EXP5 is:

> **Local q sensitivity is reproducibly non-uniform, but it is not adequately described by a universal sparse set of decision times or a single globally stable q direction.**

EXP5 must test whether the stable object is a **state-conditioned anisotropic local response geometry**.

---

## 3. Core EXP5 question

> **Does conditioning on audited physical state reveal a reproducible local q-sensitivity field or low-dimensional sensitive subspace that is obscured by scalar time/progress alignment?**

EXP5 separates:

```text
A. state misalignment
B. finite-radius nonlinearity
C. genuine trajectory specificity
```

---

## 4. Research questions

### RQ1 — State-conditioned scalar replication

Does reference-only physical-state matching improve cross-demo scalar q-sensitivity agreement over normalized time and EXP4 scalar progress?

### RQ2 — Subspace replication

Do leading right-singular subspaces of the local finite-difference response operator agree across state-matched demonstrations?

### RQ3 — Radius dependence

Are operator scale, sign symmetry, and principal subspaces stable between:

```text
0.0025 × joint range
0.0050 × joint range
```

and, on a frozen calibration subset:

```text
0.0100 × joint range
```

?

### RQ4 — Held-out direction prediction

Can an operator estimated from seven orthonormal basis directions predict the response to an eighth held-out random direction?

### RQ5 — Held-out confirmation

Do state-conditioned improvements developed on demos 3–9 reproduce on new demos 10–19?

### RQ6 — Eligibility for control

Does EXP5 support a state-conditioned sensitivity field strongly enough to justify a later oracle adaptive-decision scheduler?

Sparse or latent RL remains forbidden until the frozen EXP5 decision rule passes.

---

## 5. Claim boundary at EXP5 start

Allowed:

> Small local q perturbations produce reproducibly non-uniform remaining-horizon physical effects in the tested LIBERO tasks.

Allowed:

> Local response geometry is strongly anisotropic in the EXP4 held-out cohort.

Not allowed:

```text
universal sparse decision times exist
a global critical q direction exists
physical progress solves temporal misalignment
contact/gripper/predicate events are reliable triggers
scalar time criticality is sufficient
q alone is a Markov state
sparse RL or latent RL is beneficial
```

---

## 6. Cohort design and leakage control

### Development cohort

Use only:

```text
demo_3 ... demo_9
```

to develop and freeze:

```text
state descriptor
feature scaling
distance metric
matching algorithm
temporal constraint
prototype construction
match rejection rule
state-aligned interpolation
decision thresholds
```

EXP4 intervention outcomes may be used only for explicitly labeled development analysis.

### Confirmatory cohort

Use new demonstrations:

```text
demo_10 ... demo_19
```

for each of the same three tasks.

Target:

```text
3 tasks
10 demos per task
30 demos total
```

Before confirmatory q outcomes, audit all requested demos, generate same-runtime local references, validate final success, validate corrected Condition D, and validate state descriptors.

If fewer than 10 demos qualify for any task:

```text
apply the frozen 2026-08-14 completion rule above
scan unused demos in increasing index order
stop only if demo20–49 cannot supply 10 qualified references for that task
report every attempted and rejected candidate
```

---

## 7. Five-fold cross-fitting

Within the confirmatory cohort, use deterministic five-fold cross-fitting per task.

For each held-out fold fit using only training-fold **reference trajectories**:

```text
continuous feature scales
shrinkage covariance
state prototypes
matching bandwidth / rejection threshold
any optional reference-only encoder
```

No intervention outcome may enter matching, normalization, prototype construction, or representation learning.

Freeze fold membership before q outcomes.

---

## 8. Confirmatory branch budget

Target:

```text
16 state-coverage branches per demo
```

giving:

```text
30 demos × 16 = 480 branches
```

Branches must be selected from reference-only state prototypes / coverage.

Do not choose branches using EXP3/EXP4 criticality, success flips, operator norm, or event outcomes.

Freeze the selection algorithm first.

---

## 9. Primary physical state descriptor

The primary descriptor must be explicit, low-dimensional, and physically auditable.

All features come from unperturbed references at the corrected pre-policy boundary.

### Shared robot features

Include where deterministically available:

```text
7 joint-range-normalized Panda q positions
7 normalized Panda q velocities
EEF position
EEF continuous 6D orientation
EEF linear velocity
EEF angular velocity
gripper opening / command
PandaGripper.current_action
task-relevant contact indicators
```

EXP2 showed `PandaGripper.current_action` can be causally necessary. Do not omit it from the state audit.

### Drawer-specific features

Audit exact runtime identifiers for:

```text
middle drawer joint coordinate
middle drawer joint velocity
EEF-to-handle relative position/orientation
gripper-to-handle distance
handle/gripper contact state
exact Open predicate state
```

### Stove-specific features

Audit exact runtime identifiers for:

```text
stove-button joint coordinate
button joint velocity
EEF-to-button relative position/orientation
finger/gripper-to-button distances
button contact state
exact TurnOn predicate state
```

### Bowl-specific features

Audit exact runtime identifiers for:

```text
bowl pose relative to plate
bowl velocity relative to plate
bowl-to-plate planar displacement
bowl vertical clearance
bowl lift above initial height
EEF pose relative to bowl
gripper-to-bowl distance
gripper-bowl contact
bowl-plate contact
exact On predicate state
```

Do not manufacture missing sites or infer identifiers by string matching alone.

---

## 10. Descriptor scaling

Scale continuous features from development reference data only.

Use immutable physical floors to prevent near-constant channels from exploding.

Binary predicate/contact features must not dominate the distance metric.

Freeze:

```text
state_descriptor_schema.json
state_descriptor_scaling.json
```

Record feature name, task, unit, source, distribution, scale, floor, weight, and rationale.

---

## 11. Primary state distance

Use a task-specific shrinkage Mahalanobis distance.

Fit covariance only on fold-permitted reference data.

Freeze:

```text
shrinkage method
regularization
binary/continuous handling
missing-value handling
distance normalization
```

If the intended implementation is unavailable, inspect installed libraries before choosing another one.

Do not silently switch to Euclidean distance.

---

## 12. Monotone state matching

Primary matching:

```text
1. build dense descriptor sequences
2. compute pairwise state costs
3. perform monotone matching
4. constrain with a frozen Sakoe-Chiba-style temporal window
5. reject matches above a training-reference-only distance threshold
6. preserve rejected matches and reasons
```

The temporal window prevents pathological backwards matching but must not reduce the method to normalized time.

Freeze the window before confirmatory outcomes.

---

## 13. State prototypes and branch selection

Use reference-only prototypes to select 16 branches per confirmatory demo.

A valid high-level scheme is:

```text
cluster / medoid development reference states
map prototypes into each confirmatory reference
select nearest valid state per prototype
maximize prototype coverage
resolve duplicates deterministically
```

Freeze the exact algorithm.

Save:

```text
prototype_manifest.json
confirmatory_branch_manifest.json
match_tables.parquet
```

---

## 14. Alignment baselines

Compare against all frozen baselines:

```text
1. normalized time
2. EXP4 scalar physical progress
3. shared robot features only
4. task-object geometry only
5. unconstrained nearest-neighbor diagnostic
```

Do not add/drop baselines after confirmatory outcomes.

---

## 15. Optional reference-only encoder

Only if the physical descriptor fails a preregistered reference-coverage gate may a small reference-only encoder be added.

Allowed training data:

```text
unperturbed reference trajectories only
```

Forbidden:

```text
q-intervention effects
criticality labels
branch effect labels
success/failure labels beyond demo qualification
```

The physical descriptor remains primary.

---

## 16. Directions

Use:

```text
7 orthonormal directions
+ 1 held-out random direction
both signs
```

Generate from a new frozen PCG64 seed.

The basis convention must be deterministic and auditable.

---

## 17. Perturbation radii

All 480 branches:

```text
r_small = 0.0025 × audited joint range
r_main  = 0.0050 × audited joint range
```

Frozen stratified 20% calibration subset:

```text
r_large = 0.0100 × audited joint range
```

Select calibration branches before outcomes and stratify by task, demo, and state prototype.

---

## 18. Planned intervention budget

Primary two-radius layer:

```text
480 × 8 directions × 2 signs × 2 radii
= 15,360 interventions
```

Additional large-radius calibration:

```text
96 × 8 × 2
= 1,536 interventions
```

Expected total:

```text
16,896 interventions
```

plus matched-zero continuations.

If compute is excessive, reduce branches from 16 to 12 before reducing confirmatory demos or removing the smaller radius.

Any budget change must be preregistered before outcomes.

---

## 19. Corrected-D substrate

Use corrected Condition D unchanged:

```text
MuJoCo mjSTATE_INTEGRATION
explicit OSC state
robot buffers
environment timing/done
PandaGripper.current_action
```

Do not minimize the state schema in EXP5.

Use the exact validated pre-policy boundary.

Do not insert an extra `mj_forward`.

Do not alter controller targets.

---

## 20. Zero-control gate

Run at least two matched-zero continuations at every confirmatory branch.

Required:

```text
final-success agreement = 100%
all INTEGRATION values finite
INTEGRATION L2 median <= 1e-10
INTEGRATION L2 P95 <= 1e-8
INTEGRATION L2 max <= 1e-6
terminal object-pose L2 P95 <= 1e-6
```

Systematic failure stops EXP5.

Do not relax thresholds.

---

## 21. Intervention gate

For every branch/direction/sign/radius:

```text
modify only seven Panda arm qpos entries
verify runtime q indices
verify runtime joint limits
preserve every non-arm integration component
replay identical future action suffix
```

Require:

```text
non-arm INTEGRATION Linf <= 1e-12
all q values valid
all arrays finite
all required directions/signs/radii present
no post-outcome branch deletion
```

---

## 22. Preserve EXP4 physical channels

Retain:

```text
arm q
arm qvel
EEF position
EEF orientation
task-object position
task-object orientation
```

Keep contact, predicate, success, and full integration norm separate.

---

## 23. Signed physical response

EXP5 must preserve signed normalized future-response features.

Do not save only nonnegative divergences.

Store signed differences sufficient to estimate:

```text
delta q
delta qvel
delta EEF position
delta EEF orientation representation
delta task-object position
delta task-object orientation representation
```

Freeze orientation representation and normalization before outcomes.

---

## 24. Local finite-difference operator

For branch state `x`, basis direction `d_j`, radius `r`:

```text
J_r[:, j] =
[y(x + r d_j) - y(x - r d_j)]
/
(2r)
```

where `y` is the preregistered duration-normalized signed physical response vector.

Do not call `J_r` a Jacobian when the local-linearity gate fails. Call it a finite-radius response operator.

---

## 25. Gram matrix and basis-invariant geometry

Construct:

```text
G_r = J_r^T J_r
```

Report:

```text
spectral norm
Frobenius norm
leading eigenvalue share
effective rank
top-1 right-singular subspace
top-2 right-singular subspace
condition number where meaningful
```

Subspace comparison must be basis-order and sign invariant.

---

## 26. Primary subspace similarity

For projector:

```text
P = V_k V_k^T
```

recommended similarity:

```text
similarity(P_a, P_b)
=
1 - ||P_a - P_b||_F / sqrt(2k)
```

Freeze:

```text
k = 1 primary
k = 2 secondary
```

Add exact known-answer tests.

---

## 27. Radius-linearity analysis

Compare 0.0025 vs 0.005, and on calibration branches 0.01.

Report:

```text
relative spectral-norm discrepancy
top-1 projector similarity
top-2 projector similarity
sign asymmetry
held-out random-direction prediction
response scaling with radius
```

Do not average radii before linearity analysis.

---

## 28. Frozen local-linearity gate

Freeze exact thresholds before q outcomes.

Candidate criteria should jointly constrain:

```text
cross-radius top-1 subspace similarity
operator norm scaling
sign asymmetry
held-out random-direction prediction
```

Do not tune thresholds from confirmatory results.

---

## 29. Held-out random-direction prediction

Estimate the response to the eighth direction using the seven-basis operator.

Compare predicted vs actual signed response and scalar effect.

Report:

```text
per-branch prediction error
per-demo rank rho
task-level rho
median rho
radius dependence
```

---

## 30. State-conditioned scalar sensitivity

Preserve EXP4-compatible `S_RMS` at each radius separately.

Evaluate cross-demo scalar agreement after state matching.

Do not merge radii.

---

## 31. Primary matched-state comparisons

For matched cross-demo branch pairs compare:

```text
scalar sensitivity agreement
top-1 subspace similarity
top-2 subspace similarity
```

under:

```text
normalized time
EXP4 scalar progress
primary state matching
robot-only descriptor
object-only descriptor
```

The question is causal-quantity replication, not visual alignment quality.

---

## 32. Inference unit

Demonstration remains the main independent unit.

Do not treat interventions, directions, radii, or simulator steps as independent samples.

Use hierarchical inference appropriate to the crossing/nesting structure.

---

## 33. Primary confirmatory decision rule

Freeze exact values before confirmatory q outcomes.

Proposed strong classification requires all:

```text
1. state matching improves scalar rank agreement
   over the better EXP4 baseline by median >= 0.15

2. cluster-bootstrap 95% CI lower bound > 0

3. at least 2/3 tasks have state-matched scalar Spearman >= 0.60

4. at least 2/3 tasks have median matched top-1 subspace similarity >= 0.70

5. at least 70% of confirmatory demos have
   cross-radius top-1 similarity >= 0.70

6. held-out random-direction prediction median rho >= 0.60

7. leave-one-demo-out scalar improvement remains positive

8. leave-one-task-out conclusion is qualitatively stable

9. multiplicity-adjusted confirmatory families pass FDR 0.05
```

Review and freeze before outcomes.

Do not alter because results almost pass.

---

## 34. Formal classification priority

Use:

```text
1. state_conditioned_replicated_anisotropic_criticality
2. state_alignment_only_without_subspace_replication
3. subspace_replication_without_scalar_sparsity
4. finite_radius_nonlinearity_dominates
5. trajectory_specific_criticality
6. no_confirmatory_support
```

Do not collapse to generic PASS/FAIL.

---

## 35. Interpretation tree

### A — state_conditioned_replicated_anisotropic_criticality

Allowed claim:

> Local q sensitivity is reproducible when conditioned on physical state, and its dominant anisotropic subspace is more stable than global time or scalar progress criticality.

Then an **oracle state-conditioned adaptive decision scheduler** becomes scientifically eligible.

Do not jump directly to latent RL.

### B — state_alignment_only_without_subspace_replication

State similarity predicts magnitude better, but sensitive q directions remain trajectory-specific.

Do not constrain control to a fixed q subspace.

### C — subspace_replication_without_scalar_sparsity

Pivot from sparse time to:

```text
local sensitive action subspace
subspace-constrained exploration
subspace-conditioned control updates
```

Continuous-time policy updates remain possible.

### D — finite_radius_nonlinearity_dominates

Treat EXP3/EXP4 as finite-radius causal-response experiments rather than Jacobian estimates.

Continue with radius-calibrated operators.

### E — trajectory_specific_criticality

Conclude q-criticality under replayed open-loop continuation is strongly trajectory-specific.

Do not proceed to a universal sparse-decision controller.

Any later action-chunk / latent-action pivot must define a new estimand.

---

## 36. GPU requirement

Use RTX 4090 for eligible numerical workloads.

CPU remains the simulator truth path.

GPU workloads should include where appropriate:

```text
dense descriptor distances
batched covariance / Mahalanobis calculations
state-matching cost tensors
finite-difference operator assembly
Gram matrices
SVD/eigendecomposition
projector/principal-angle calculations
held-out direction prediction
hierarchical bootstrap
permutation tests
cross-fitting aggregation
```

Do not move MuJoCo/robosuite stepping to GPU.

---

## 37. GPU audit and equivalence

Record:

```text
torch version
torch CUDA version
torch.cuda.is_available()
GPU name
device index
driver
VRAM
dtype
selected device
```

Run CPU/GPU equivalence on a frozen calibration set covering all tasks and primary radii.

Compare:

```text
descriptor scaling
Mahalanobis distances
match costs
selected matches
central finite differences
Gram matrices
singular values
projectors
subspace similarities
held-out prediction
bootstrap
permutation
rank statistics
```

Prefer float64.

No automatic CPU fallback in a run labeled GPU.

If equivalence fails:

```text
STOP GPU formal path
CPU remains source of truth
```

---

## 38. Required development outputs

Using demos 3–9 create:

```text
development_state_descriptor.parquet
development_pairwise_match_tables.parquet
development_baseline_comparisons.parquet
development_state_matching_report.md
```

Explicitly compare:

```text
time
scalar progress
state matching
robot-only
object-only
```

---

## 39. Confirmatory availability audit

Before q outcomes audit demos 10–19:

```text
file existence
episode length
model XML
same-runtime final success
snapshot round trip
corrected-D restore
state descriptor completeness
task geometry availability
```

Any unresolved cohort deficit after exhausting the amended ordered completion rule
stops the confirmatory run.

---

## 40. Confirmatory local references

At every boundary save:

```text
corrected-D snapshot
shared physical descriptor
task-specific descriptor
contacts
predicate
q/qvel
EEF
gripper
task object state
```

Only references may be used for matching/prototype fitting.

---

## 41. Required frozen manifests

Before outcomes commit and hash:

```text
crossfit_fold_manifest.json
state_descriptor_schema.json
state_descriptor_scaling.json
state_distance_spec.json
prototype_manifest.json
matching_spec.json
confirmatory_branch_manifest.json
match_rejection_manifest.json
direction_basis_manifest.json
heldout_direction_manifest.json
radius_manifest.json
large_radius_calibration_subset.json
effect_channel_schema.json
effect_normalization.json
signed_output_vector_spec.json
scalar_metric_spec.json
operator_metric_spec.json
linearity_gate.json
statistical_analysis_plan.json
gpu_analysis_spec.json
scientific_decision_rule.json
```

All must record schema version, freeze time, project SHA, source hashes/runs, and outcome-blind declaration.

---

## 42. Full matched-zero gate

Run zero controls for all confirmatory branches.

No q outcomes until the complete confirmatory zero cohort passes.

---

## 43. Formal q sweep

Execute every preregistered branch × direction × sign × radius combination.

Recommended sharding:

```text
task / demo / radius
```

Each shard must be immutable and independently auditable.

Never patch a completed run in place.

---

## 44. Raw lock before analysis

After q outcomes:

```text
hash raw artifacts
write raw_hash_manifest.json
commit hashes
```

Only then start confirmatory inference.

---

## 45. Formal analysis order

```text
1. execution validity
2. zero floor
3. radius-specific scalar sensitivity
4. operator construction
5. cross-radius linearity
6. state-match scalar replication
7. top-1/top-2 subspace replication
8. held-out direction prediction
9. cross-fitting aggregation
10. LODO / LOTO
11. terminal relevance
12. secondary event analysis
13. frozen classification
```

Do not inspect only promising task subsets first.

---

## 46. Research log

Every meaningful step must append to the research log.

Record:

```text
timestamp
EXP5 stage
research question
pre-run expectation
run ID
project SHA
LIBERO SHA
robosuite version
MuJoCo version
Python version
machine
GPU/CUDA
GPU actually used
exact command
exact config
development/confirmatory cohort
crossfit fold
descriptor version
matching hash
direction/radius hashes
tests
metrics
warnings
failures
interpretation
alternative explanations
claim impact
next experiment
```

Never delete negative or superseded results.

---

## 47. Evidence table

After EXP5 update `paper/evidence_table.md`.

Add separate rows for:

```text
state-conditioned scalar replication
state-matching improvement
top-1 subspace replication
top-2 subspace replication
radius linearity
held-out direction prediction
trajectory specificity
task specificity
eligibility for sparse control
```

Keep EXP1–EXP4 evidence intact.

---

## 48. Required project structure

```text
experiments/
└── exp5_state_conditioned_anisotropic/
    ├── EXP5.md
    ├── README.md
    ├── configs/
    └── manifests/

scripts/
└── exp5/
    ├── audit_confirmatory_demos.py
    ├── generate_confirmatory_references.py
    ├── build_state_descriptors.py
    ├── fit_state_matcher.py
    ├── build_crossfit_folds.py
    ├── build_prototypes.py
    ├── freeze_protocol.py
    ├── validate_gpu_backend.py
    ├── run_zero_controls.py
    ├── run_interventions.py
    ├── assemble_operators.py
    ├── analyze_linearity.py
    ├── analyze_state_matching.py
    ├── analyze_subspaces.py
    ├── analyze_heldout_direction.py
    ├── analyze_terminal_relevance.py
    └── generate_report.py
```

Reuse existing EXP2–EXP4 modules whenever possible.

---

## 49. Required formal artifacts

At minimum:

```text
zero_controls.parquet
interventions.parquet
per_step_effects.parquet
signed_output_vectors.parquet or indexed arrays
scalar_branch_summary.parquet
operator_summary.parquet
operator_matrices or indexed binary arrays
subspace_similarity.parquet
cross_radius_linearity.parquet
state_match_tables.parquet
crossfit_results.parquet
heldout_direction_prediction.parquet
terminal_relevance.parquet
gpu_audit.json
gpu_cpu_equivalence.json
scientific_decision.json
failure_examples.json
raw_hash_manifest.json
```

---

## 50. Required plots

Generate at least:

```text
state_match_distance_distribution.png
time_vs_progress_vs_state_alignment.png
scalar_replication_by_alignment.png
top1_subspace_similarity_by_task.png
top2_subspace_similarity_by_task.png
operator_spectrum_by_state_prototype.png
leading_eigenvalue_share_by_state.png
cross_radius_spectral_consistency.png
cross_radius_subspace_similarity.png
sign_asymmetry_by_radius.png
heldout_direction_prediction.png
state_conditioned_scalar_vs_terminal_outcome.png
state_conditioned_spectral_vs_terminal_outcome.png
task_specificity_summary.png
lodo_loto_summary.png
gpu_cpu_equivalence.png
```

No placeholder plots.

---

## 51. Required tests

Retain prior tests and add known-answer coverage for:

```text
state descriptor extraction
feature scaling
physical-floor handling
shrinkage covariance
Mahalanobis distance
monotone matching
Sakoe-Chiba constraint
match rejection
prototype selection
crossfit leakage prevention
branch coverage
basis orthogonality
radius handling
sign pairing
signed output vector
central finite difference
Gram matrix
spectral norm
effective rank
projector construction
top-1/top-2 similarity
held-out direction prediction
cross-radius linearity
GPU/CPU equivalence
bootstrap
permutation
BH FDR
raw hash lock
run non-overwrite
```

Add synthetic cases with exactly known subspace similarity.

---

## 52. Stop conditions

Stop rather than improvise if:

```text
demos 10–19 plus the ordered unused completion pool cannot provide 10 qualified
references per task
confirmatory references remain below 10 per task after the amended completion rule
corrected Condition D regresses
state features cannot be exactly audited
matching requires confirmatory outcome information
cross-fitting leaks held-out demos
basis/radius/sign violates joint limits
zero floor fails
non-arm state changes
NaN/Inf appears
operator output vector needs outcome-dependent redefinition
linearity thresholds need post-hoc changes
GPU and CPU disagree beyond tolerance
formal branch deletion becomes necessary
classification requires threshold changes
```

---

## 53. Git discipline

Use atomic commits.

Suggested boundaries:

```text
EXP5 scaffold
confirmatory availability audit
state descriptor
state matching
crossfit folds
prototype/branch selection
direction/radius design
GPU backend
protocol freeze
confirmatory zero gate
formal q intervention code
raw lock
operator assembly
linearity analysis
state alignment analysis
subspace analysis
held-out direction analysis
EXP5 report
evidence table
```

Do not push GitHub unless explicitly requested.

---

## 54. Exact execution order

```text
read prior project/report/evidence/log files
inspect repository and Git state
run current tests
run pip check

EXP5-0:
audit development and confirmatory cohort availability

EXP5-1:
build physical state descriptor on references only

EXP5-2:
develop state matching using EXP4 demos 3–9

EXP5-3:
freeze cross-fitting, matching, prototype rules

EXP5-4:
generate confirmatory local references for demos 10–19

if confirmatory reference gate fails:
    STOP

EXP5-5:
select state-coverage branches using reference-only prototypes

EXP5-6:
freeze basis, held-out direction, radii, calibration subset

EXP5-7:
freeze signed output vector and operator metrics

EXP5-8:
freeze local-linearity gate

EXP5-9:
freeze statistical plan and scientific decision rule

EXP5-10:
validate GPU backend against CPU reference

if GPU equivalence fails:
    CPU remains truth path
    diagnose before formal GPU inference

commit/hash all preregistration manifests
run full tests

EXP5-11:
run matched-zero gate across all confirmatory branches

if zero gate fails:
    STOP

EXP5-12:
run all 0.0025 / 0.005 interventions

EXP5-13:
run preregistered 0.01 calibration subset

lock/hash raw artifacts

EXP5-14:
assemble radius-specific response operators

EXP5-15:
cross-radius linearity analysis

EXP5-16:
state-conditioned scalar replication

EXP5-17:
top-1/top-2 subspace replication

EXP5-18:
held-out random-direction prediction

EXP5-19:
crossfit confirmatory aggregation

freeze primary classification

EXP5-20:
terminal relevance and secondary event analysis

write reports/exp5_report.md
write reports/next_exp_from5.md
update research_log
update paper/evidence_table.md
run full tests
run pip check
commit final report
STOP
```

Do not automatically start sparse or latent RL.

---

## 55. Required final EXP5 report

`reports/exp5_report.md` must answer:

```text
Did demos 10–19 exist and pass same-runtime validation?
How many confirmatory demos/branches/interventions completed?
Did corrected Condition D remain exact?
Did state matching improve scalar cross-demo replication over time and progress?
Did the improvement replicate under cross-fitting?
Did top-1 sensitive subspaces replicate?
Did top-2 subspaces replicate?
Was response more stable at 0.0025 than 0.005?
What fraction of branches passed the local-linearity gate?
What happened at 0.01 on the calibration subset?
Can the basis operator predict the held-out random direction?
How state-dependent were principal q directions?
How task-specific were results?
Did scalar or spectral sensitivity better predict terminal consequences?
Was GPU actually used?
Did CPU/GPU equivalence pass?
Which formal EXP5 classification was selected?
What is the strongest allowed claim?
Is an oracle adaptive scheduler scientifically eligible?
Is latent RL scientifically eligible?
What should EXP6 test?
```

---

## 56. next_exp_from5.md logic

If:

```text
state_conditioned_replicated_anisotropic_criticality
```

recommend an **oracle state-conditioned adaptive decision scheduler** before learned sparse policy or latent RL.

If:

```text
subspace_replication_without_scalar_sparsity
```

recommend **subspace-constrained exploration/control** rather than time sparsification.

If:

```text
finite_radius_nonlinearity_dominates
```

recommend a smaller-radius / radius-adaptive response study.

If:

```text
trajectory_specific_criticality
```

record the q-criticality mainline as insufficient for a universal controller and preregister a different action-chunk / latent-action estimand.

Do not recommend latent RL merely because it is the long-term goal.

---

## 57. End-of-session Codex response

At the end of every session report:

```text
current EXP5 stage
files changed
commits created
exact commands run
tests passed / failed
pip-check status
development demos used
confirmatory demos audited
confirmatory references passed
branches frozen
zero-control status
interventions completed by radius
GPU detected
GPU workloads actually executed
GPU/CPU equivalence status
state-matching status
operator/subspace status
current scientific result if legally available
unresolved issues
strongest allowed claim
claims still forbidden
single highest-value next action
```

---

## 58. Final scientific principle

EXP3 and EXP4 establish reproducible temporal non-uniformity, but they also show that the causal response cannot be summarized reliably by:

```text
one global time axis
one scalar progress coordinate
one globally stable q direction
one simple contact/gripper event trigger
```

EXP5 must determine whether the stable causal object is instead:

```text
a state-conditioned scalar sensitivity field
a state-conditioned low-dimensional sensitive subspace
a finite-radius nonlinear response geometry
or
a fundamentally trajectory-specific effect
```

Only the first two outcomes justify continuing toward adaptive sparse control.
