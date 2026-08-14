# EXP6 Codex Prompt — Multi-Radius Convergence and Trust-Region Identification of q Response

**Project:** Decision-Sparse Latent RL for Contact-Rich Manipulation  
**Experiment:** EXP6 — Multi-Radius Convergence and Trust-Region Identification of q Response  
**Date:** 2026-08-14  
**Status:** Ready for preregistration; do not execute before protocol freeze  
**Source classification:** `finite_radius_nonlinearity_dominates`  
**Primary purpose:** determine whether the finite-difference q response converges as perturbation radius approaches zero above the exact matched-zero floor, and identify an empirical local trust region where convergence exists.  
**Simulation:** exact corrected-D CPU MuJoCo / robosuite path validated in EXP2–EXP5.  
**GPU:** RTX 4090 float64 for eligible operator, spectrum, bootstrap, permutation, and trust-region analysis after CPU/GPU equivalence validation.  
**Forbidden in EXP6:** sparse controller training, latent RL, OpenPI, SVM, learned scheduler, learned event trigger, real-robot RL.

---

## 1. Read before modifying code

Read completely:

```text
PROJECT.md
experiments/exp1_decision_sparsity/EXP1.md
experiments/exp2_simulator_reconciliation/EXP2.md
experiments/exp3_time_indexed_criticality/EXP3.md
experiments/exp4_replicated_progress_criticality/EXP4.md
experiments/exp5_state_conditioned_anisotropic/EXP5.md
reports/exp1_report.md
reports/exp2_report.md
reports/exp3_report.md
reports/exp4_report.md
reports/exp5_report.md
reports/next_exp_from5.md
paper/evidence_table.md
the latest research_log entry
```

Inspect and reuse the exact passing implementations/manifests from EXP2–EXP5. Do not duplicate corrected-D snapshot restore, zero-twin validation, q masking, signed physical-output extraction, operator construction, GPU equivalence code, or raw artifact locking.

Never guess paths, APIs, task IDs, runtime object paths, MuJoCo IDs, controller fields, contact names, joint ranges, state layouts, or manifest schemas.

---

## 2. EXP5 result that defines EXP6

EXP5 completed:

```text
30 qualified same-runtime trajectories
480 branches
16,896 q interventions
658,288 future-step records
1,056 radius-specific response operators
```

Matched-zero remained exact over all 480 branches.

Formal classification:

```text
finite_radius_nonlinearity_dominates
```

Key failures:

```text
local-linearity pass:
21 / 480 branches = 4.375%

small/main top-1 similarity:
Drawer 0.174
Bowl   0.139
Stove  0.317

demos with median small/main top-1 >= 0.70:
1 / 30

main/large top-1 similarity:
0.1617

main/large relative spectral discrepancy:
0.4369
```

State matching did not improve replication:

```text
median improvement = -0.0363
95% CI = [-0.2795, 0.0334]
BH q = 0.9950
```

Held-out direction ranking was the one positive diagnostic:

```text
median rho = 0.6265
```

but signed vector errors remained too large for a stable local linear interpretation.

The next question is therefore not better state matching. The next question is whether a stable local response exists at smaller radii.

---

## 3. Core scientific question

> As Panda arm-q perturbation radius approaches zero above the exact matched-zero floor, do response magnitude and physical-q principal subspaces converge?

EXP6 distinguishes:

```text
A. finite-radius curvature
B. contact-mode discontinuity
C. numerical under-resolution
D. persistent nonsmooth / trajectory-specific response
```

---

## 4. Research questions

1. Do adjacent-radius operators become more similar as radius decreases?
2. Does spectral scale converge?
3. Do top-1 and top-2 q-sensitive subspaces stabilize?
4. Does antithetic sign asymmetry shrink?
5. Does held-out direction prediction improve?
6. Is convergence failure associated with contact-mode divergence?
7. What is the largest branch-specific radius interval satisfying all convergence criteria?

---

## 5. Claim boundary

Supported:

```text
temporal non-uniformity replicates
q response is anisotropic
finite-radius response is measurable
corrected-D branching is exact
0.0025–0.01 response geometry is radius dependent
```

Not supported:

```text
a radius-invariant Jacobian exists
a stable sensitive q subspace exists
state matching recovers a reusable subspace
oracle scheduler is eligible
latent RL is eligible
contact is a validated decision trigger
```

EXP6 is an estimand-validation experiment.

---

## 6. Frozen cohort

Reuse the 30 qualified EXP5 trajectories:

```text
Drawer:
demos 10–16, 18, 19, 20

Bowl:
demos 10–19

Stove:
demos 10–19
```

Do not alter this cohort based on EXP5 q outcomes.

Drawer demo17 remains preserved as a failed qualification case.

---

## 7. Outcome-blind branch subset

Select:

```text
8 branches per demo
30 demos
= 240 branches
```

using unperturbed reference states only.

Do not use EXP5 criticality, operator norm, success flips, linearity outcomes, or held-out prediction quality to choose branches.

Stratify reference-only coverage over:

```text
normalized time
task progress
contact / no-contact
gripper state
predicate phase
```

Freeze and save:

```text
exp6_branch_manifest.json
```

with task, demo, branch index, normalized time, progress, contact state, gripper state, predicate state, prototype/stratum, selection distance, and deterministic replacement reason.

---

## 8. Radius ladder

Primary ladder:

```text
0.000625 × joint range
0.001250 × joint range
0.002500 × joint range
0.005000 × joint range
```

Optional:

```text
0.0003125 × joint range
```

may be added only if a preregistered pre-outcome resolution calibration shows that its antithetic signal is reliably measurable above the zero floor.

Do not add or remove radii after formal outcomes.

---

## 9. Direction design

At every branch and radius use:

```text
7 orthonormal basis directions
1 held-out random direction
both signs
```

Use a new frozen seed for randomized basis/order/held-out direction generation.

Coordinate system remains joint-range-scaled Panda q.

---

## 10. Formal intervention budget

Primary:

```text
240 branches × 4 radii × 8 directions × 2 signs
= 15,360 interventions
```

plus matched-zero continuations.

Optional 0.0003125 radius, if calibration passes, adds:

```text
3,840 interventions
```

Do not add it unless the calibration gate passes before the formal sweep.

---

## 11. Numerical-resolution calibration

Before formal EXP6, freeze:

```text
2 branches per task
= 6 calibration branches
```

selected reference-only and covering contact/non-contact and multiple trajectory phases.

At every calibration branch and proposed radius:

```text
at least 4 matched-zero continuations
repeat every signed intervention twice
```

Measure:

```text
zero floor
q injection precision
repeatability
operator repeatability
direction/sign rank repeatability
signal-to-floor
```

---

## 12. Calibration gates

Require:

```text
all zero repetitions pass corrected-D

non-arm INTEGRATION Linf <= 1e-12

q injection matches requested delta within frozen tolerance

all states finite

repeat scalar effects agree within frozen tolerance

repeat spectral/operator summaries agree within frozen tolerance

direction/sign ranking is deterministic

response norm >= 100 × measured zero-floor upper bound
```

If 0.000625 is not resolvable:

```text
STOP formal EXP6
classification candidate:
numerical_resolution_prevents_local_limit_test
```

Do not silently remove the radius and continue.

---

## 13. Zero-floor estimator

Do not assume the floor stays exactly zero.

From repeated zero controls preserve:

```text
integration-state floor
signed physical-output floor
scalar-effect floor
operator-spectral floor
```

If exact zero is observed, store exact zero.

If ratios require a denominator, use a frozen resolution constant defined before formal outcomes.

---

## 14. Corrected-D substrate

Use unchanged:

```text
mjSTATE_INTEGRATION
explicit OSC state
robot buffers
environment timing/done
PandaGripper.current_action
```

Use the exact validated pre-policy boundary.

Do not insert extra `mj_forward`.

Do not alter controller targets.

---

## 15. Matched-zero gate

At all 240 formal branches run at least 2 matched-zero continuations.

Calibration branches require at least 4.

Hard gate:

```text
final-success agreement = 100%
all states finite
INTEGRATION L2 median <= 1e-10
INTEGRATION L2 P95 <= 1e-8
INTEGRATION L2 max <= 1e-6
terminal object-pose P95 <= 1e-6
```

Systematic failure stops EXP6.

---

## 16. Intervention gate

For every branch/radius/direction/sign:

```text
modify only Panda arm qpos
verify q indices from runtime
verify exact joint limits
verify requested radius
preserve non-arm integration state
replay identical future action suffix
```

Require:

```text
non-arm INTEGRATION Linf <= 1e-12
all perturbed q valid
all arrays finite
all required radii/directions/signs present
no post-outcome branch deletion
```

---

## 17. Signed physical-output vector

Reuse the exact EXP5 signed physical output unless a preregistered implementation bug is discovered before formal outcomes.

Preserve signed:

```text
delta arm q
delta arm qvel
delta EEF position
delta EEF orientation representation
delta task-object position
delta task-object orientation representation
```

Do not redefine the output vector to improve convergence.

---

## 18. Radius-resolved response operator

For radius `r` and basis direction `d_j`:

```text
J_r[:, j] =
[y(q + r d_j) - y(q - r d_j)]
/
(2r)
```

Construct:

```text
G_r = J_r^T J_r
```

Report per radius:

```text
spectral norm
Frobenius norm
leading eigenvalue share
effective rank
top-1 projector
top-2 projector
```

Use the term `finite-radius response operator` until convergence criteria are satisfied.

---

## 19. Adjacent-radius comparisons

Compare:

```text
0.000625 vs 0.00125
0.00125  vs 0.0025
0.0025   vs 0.005
```

If optional radius passes:

```text
0.0003125 vs 0.000625
```

Report:

```text
top-1 projector similarity
top-2 projector similarity
relative spectral discrepancy
Frobenius discrepancy
sign asymmetry
held-out vector relative error
held-out scalar prediction/rank
```

---

## 20. Projector similarity

Reuse the EXP5 basis-invariant metric:

```text
P = V_k V_k^T

similarity(P_a, P_b)
=
1 - ||P_a - P_b||_F / sqrt(2k)
```

with:

```text
k=1 primary
k=2 secondary
```

Add regression tests.

---

## 21. Sign asymmetry

Reuse the EXP5 asymmetry definition if already frozen and suitable.

Otherwise freeze before outcomes a normalized antithetic asymmetry such as:

```text
A =
||response_plus + response_minus||
/
(||response_plus|| + ||response_minus|| + eps)
```

where `eps` is frozen before outcomes.

High asymmetry indicates curvature, mode switching, or other finite-radius nonlinear behavior.

---

## 22. Held-out direction prediction

Use seven-basis operators to predict the eighth direction.

Report by radius:

```text
signed vector relative error
scalar prediction error
per-demo rank rho
task-level rho
```

The important trend is whether prediction improves as radius decreases.

---

## 23. Empirical local trust region

For every branch estimate the largest adjacent-radius interval satisfying all:

```text
adjacent-radius top-1 similarity >= 0.80
adjacent-radius top-2 similarity >= 0.75
relative spectral discrepancy <= 0.20
antithetic sign asymmetry <= 0.25
held-out vector relative error <= 0.35
response norm >= 100 × measured zero-floor upper bound
```

Review and freeze these thresholds before formal outcomes.

If no adjacent pair passes:

```text
trust_region_status = unresolved
```

Do not force a radius.

---

## 24. Trust-region outputs

Save per branch:

```text
smallest resolvable radius
largest convergent radius
largest convergent interval
trust-region status
failed criteria
signal-to-floor
contact-mode behavior
```

Aggregate by task, demo, contact state, gripper state, and predicate phase.

---

## 25. Contact-mode analysis

EXP6 must separate ordinary curvature from hybrid contact-mode switching.

Freeze contact groups from unperturbed reference trajectories only.

At every branch/radius/sign preserve exact named contact-pair groups over the first:

```text
1
3
5
10
```

future policy steps.

Record whether antithetic perturbations preserve or diverge in contact mode.

---

## 26. Contact-mode divergence outputs

For each plus/minus pair save:

```text
same contact mode at step 1
same through step 3
same through step 5
same through step 10
first contact-mode divergence step
exact pair-set changes
```

Do not reduce to contact count alone if exact pair identities exist.

---

## 27. Contact-conditioned convergence test

Compare convergence failure between:

```text
contact-mode preserved
vs
contact-mode diverged
```

with demonstration-clustered inference.

This is an explanatory mechanism test, not authorization to use contact as a scheduler trigger.

---

## 28. Primary hypotheses

### H1 — Small-radius subspace convergence

At least 70% of demonstrations must have median:

```text
top-1 similarity >= 0.80
```

for:

```text
0.000625 vs 0.00125
```

with hierarchical 95% CI lower bound > 0.65.

### H2 — Scale convergence

At least 70% of branches must have:

```text
relative spectral discrepancy <= 0.20
```

for the same radius pair.

### H3 — Held-out prediction

At the smallest validated radius:

```text
demo-median held-out rank rho >= 0.65
median vector relative error <= 0.35
```

### H4 — Contact explanation

Convergence failure must be higher under contact-mode divergence with:

```text
demo-clustered CI excluding zero
BH-FDR < 0.05
```

Freeze the statistical plan before outcomes.

---

## 29. Formal classifications

Priority order:

```text
1. small_radius_local_operator_converges
2. contact_mode_conditioned_convergence
3. numerical_resolution_prevents_local_limit_test
4. nonsmooth_response_persists_below_exp5_radius
5. no_support
```

Do not collapse these into pass/fail.

---

## 30. Interpretation tree

### `small_radius_local_operator_converges`

Allowed claim:

> A measurable local q-response operator converges over a small-radius trust region for a substantial fraction of tested states.

Next experiment:

```text
state-conditioned replication using only inside-trust-region operators
```

Only after that passes may an oracle trust-region-aware scheduler be tested.

### `contact_mode_conditioned_convergence`

Allowed claim:

> Local operator convergence is conditional on contact-mode preservation; mode switching explains a substantial portion of nonsmooth response.

Next experiment:

```text
hybrid contact-mode-conditioned response field
```

Contact prediction must be validated independently before control use.

### `numerical_resolution_prevents_local_limit_test`

Allowed claim:

> Current numerical/runtime resolution is insufficient to establish the small-radius limit.

Do not infer nonsmoothness.

Next step may be higher-precision instrumentation or deterministic differentiable local-dynamics diagnostics.

### `nonsmooth_response_persists_below_exp5_radius`

Allowed claim:

> q response remains strongly radius dependent below the EXP5 scale and does not support a stable local Jacobian/subspace abstraction under the current causal continuation.

This closes the universal q-Jacobian/subspace scheduler mainline.

Any later action-chunk / latent-action project must define a new causal estimand.

---

## 31. GPU policy

Use RTX 4090 float64 for eligible analysis:

```text
operator assembly
Gram matrices
SVD/eigendecomposition
projector comparisons
radius comparisons
held-out prediction
bootstrap
permutation
trust-region estimation
contact-stratified aggregation
```

MuJoCo remains CPU.

No physics backend change.

---

## 32. GPU/CPU equivalence

Before formal GPU analysis, validate a frozen subset covering:

```text
all 3 tasks
all validated radii
contact/no-contact
multiple operator spectra
```

Compare CPU vs GPU for:

```text
central differences
Gram matrices
singular/eigen values
projectors
projector similarities
spectral discrepancy
sign asymmetry
held-out prediction
trust-region criteria
bootstrap
permutation
```

Use float64.

No automatic CPU fallback in a run labeled GPU.

If equivalence fails:

```text
STOP GPU formal path
CPU remains source of truth
```

---

## 33. Research log

Every meaningful step, including failures, must append to `research_log`.

Record:

```text
timestamp
EXP6 stage
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
branch manifest hash
radius manifest hash
direction manifest hash
contact schema hash
tests
zero floor
signal-to-floor
metrics
warnings
failures
interpretation
alternative explanations
claim impact
next experiment
```

Never delete negative results.

---

## 34. Evidence table

After EXP6 update `paper/evidence_table.md`.

Add separate rows for:

```text
small-radius subspace convergence
small-radius spectral convergence
held-out direction prediction by radius
empirical trust-region existence
contact-mode-conditioned convergence
numerical resolution
persistent nonsmoothness
scheduler eligibility
latent-RL eligibility
```

Keep EXP1–EXP5 evidence intact.

---

## 35. Required manifests

Freeze/hash before formal outcomes:

```text
exp6_cohort_manifest.json
exp6_branch_manifest.json
radius_manifest.json
direction_basis_manifest.json
heldout_direction_manifest.json
numerical_calibration_manifest.json
zero_floor_spec.json
signed_output_vector_spec.json
operator_metric_spec.json
asymmetry_metric_spec.json
trust_region_spec.json
contact_mode_schema.json
contact_divergence_spec.json
statistical_analysis_plan.json
gpu_analysis_spec.json
scientific_decision_rule.json
```

Each manifest must record schema version, freeze timestamp, project SHA, source hashes/runs, seed derivation, and outcome-blind declaration.

---

## 36. Project structure

Create:

```text
experiments/
└── exp6_radius_convergence/
    ├── EXP6.md
    ├── README.md
    ├── configs/
    └── manifests/

scripts/
└── exp6/
    ├── build_branch_subset.py
    ├── audit_radius_resolution.py
    ├── freeze_protocol.py
    ├── run_zero_controls.py
    ├── run_radius_interventions.py
    ├── assemble_radius_operators.py
    ├── analyze_radius_convergence.py
    ├── estimate_trust_regions.py
    ├── analyze_contact_modes.py
    ├── validate_gpu_backend.py
    └── generate_report.py
```

Reuse EXP5 modules where possible.

---

## 37. Required formal artifacts

At minimum:

```text
zero_controls.parquet
zero_repeatability.parquet
radius_interventions.parquet
per_step_effects.parquet
signed_output_vectors.parquet or indexed arrays
radius_operator_summary.parquet
operator_matrices or indexed arrays
adjacent_radius_comparisons.parquet
trust_region_summary.parquet
contact_mode_transitions.parquet
contact_conditioned_convergence.parquet
heldout_direction_prediction.parquet
signal_to_floor.parquet
gpu_audit.json
gpu_cpu_equivalence.json
scientific_decision.json
failure_examples.json
raw_hash_manifest.json
```

---

## 38. Required plots

Generate at least:

```text
signal_to_zero_floor_by_radius.png
repeatability_by_radius.png
spectral_norm_vs_radius.png
top1_similarity_adjacent_radii.png
top2_similarity_adjacent_radii.png
spectral_discrepancy_adjacent_radii.png
sign_asymmetry_vs_radius.png
heldout_prediction_error_vs_radius.png
trust_region_radius_distribution.png
trust_region_by_task.png
trust_region_by_contact_state.png
contact_mode_divergence_vs_convergence.png
contact_mode_divergence_timing.png
operator_spectrum_vs_radius.png
demo_level_small_radius_convergence.png
gpu_cpu_equivalence.png
```

No placeholder plots.

---

## 39. Required tests

Retain all prior tests and add:

```text
radius manifest determinism
small-radius q injection accuracy
radius ordering
repeatability calculation
zero-floor estimator
signal-to-floor
adjacent-radius pairing
projector similarity
spectral discrepancy
sign asymmetry
held-out prediction
trust-region logic
unresolved trust-region behavior
contact-pair grouping
contact-mode divergence timing
contact-stratified inference
GPU/CPU equivalence
raw hash locking
run non-overwrite
```

Add synthetic operators with exactly known convergence and non-convergence.

---

## 40. Stop conditions

Stop rather than improvise if:

```text
branch subset depends on EXP5 outcomes
0.000625 is not numerically resolvable
corrected-D zero twins regress
q injection accuracy fails
non-arm state changes
joint limits fail
NaN/Inf appears
operator output definition changes after outcomes
trust-region thresholds need post-hoc changes
contact-mode schema cannot be grounded
GPU/CPU disagreement exceeds tolerance
formal branch deletion becomes necessary
```

---

## 41. Git discipline

Use atomic commits.

Suggested boundaries:

```text
EXP6 scaffold
branch subset
radius calibration
zero-floor estimator
contact-mode schema
protocol freeze
GPU equivalence
formal zero controls
formal radius sweep
raw lock
operator assembly
radius-convergence analysis
trust-region analysis
contact-mode analysis
EXP6 report
evidence table
```

Do not push GitHub unless explicitly requested.

---

## 42. Exact execution order

```text
read all prior project/report/evidence/log files
inspect Git state
run current tests
run pip check

EXP6-0:
build outcome-blind 240-branch subset

EXP6-1:
audit contact/gripper/predicate strata

EXP6-2:
freeze radius ladder and optional-small-radius rule

EXP6-3:
run 6-branch numerical-resolution calibration
with repeated zeros and repeated signed interventions

if 0.000625 not resolvable:
    classify numerical_resolution_prevents_local_limit_test
    write report
    STOP

if optional 0.0003125 passes:
    include it only according to frozen rule

EXP6-4:
freeze zero-floor estimator
freeze operator metrics
freeze trust-region criteria
freeze contact-mode schema
freeze statistical plan
freeze scientific classification

EXP6-5:
validate GPU/CPU equivalence

commit/hash all frozen manifests
run full tests

EXP6-6:
run matched-zero controls for all 240 branches

if zero gate fails:
    STOP

EXP6-7:
run all formal radius interventions

lock raw artifacts
hash raw artifacts

EXP6-8:
assemble radius-specific operators

EXP6-9:
analyze adjacent-radius convergence

EXP6-10:
estimate branch-level empirical trust regions

EXP6-11:
analyze held-out direction prediction by radius

EXP6-12:
analyze contact-mode divergence and convergence

freeze primary classification

EXP6-13:
secondary task/contact/gripper/predicate stratified analysis

write reports/exp6_report.md
write reports/next_exp_from6.md
update research_log
update paper/evidence_table.md
run full tests
run pip check
commit final report
STOP
```

Do not automatically start scheduler or latent RL training.

---

## 43. Final report requirements

`reports/exp6_report.md` must explicitly answer:

```text
Was 0.000625 numerically resolvable?
Was optional 0.0003125 tested and why?
Did corrected-D remain exact?
How many interventions completed?
What was signal-to-floor at each radius?
Did spectral norm converge as radius decreased?
Did top-1 subspace converge?
Did top-2 subspace converge?
What fraction of branches had a valid empirical trust region?
What was trust-region radius by task?
Did sign asymmetry decrease with radius?
Did held-out direction vector prediction improve at smaller radii?
Was convergence better when contact mode was preserved?
Did contact-mode divergence significantly explain convergence failure?
Were contact effects task-specific?
Which formal classification was selected?
What is the strongest allowed claim?
Is a trust-region-aware oracle scheduler eligible?
Is latent RL eligible?
What should EXP7 test?
```

---

## 44. `next_exp_from6.md` logic

If:

```text
small_radius_local_operator_converges
```

recommend state-conditioned replication using only inside-trust-region operators, followed by an oracle trust-region-aware scheduler only if replication passes.

If:

```text
contact_mode_conditioned_convergence
```

recommend a hybrid contact-mode-conditioned response field before any scheduler.

If:

```text
numerical_resolution_prevents_local_limit_test
```

recommend improved precision/instrumentation or a deterministic differentiable local-dynamics diagnostic. Do not infer nonsmoothness.

If:

```text
nonsmooth_response_persists_below_exp5_radius
```

close the universal q-Jacobian/subspace scheduling mainline. Any later action-chunk or latent-action study must define a fresh causal estimand.

---

## 45. End-of-session Codex response

At the end of every session report:

```text
current EXP6 stage
files changed
commits created
exact commands run
tests passed / failed
pip-check status
branches frozen
radii frozen
numerical calibration status
zero floor
matched-zero gate
interventions completed per radius
GPU detected
GPU workloads executed
GPU/CPU equivalence
radius-convergence status
trust-region status
contact-mode analysis status
formal classification if legally available
strongest allowed claim
claims still forbidden
single highest-value next action
```

---

## 46. Final principle

EXP5 showed that the response operator at 0.0025–0.01 joint-range scale is not stable enough to be treated as a reusable Jacobian or sensitive q subspace.

EXP6 must answer the more fundamental question:

> **Does a stable local q-response limit exist at smaller, still measurable perturbation radii?**

Only if that limit exists should the project return to state-conditioned replication or controller design.
