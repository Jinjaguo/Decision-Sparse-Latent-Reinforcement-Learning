# EXP7 Codex Prompt — Contact-Mode-Conditioned Local Response and Boundary-Margin Validation

**Project:** Decision-Sparse Latent RL for Contact-Rich Manipulation  
**Experiment:** EXP7 — Contact-Mode-Conditioned Local Response and Boundary-Margin Validation  
**Date:** 2026-08-14  
**Source classification:** `contact_mode_conditioned_convergence`  
**Status:** Ready for preregistration; do not execute before protocol freeze  
**Primary purpose:** determine whether q-response becomes locally convergent and reusable when conditioned on an exact unilateral contact mode, distance to its boundary, and a short response horizon.  
**Simulation:** exact corrected-D CPU MuJoCo / robosuite path validated in EXP2–EXP6.  
**GPU:** may be used only after a newly frozen scale-aware CPU/GPU equivalence gate passes.  
**Forbidden:** sparse RL, latent RL, OpenPI, SVM, MPC, learned scheduler, learned event trigger, real-robot RL.

---

## 1. Required reading

Before touching code, read completely:

```text
PROJECT.md
experiments/exp1_decision_sparsity/EXP1.md
experiments/exp2_simulator_reconciliation/EXP2.md
experiments/exp3_time_indexed_criticality/EXP3.md
experiments/exp4_replicated_progress_criticality/EXP4.md
experiments/exp5_state_conditioned_anisotropic/EXP5.md
experiments/exp6_radius_convergence/EXP6.md
reports/exp1_report.md
reports/exp2_report.md
reports/exp3_report.md
reports/exp4_report.md
reports/exp5_report.md
reports/exp6_report.md
reports/next_exp_from6.md
paper/evidence_table.md
the latest research_log entry
```

Reuse the exact passing corrected-D, contact-pair logging, signed-response, operator, zero-control, raw-lock, and CPU/GPU audit implementations from EXP2–EXP6. Search the repository before creating duplicate code.

Never guess task IDs, demo IDs, geom names, contact groups, MuJoCo IDs, signed-distance APIs, predicates, q indices, joint limits, file paths, or manifest schemas.

---

## 2. EXP6 result that defines EXP7

EXP6 completed:

```text
30 demos
240 outcome-blind branches
5 radii
19,200 q interventions
774,560 future-step records
1,200 finite-radius response operators
960 adjacent-radius comparisons
9,600 exact named contact-mode comparisons
```

The causal substrate remained exact:

```text
240/240 matched-zero branches exact
maximum q injection error = 2.22e-16
maximum non-arm state change = 0
```

Numerical under-resolution was ruled out. Even the smallest admitted radius:

```text
0.0003125 × joint range
```

had enormous signal-to-floor and exact repeatability.

But the global local-linear interpretation failed:

```text
H1:
0/30 demos passed the small-radius top-1 convergence requirement

H2:
37.08% branches passed the spectral-discrepancy criterion

H3:
held-out rank rho = 0.7381
signed-vector relative error = 0.7049

resolved full trust regions:
7/240 branches
all 7 from Bowl
```

Median sign asymmetry remained near 0.9 across all radii.

The one supported mechanism was contact-mode divergence:

```text
failure when exact contact mode diverged = 100%
failure when contact mode was preserved = 92%
difference = +0.0778
95% CI = [0.0222, 0.1389]
BH q = 0.01525
```

This is a real mechanism effect, but contact preservation is clearly not sufficient.

---

## 3. Core EXP7 scientific question

> **Within a fixed unilateral contact mode, sufficiently far from its boundary and over a short horizon, does a reproducible small-radius q-response operator exist?**

EXP7 replaces the single smooth-field assumption with a hybrid-field hypothesis:

```text
(mode m, physical state x, boundary margin b, horizon H)
    -> local q-response operator
```

EXP7 is still a causal measurement experiment. It must not train a controller.

---

## 4. Research questions

1. Does one-step response converge across the two smallest radii when both perturbation signs preserve the exact reference contact mode?
2. Is convergence higher in mode-interior states than near contact boundaries?
3. Does convergence degrade from H=1 to H=3, H=5, and remaining horizon?
4. Can a seven-basis within-mode operator accurately predict an eighth held-out q direction?
5. Do same-mode / same-margin states across demonstrations have more similar sensitive subspaces than normalized-time or scalar-progress matches?
6. Can contact-mode preservation be predicted from pre-intervention reference features on held-out demonstrations?

---

## 5. Claim boundary at EXP7 start

Supported:

```text
temporal q-response non-uniformity replicates
response is anisotropic
global radius-invariant operators are unsupported
numerical under-resolution is not the explanation
contact-mode switching increases convergence failure
```

Not supported:

```text
contact preservation is sufficient
contact is a scheduler trigger
a reusable within-mode Jacobian exists
boundary margin explains the remaining failure
contact modes are predictable enough for control
oracle scheduler is eligible
latent RL is eligible
```

---

## 6. Independent cohort

Use a new same-runtime corrected-D cohort if available.

Target:

```text
10 newly qualified successful demos per task
30 demos total
```

for the same three task families.

Do not assume exact demo indices. Audit unused demonstrations programmatically.

Freeze an ordered qualification rule:

```text
scan unused demos in deterministic ascending order
accept only successful, finite, exact corrected-D references
preserve every failed qualification
stop after 10 accepted demos per task
```

If ten genuinely new demos cannot be obtained for a task, stop the claim of independent replication and write the shortfall. A reused cohort may be run only if explicitly labeled replication-with-reuse.

---

## 7. Exact contact-mode schema

At every unperturbed pre-policy boundary preserve:

```text
exact named geom-pair set
target–gripper pair set
target–environment pair set
gripper–environment pair set
task-object–environment pairs where relevant
exact task predicate
gripper command/state
PandaGripper.current_action
normal relative velocity for task-relevant pairs
signed separation / gap for preregistered pairs
normalized time
physical task progress
```

Primary mode identity must not be raw contact count.

Define a mode as a frozen tuple of physically grouped contact occupancies. Exact groups must come from runtime/source audit, not task-name guesses.

Save:

```text
contact_mode_schema.json
contact_pair_group_manifest.json
```

---

## 8. Signed contact separation / boundary margin

Audit the exact MuJoCo 3.2.3 mechanism available for signed geom separation or distance.

Do not assume an API. Inspect the installed bindings and active runtime model.

For every relevant pair record:

```text
geom A
geom B
signed gap
normal relative velocity
contact active?
```

If a reliable signed separation cannot be recovered, stop the boundary-margin analysis. Do not replace it with body-center distance.

---

## 9. Boundary-margin calibration

Before q outcomes, perform repeated corrected-D zero measurements and freeze:

```text
m_near
m_far
```

Define:

```text
interior:
|gap| >= m_far

near boundary:
m_near <= |gap| < m_far

boundary ambiguous:
|gap| < m_near
```

Derive `m_near` from repeated-zero geometric precision plus a frozen safety multiplier.

Derive `m_far` from a fixed physical threshold or a reference-only rule.

Never choose margins from q-response convergence outcomes.

Save:

```text
boundary_margin_spec.json
```

with units and derivation.

---

## 10. Branch design

Target:

```text
12 branches per demo
30 demos
= 360 branches
```

Select using reference features only.

Balance as far as possible over:

```text
contact / no-contact
interior / near / ambiguous boundary
early / middle / late progress
gripper-command state
predicate false / true phase
```

Use deterministic nearest-unused replacement.

Preserve empty strata and shortfalls rather than changing quotas after seeing responses.

---

## 11. Perturbation design

Use the three smallest EXP6-validated radii:

```text
0.0003125 × joint range
0.0006250 × joint range
0.0012500 × joint range
```

At every branch/radius:

```text
7 orthonormal q directions
1 held-out random direction
both signs
```

Use a newly frozen seed/order if required.

All q values must pass audited joint limits before formal outcomes.

---

## 12. Formal budget

```text
360 branches
× 3 radii
× 8 directions
× 2 signs
= 17,280 interventions
```

plus at least two matched-zero continuations per branch.

Do not compensate for cohort shortfall by outcome-dependent oversampling.

---

## 13. Horizon stratification — the main design change

For every intervention compute separate response summaries over:

```text
H=1 policy step
H=3 policy steps
H=5 policy steps
remaining horizon
```

Primary horizon:

```text
H=1
```

Remaining horizon is secondary historical comparison.

Do not average horizons.

The core hypothesis is that remaining-horizon responses cross later hybrid boundaries and therefore fail to behave like a local derivative.

---

## 14. Contact-mode outcomes remain outcomes

For every plus/minus pair classify separately at H=1, H=3, and H=5:

```text
A. both signs preserve reference mode
B. both signs enter the same new mode
C. signs enter different modes
D. only one sign changes mode
```

Never delete mode-changing branches.

Save exact contact-pair transitions.

---

## 15. Intent-to-perturb and conditional analyses

Both are mandatory:

```text
unconditional intent-to-perturb result:
all branches

conditional within-mode result:
only branches satisfying the frozen preserved-mode rule
```

Also report mode-changing and boundary-stratified results.

Never publish only the “clean” preserved-mode subset.

---

## 16. Corrected-D gate

Use unchanged:

```text
mjSTATE_INTEGRATION
explicit OSC state
robot buffers
environment timing/done
PandaGripper.current_action
```

Use the exact validated pre-policy boundary.

No extra `mj_forward`.

No controller-target repair.

---

## 17. Zero-control gate

At every formal branch run at least two zero continuations.

Require:

```text
success agreement = 100%
all arrays finite
integration L2 median <= 1e-10
integration L2 P95 <= 1e-8
integration L2 max <= 1e-6
terminal object-pose P95 <= 1e-6
contact-mode identity exact under zero twins
signed-gap repeatability within the frozen precision gate
```

Systematic failure stops EXP7.

---

## 18. q-intervention gate

For all formal interventions require:

```text
q injection error <= 1e-15
non-arm INTEGRATION Linf <= 1e-12
all q within audited limits
all arrays finite
all directions/signs/radii present
no post-outcome branch deletion
```

---

## 19. Signed physical response and horizon operators

Reuse the EXP5/EXP6 signed physical output definition.

Do not redefine it to improve EXP7.

For each horizon H and radius r:

```text
J_{r,H}[:, j]
=
[y_H(q + r d_j) - y_H(q - r d_j)]
/
(2r)

G_{r,H}
=
J_{r,H}^T J_{r,H}
```

Preserve:

```text
spectral norm
Frobenius norm
leading eigenvalue share
effective rank
top-1 projector
top-2 projector
sign asymmetry
held-out prediction error
```

Call these finite-radius response operators unless the within-mode convergence gate passes.

---

## 20. Primary convergence population

Primary radius pair:

```text
0.0003125 vs 0.000625
```

Primary horizon:

```text
H=1
```

Primary population:

```text
branches where both antithetic signs preserve the exact reference mode for one step
```

Report:

```text
top-1 projector similarity
top-2 projector similarity
relative spectral discrepancy
sign asymmetry
held-out vector relative error
```

---

## 21. H1 — Within-mode one-step convergence

Among preserved-mode one-step branches, require at least 70% of demonstrations to have demo-median:

```text
top-1 similarity >= 0.80
top-2 similarity >= 0.75
spectral discrepancy <= 0.20
sign asymmetry <= 0.25
```

Also require hierarchical 95% CI lower bound for demo-median top-1 > 0.65.

Freeze aggregation before outcomes.

---

## 22. H2 — Boundary-margin mechanism

Within the preserved-mode population compare:

```text
interior
vs
near + ambiguous boundary
```

Convergence must be higher in interior states with:

```text
demo-clustered CI excluding zero
BH q < 0.05
```

Do not redefine strata after outcomes.

---

## 23. H3 — Mode-conditioned held-out prediction

At smallest radius and H=1 in preserved-mode branches require:

```text
demo-median rank rho >= 0.65
median signed-vector relative error <= 0.35
```

Report overall and by task.

---

## 24. H4 — Cross-demo mode-conditioned reuse

Within the same frozen contact-mode group and boundary-margin stratum:

compare top-1 subspace similarity against the better of:

```text
normalized time
EXP4 scalar progress
```

Require median improvement >= 0.15 with hierarchical CI lower bound > 0.

Do not compare incompatible contact modes in the primary test.

---

## 25. H5 — Contact-mode predictability

Train a separate reference/pre-intervention mode-transition predictor.

Allowed features may include only audited pre-intervention variables such as:

```text
signed gap
normal relative velocity
current contact mode
EEF relative pose
q/qvel
gripper state/current_action
task progress
candidate q direction
candidate radius
```

No response vectors, criticality labels, success flips, or terminal outcome features may be inputs.

Primary target:

```text
mode preserved at next step?
```

Use cross-fitting across demonstrations.

Freeze folds before labels are analyzed.

Report:

```text
AUROC
AUPRC
Brier score
expected calibration error
reliability curve
sensitivity/specificity at a frozen threshold
```

Do not use accuracy alone.

Freeze H5 discrimination/calibration thresholds before formal predictor evaluation.

Passing H5 does **not** authorize control inside EXP7.

---

## 26. Horizon-effect analysis

Within preserved-mode branches compare:

```text
H=1
H=3
H=5
remaining horizon
```

Predeclare whether expected convergence should monotonically degrade with horizon.

Report effect sizes and cluster-aware confidence intervals.

If H=1 converges while longer horizons fail, that supports a local hybrid-dynamics interpretation.

---

## 27. Exact-mode ablations

Predeclare:

```text
exact named mode vs raw contact count
target–gripper mode vs all contact groups
interior vs near vs ambiguous
H=1 vs H=3 vs H=5 vs remaining
conditional preserved-mode vs unconditional intent-to-perturb
mode-conditioned vs time
mode-conditioned vs scalar progress
```

Do not add ablations after formal outcomes.

---

## 28. Formal classification priority

Freeze before outcomes:

```text
1. within_mode_short_horizon_operator_converges
2. boundary_margin_explains_hybrid_nonsmoothness
3. contact_modes_explanatory_but_not_predictable
4. within_mode_nonsmoothness_persists
5. contact_schema_not_identifiable
6. no_support
```

Classification 1 requires:

```text
H1 + H3 + H4
```

H2 alone cannot authorize a controller.

H5 must pass independently before a later contact-aware scheduler is eligible.

---

## 29. Interpretation

### `within_mode_short_horizon_operator_converges`

Allowed claim:

> Within a fixed contact mode and short horizon, the small-radius q response admits a reproducible local operator that is more stable than the unconditional remaining-horizon response.

Next step: independent state/mode replication inside validated trust regions.

Only if mode prediction also passes may a later oracle contact-aware scheduler be tested.

### `boundary_margin_explains_hybrid_nonsmoothness`

Allowed claim:

> Proximity to unilateral contact boundaries explains a significant portion of local response nonconvergence.

Continue hybrid measurement/modeling. No control yet.

### `contact_modes_explanatory_but_not_predictable`

Contact is scientifically explanatory but not operationally reliable enough for scheduling.

### `within_mode_nonsmoothness_persists`

If same-mode, interior, H=1 responses still fail, close the universal q-response/Jacobian scheduler mainline.

Any future action-chunk / latent-action project must define a new causal estimand.

### `contact_schema_not_identifiable`

If exact mode/gap cannot be measured reproducibly, stop the hybrid-contact claim. Do not substitute raw contact count post hoc.

---

## 30. GPU policy

EXP6 correctly rejected formal GPU use under its frozen absolute gate.

EXP7 may introduce a new **scale-aware** equivalence protocol, frozen before formal analysis.

Freeze both:

```text
absolute tolerance
relative tolerance
```

for Gram/eigenspectrum/operator quantities.

This is a new audit design, not a reinterpretation of EXP6.

Use float64.

If the new gate fails, formal analysis remains CPU-only.

Never loosen thresholds after seeing failure.

---

## 31. Research log

Every meaningful step, including failures, must append to `research_log`.

Record:

```text
timestamp
EXP7 stage
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
cohort manifest hash
contact schema hash
boundary spec hash
branch manifest hash
radius/direction hashes
predictor fold hash
tests
metrics
warnings
failures
interpretation
alternative explanations
claim impact
next experiment
```

Preserve negative runs.

---

## 32. Evidence table

After EXP7 update `paper/evidence_table.md`.

Add rows for:

```text
within-mode one-step convergence
boundary-margin effect
horizon-dependent convergence
mode-conditioned held-out prediction
mode-conditioned cross-demo subspace reuse
contact-mode predictability
exact-mode vs contact-count ablation
hybrid-response-field support
scheduler eligibility
latent-RL eligibility
```

Keep EXP1–EXP6 evidence intact.

---

## 33. Required manifests

Freeze/hash before formal outcomes:

```text
exp7_cohort_manifest.json
contact_mode_schema.json
contact_pair_group_manifest.json
signed_gap_schema.json
boundary_margin_spec.json
exp7_branch_manifest.json
radius_manifest.json
direction_basis_manifest.json
heldout_direction_manifest.json
horizon_spec.json
signed_output_vector_spec.json
operator_metric_spec.json
preserved_mode_inclusion_spec.json
predictor_feature_schema.json
predictor_crossfit_manifest.json
predictor_decision_rule.json
statistical_analysis_plan.json
gpu_analysis_spec.json
scientific_decision_rule.json
```

Every manifest must record schema version, freeze time, project SHA, source hashes/runs, seed derivation, and outcome-blind declaration where applicable.

---

## 34. Required project structure

```text
experiments/
└── exp7_contact_mode_conditioned/
    ├── EXP7.md
    ├── README.md
    ├── configs/
    └── manifests/

scripts/
└── exp7/
    ├── audit_independent_cohort.py
    ├── audit_contact_geometry.py
    ├── calibrate_boundary_margin.py
    ├── build_branch_manifest.py
    ├── freeze_protocol.py
    ├── run_zero_controls.py
    ├── run_interventions.py
    ├── classify_mode_outcomes.py
    ├── assemble_horizon_operators.py
    ├── analyze_within_mode_convergence.py
    ├── analyze_boundary_margin.py
    ├── analyze_horizon_effect.py
    ├── fit_mode_predictor.py
    ├── analyze_mode_reuse.py
    ├── validate_gpu_backend.py
    └── generate_report.py
```

Reuse EXP6 modules where possible.

---

## 35. Required formal artifacts

At minimum:

```text
reference_contact_geometry.parquet
boundary_margin_calibration.parquet
zero_controls.parquet
interventions.parquet
per_step_effects.parquet
mode_outcomes.parquet
horizon_operator_summary.parquet
operator_matrices or indexed arrays
within_mode_convergence.parquet
boundary_margin_analysis.parquet
horizon_comparison.parquet
heldout_direction_prediction.parquet
mode_conditioned_crossdemo.parquet
mode_predictor_predictions.parquet
mode_predictor_metrics.json
gpu_audit.json
gpu_cpu_equivalence.json
scientific_decision.json
failure_examples.json
raw_hash_manifest.json
```

---

## 36. Required plots

Generate at least:

```text
signed_gap_repeatability.png
contact_mode_frequency.png
boundary_margin_distribution.png
within_mode_top1_by_radius.png
within_mode_top2_by_radius.png
within_mode_spectral_discrepancy.png
within_mode_sign_asymmetry.png
convergence_by_boundary_margin.png
convergence_by_horizon.png
intent_to_perturb_vs_conditional.png
mode_transition_categories.png
heldout_vector_error_within_mode.png
crossdemo_subspace_time_vs_progress_vs_mode.png
mode_predictor_roc_pr.png
mode_predictor_calibration.png
task_specific_hybrid_summary.png
gpu_cpu_equivalence.png
```

No placeholder plots.

---

## 37. Required tests

Retain all previous tests and add known-answer tests for:

```text
exact contact-pair grouping
contact-mode tuple construction
signed-gap extraction
signed-gap repeatability
boundary-margin classification
branch-stratum selection
mode-preserved classification
mode-change category logic
H=1/H=3/H=5 aggregation
within-mode inclusion rules
intent-to-perturb preservation
operator assembly by horizon
projector similarity
boundary-margin statistics
predictor leakage prevention
predictor fold determinism
predictor calibration metrics
GPU scale-aware equivalence
raw hash locking
run non-overwrite
```

Add synthetic hybrid examples with known mode transitions and known within-mode convergence.

---

## 38. Stop conditions

Stop rather than improvise if:

```text
new independent cohort cannot be obtained as preregistered
corrected-D regresses
exact contact groups cannot be identified
signed separation cannot be measured reproducibly
m_near/m_far require outcome-dependent tuning
branch selection uses EXP6 q outcomes
q injection/non-arm preservation fails
joint limits fail
NaN/Inf appears
conditional inclusion changes after outcomes
predictor leaks q-response labels into features
GPU equivalence requires post-hoc threshold changes
formal branch deletion becomes necessary
```

---

## 39. Git discipline

Use atomic commits.

Suggested boundaries:

```text
EXP7 scaffold
independent cohort audit
contact geometry audit
boundary-margin calibration
branch manifest
radius/direction/horizon manifests
predictor schema
protocol freeze
GPU equivalence
formal zero gate
formal intervention sweep
raw lock
mode classification
horizon operators
within-mode analysis
boundary analysis
predictor analysis
cross-demo reuse
EXP7 report
evidence table
```

Do not push GitHub unless explicitly requested.

---

## 40. Exact execution order

```text
read all prior project/report/evidence/log files
inspect Git state
run current tests
run pip check

EXP7-0:
audit new independent cohort

if independent cohort unavailable:
    report shortfall
    freeze labeled reuse fallback before continuing

EXP7-1:
audit exact contact groups and signed separation API

if contact schema not identifiable:
    classify contact_schema_not_identifiable
    report
    STOP

EXP7-2:
run repeated-zero boundary-margin calibration
freeze m_near and m_far

EXP7-3:
build 360 outcome-blind branch manifest

EXP7-4:
freeze 3 radii, directions, signs, H=1/3/5/remaining

EXP7-5:
freeze within-mode inclusion rule
freeze operator metrics
freeze H1–H5 statistics
freeze predictor features/folds/thresholds
freeze decision rule

EXP7-6:
validate scale-aware GPU/CPU equivalence

commit/hash all frozen manifests
run full tests

EXP7-7:
run formal matched-zero controls

if zero/contact-geometry gate fails:
    STOP

EXP7-8:
run all formal q interventions

lock raw artifacts
hash raw artifacts

EXP7-9:
classify mode outcomes

EXP7-10:
assemble H=1,3,5,remaining operators

EXP7-11:
analyze unconditional intent-to-perturb response

EXP7-12:
analyze within-mode H=1 convergence

EXP7-13:
analyze boundary-margin effect

EXP7-14:
analyze horizon degradation

EXP7-15:
analyze held-out direction prediction

EXP7-16:
analyze cross-demo mode-conditioned subspace reuse

EXP7-17:
fit/evaluate cross-fitted contact-mode predictor

freeze formal classification

EXP7-18:
run preregistered ablations and secondary analyses

write reports/exp7_report.md
write reports/next_exp_from7.md
update research_log
update paper/evidence_table.md
run full tests
run pip check
commit final report
STOP
```

Do not automatically begin scheduler, latent RL, MPC, or VLA experiments.

---

## 41. Final report must answer

```text
Was an independent 30-demo cohort obtained?
Did corrected-D remain exact?
Could exact contact modes be identified?
Was signed separation repeatable?
What m_near and m_far were frozen?
How many branches/interventions completed?
What fraction of antithetic pairs preserved mode at each radius/horizon?
Did H=1 within-mode response converge?
Did H=3/H=5 degrade?
Did remaining-horizon response remain nonconvergent?
Was convergence better in interior than near/ambiguous states?
Did held-out vector prediction pass within-mode?
Did mode-conditioned cross-demo subspace similarity improve over time/progress?
Could the reference-only mode-transition predictor discriminate and calibrate?
Did exact named modes outperform contact count?
Which task benefited most?
Which formal classification was selected?
What is the strongest allowed claim?
Is an oracle contact-aware scheduler eligible?
Is latent RL eligible?
What should EXP8 test?
```

---

## 42. `next_exp_from7.md` logic

If:

```text
within_mode_short_horizon_operator_converges
```

recommend independent state/mode replication restricted to the validated short-horizon trust region. Only if mode prediction also passes may a later oracle contact-aware scheduler be considered.

If:

```text
boundary_margin_explains_hybrid_nonsmoothness
```

continue hybrid contact-boundary modeling; do not train control yet.

If:

```text
contact_modes_explanatory_but_not_predictable
```

continue measurement/prediction work; control remains ineligible.

If:

```text
within_mode_nonsmoothness_persists
```

close the universal q-response/Jacobian scheduler mainline. A future action-chunk or latent-action project must define a new causal estimand, new baselines, and a new claim boundary.

If:

```text
contact_schema_not_identifiable
```

do not substitute contact count and claim hybrid structure.

---

## 43. End-of-session Codex response

At the end of each session report:

```text
current EXP7 stage
files changed
commits created
exact commands run
tests passed / failed
pip-check status
cohort status
contact schema status
boundary-margin calibration
branches frozen
zero-control gate
interventions completed
GPU audit/equivalence
mode-preservation statistics
within-mode convergence
boundary-margin result
horizon result
held-out prediction
cross-demo reuse result
mode-predictor metrics
formal classification if legally available
strongest allowed claim
claims still forbidden
single highest-value next action
```

---

## 44. Final scientific principle

EXP6 showed that contact-mode switching is a real contributor to q-response nonconvergence, but it explains only part of the failure.

The correct EXP7 question is not:

```text
"Is contact the critical event?"
```

It is:

> **Within a fixed unilateral contact mode, sufficiently far from its boundary and over a short horizon, does a reproducible local q-response operator exist?**

If the answer is no, stop treating q-criticality as a reusable smooth local object and move to a new causal abstraction rather than continuing to search for a Jacobian that the system does not support.
