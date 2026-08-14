# Next Experiment from EXP4: Cross-Fitted State-Conditioned Anisotropic Criticality

**Proposed experiment ID:** EXP5

**Status:** proposal only; do not execute automatically

**Source result:** EXP4 classification = replicated non-uniformity without aligned
sparse times

**Primary purpose:** determine whether local q-criticality is reproducible after
conditioning on physical state and local response geometry, rather than forcing
different demonstrations onto one scalar clock or progress axis.

## 1. Decision from EXP4

EXP4 answered its main questions asymmetrically:

- temporal non-uniformity replicated in 21 held-out demonstrations: median top-20
  mass 0.4507, hierarchical 95% CI `[0.4360, 0.5560]` versus uniform 0.25;
- progress-aligned rank agreement was high in absolute terms in every task, but its
  median improvement over normalized time was only 0.0437, p = 0.2549;
- only 5/21 demonstrations were stable across basis directions;
- held-out random-direction agreement was 0.4336;
- 87.44% of measured variance was residual/interactions;
- local operators were strongly anisotropic, while sign asymmetry was often large;
- frozen contact, gripper, and predicate events remained unsupported.

The result argues against training a sparse-time policy now. The likely missing
variable is not another global time transformation. Sensitivity appears to depend on
the joint physical state and on the direction and radius of perturbation.

The next mainline step should therefore be:

> Match comparable physical states across demonstrations using reference-only
> features, then test whether their local response spectra and principal subspaces
> replicate on a new held-out cohort and at multiple perturbation radii.

This remains a causal q-criticality experiment. It is not yet latent-RL training.

## 2. Main research question

Does conditioning on audited task and robot state reveal a reproducible,
low-dimensional local q-sensitivity geometry that was obscured by normalized-time
and scalar-progress alignment?

The experiment separates three possibilities:

1. **State-misalignment:** comparable causal states occur at different times and
   cannot be represented adequately by one scalar progress coordinate.
2. **Finite-radius nonlinearity:** the EXP4 `0.005 × joint range` perturbation crosses
   local regime boundaries, making sign-paired operators unstable.
3. **Genuine trajectory specificity:** even state-matched local operators do not
   replicate, so a global sparse decision abstraction is not supported.

## 3. Hypotheses

### H1: state-conditioned scalar replication

Cross-demo rank agreement of a state-conditioned sensitivity field will exceed the
better of normalized-time and EXP4 progress alignment by a demonstration-level
median of at least 0.15.

### H2: anisotropic subspace replication

At state-matched branches, the leading local response subspaces will agree across
demonstrations even when individual basis-direction ranks do not. The primary
subspace statistic will be a basis-invariant principal-angle similarity.

### H3: local linearity

For most branches, central finite-difference operator scale and leading subspace will
be consistent between radii 0.0025 and 0.005. If not, the response must be treated as
a finite-radius nonlinear object rather than a Jacobian estimate.

### H4: held-out confirmation

Any state-conditioned improvement developed on demonstrations 3–9 will reproduce on
new successful demonstrations 10–19. Development and confirmation outcomes may not
be pooled for the primary claim.

### H5: sparse control remains conditional

Sparse-decision training becomes eligible only if state-conditioned scalar and
subspace replication both pass. Concentration alone is insufficient.

## 4. Cohorts and leakage control

### 4.1 Development cohort

Use EXP4 demonstrations 3–9 only to implement and freeze the state descriptor,
distance metric, matching algorithm, interpolation rules, and decision thresholds.
Reference states may be used freely. EXP4 intervention outcomes may be used only for
the explicitly labeled development analysis.

### 4.2 Confirmatory cohort

Use demonstrations 10–19 from the same three datasets, subject to the same exact
same-runtime success and corrected-D gates.

| Quantity | Proposed value |
|---|---:|
| Tasks | 3 |
| New demos per task | 10 |
| Total new demos | 30 |
| State-coverage branches per demo | 16 |
| Total confirmatory branches | 480 |

Before any intervention outcome, audit that all requested demos exist and replay
successfully. If fewer than ten qualify for a task, stop and report the actual
availability; do not substitute a different task or silently reuse EXP4 demos.

### 4.3 Cross-fitting within confirmation

Use five deterministic folds within each task. For each held-out pair of demos,
normalization, nearest-neighbor bandwidth, state prototypes, and any learned
reference encoder must be fitted without those demos. Intervention outcomes are
never inputs to matching or representation learning.

## 5. Reference-only state representation

The recommended primary descriptor is deliberately physical and low-dimensional.
Every feature is computed from unperturbed references at pre-policy boundaries.

### 5.1 Shared robot features

- seven joint-range-normalized Panda arm positions;
- seven range/time-normalized Panda arm velocities;
- EEF position and a continuous six-dimensional rotation representation;
- EEF linear and angular velocity where the runtime exposes them deterministically;
- gripper opening/current action;
- contact-pair indicators reduced to audited task-relevant groups.

### 5.2 Drawer-specific features

- audited middle-drawer joint coordinate and velocity;
- EEF-to-handle relative position and orientation;
- gripper-to-handle distance;
- handle/gripper contact state;
- exact `Open` predicate state.

### 5.3 Stove-specific features

- audited button joint coordinate and velocity;
- EEF-to-button relative position and orientation;
- gripper/finger-to-button distances;
- button contact state;
- exact `TurnOn` predicate state.

### 5.4 Bowl-specific features

- bowl-to-plate planar displacement and vertical clearance;
- bowl pose and velocity relative to the plate;
- EEF pose relative to the bowl;
- bowl lift above initial height;
- gripper-bowl and bowl-plate contact indicators;
- exact `On` predicate state.

Binary predicate features must not dominate distance. Continuous features are scaled
using development-reference dispersion, with immutable physical floors to prevent
near-constant channels from exploding.

## 6. State matching

### 6.1 Primary method

Use a reference-only, task-specific Mahalanobis distance with shrinkage covariance,
followed by monotone optimal matching:

1. construct a dense descriptor sequence for every successful reference;
2. calculate scaling and shrinkage covariance from training folds only;
3. compute pairwise state costs;
4. find monotone matches with a fixed Sakoe–Chiba temporal window;
5. reject a match when its distance exceeds a predeclared training-reference
   quantile;
6. assign 16 branches per confirmatory demo by maximizing coverage of frozen state
   prototypes, not by looking at q outcomes.

The temporal window prevents pathological backward matches but does not reduce the
descriptor to time. All match distances, rejected matches, duplicates, and boundary
replacements must be saved.

### 6.2 Baselines

Compare the primary state matching against:

1. normalized time, exactly as in EXP4;
2. scalar physical progress, exactly as in EXP4;
3. shared robot features without task-object geometry;
4. task-object geometry without robot dynamics;
5. unconstrained nearest neighbor, as a diagnostic for the monotonicity constraint.

No baseline may be selected after viewing confirmatory interventions.

### 6.3 Optional representation learner

Only if the physical descriptor cannot provide adequate reference-state coverage,
add a small reference-only sequence encoder with reconstruction and temporal-cycle
consistency losses. It must be trained without rewards, successes beyond cohort
qualification, q-intervention effects, criticality labels, or branch selections.
The physical descriptor remains the primary analysis; the learned representation is
secondary until independently replicated.

## 7. Direction and radius design

### 7.1 Directions

Retain a complete reproducible seven-direction orthonormal basis and both signs at
every branch. Retain one held-out random direction for basis-aggregate prediction.
Generate all directions from a new PCG64 seed frozen after the scaffold commit and
before intervention outcomes.

### 7.2 Radii

Use a two-stage design:

- all 480 branches: radii `0.0025` and `0.005 × joint range`;
- a frozen 20% stratified calibration subset: also radius `0.01`.

The subset must be stratified by task, demo, and reference-state prototype before
outcomes. It may not target branches that appeared sensitive in EXP4.

The base budget is:

```text
480 branches × 8 directions × 2 signs × 2 radii = 15,360 interventions
```

The 0.01 calibration layer adds 1,536 interventions if 96 branches are selected.
Expected total: **16,896 interventions**, plus matched-zero continuations. This is
about 4.2 times EXP4 and should be sharded by task/demo/radius with resumable hashes.

If this budget is excessive, reduce branches from 16 to 12 before reducing the
number of confirmatory demonstrations or removing the smaller radius. Independent
replication and linearity diagnosis are more important than dense temporal sampling.

## 8. Restore protocol and hard execution gates

Use corrected Condition D unchanged:

1. restore exact MuJoCo `mjSTATE_INTEGRATION` at the pre-policy boundary;
2. restore controller, robot buffers, timing fields, and
   `PandaGripper.current_action`;
3. run two matched-zero continuations at every branch;
4. alter only the seven Panda arm-q values;
5. insert no extra `mj_forward` before the next recorded policy action;
6. replay the original remaining action suffix;
7. store the same six physical channels plus contact, predicate, terminal outcome,
   integration diagnostics, and signed physical-output vectors.

Hard gates:

- corrected-D exact cohort coverage;
- two zero continuations at every branch;
- zero median/P95 integration divergence and maximum within the existing gate;
- non-arm integration L-infinity change `<= 1e-12`;
- all directions, signs, and radii present;
- all joint limits valid and all arrays finite;
- no post-outcome branch removal;
- immutable raw hashes written before analysis.

Any gate failure stops formal inference.

## 9. Metrics

### 9.1 Scalar sensitivity

Preserve EXP4's six-channel normalization and remaining-horizon duration-normalized
effect. Report `S_RMS` separately for each radius. Do not average radii before the
linearity gate.

### 9.2 Basis-invariant local geometry

For each branch and radius, construct the signed central-difference response matrix
`J` and Gram matrix `G = J^T J`. Primary geometry summaries:

- spectral norm;
- Frobenius norm;
- leading eigenvalue share;
- effective rank;
- top-1 and top-2 right-singular subspaces;
- principal angles and projection-matrix similarity between matched branches.

Subspace comparisons are invariant to basis permutation and sign, avoiding EXP4's
fragile comparison of individual basis-direction ranks.

Recommended top-k similarity:

```text
similarity(P_a, P_b) = 1 - ||P_a - P_b||_F / sqrt(2k)
```

where `P = V_k V_k^T`. Freeze `k = 1` as primary and `k = 2` as secondary.

### 9.3 Local-linearity metrics

For radii `r1` and `r2`, report:

- relative spectral-norm discrepancy after division by radius;
- top-1 and top-2 subspace similarity;
- sign asymmetry at each radius;
- held-out random-direction effect predicted from the basis operator;
- monotonic saturation from 0.0025 to 0.005 to 0.01.

Call the operator a local linear approximation only where the frozen branch-level
linearity gate passes. Else call it a finite-radius response map.

### 9.4 Outcome relevance

Keep terminal object displacement, predicate divergence, and success flips separate.
Evaluate whether state-conditioned spectral sensitivity predicts them on held-out
demos. Do not redefine criticality using success flips after the fact.

## 10. Confirmatory statistics and decision rules

Freeze exact thresholds after reference and zero audits but before q outcomes. A
recommended strong-support rule requires all of:

1. state matching improves cross-demo scalar-rank agreement over the better EXP4
   baseline by median `>= 0.15`, with cluster-bootstrap 95% CI lower bound above 0;
2. at least two of three tasks have state-matched scalar Spearman `>= 0.60`;
3. median matched top-1 subspace similarity `>= 0.70` in at least two tasks;
4. at least 70% of confirmatory demos achieve within-demo cross-radius top-1
   subspace similarity `>= 0.70`;
5. held-out random-direction prediction median rho `>= 0.60`;
6. leave-one-demo-out scalar improvement remains positive and leave-one-task-out
   conclusions agree;
7. all multiplicity-adjusted confirmatory tests pass at FDR 0.05.

Classification priority:

1. `state_conditioned_replicated_anisotropic_criticality`;
2. `state_alignment_only_without_subspace_replication`;
3. `subspace_replication_without_scalar_sparsity`;
4. `finite_radius_nonlinearity_dominates`;
5. `trajectory_specific_criticality`;
6. `no_confirmatory_support`.

Do not collapse these outcomes into a single pass/fail label. They imply different
next steps.

## 11. GPU plan and equivalence

Use the RTX 4090 for dense matching, batched SVD/eigendecomposition, hierarchical
bootstrap, and permutation tests in float64. Preserve a CPU truth path for:

- descriptor scaling and selected pairwise distances;
- monotone-match reconstruction;
- central finite differences;
- principal-angle calculations;
- bootstrap medians with an explicit even-sample convention;
- exact ranks and tie handling.

Run CPU/GPU equivalence on at least one complete trajectory and a stratified subset
from all tasks/radii before formal analysis. No automatic device fallback is allowed
for a run labeled GPU. Hardware use is a computational optimization, not a change to
the scientific estimand.

## 12. Expected interpretations

### Outcome A: scalar and subspace replication both pass

The evidence would support a state-conditioned, anisotropic criticality field. The
next experiment could finally test an adaptive sparse decision scheduler that queries
the policy when predicted spectral sensitivity is high.

### Outcome B: subspaces replicate but scalar sparsity does not

The project should pivot from sparse times to low-dimensional sensitive action
subspaces. A policy could update continuously in time but restrict exploration or
adaptation to the local high-gain subspace.

### Outcome C: smaller radius repairs consistency

EXP4 was outside a reliable local-linear regime. Continue with radius-calibrated
operators and do not interpret EXP3/EXP4 finite differences as Jacobians.

### Outcome D: state matching and subspaces both fail

Conclude that q-criticality is trajectory-specific under replayed open-loop action
suffixes. Do not proceed to a universal sparse-decision controller. A justified
method pivot would then compare action-chunk or latent-action abstractions under a
new causal estimand rather than continuing to search for global critical times.

## 13. Relation to recent literature

Literature scan date: **2026-08-14**. These papers motivate possible later action
representations; none overrides the EXP5 requirement to validate state-conditioned
causal structure first.

- Zhao et al., [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware
  (ACT)](https://arxiv.org/abs/2304.13705), introduced action chunking to reduce the
  effective decision frequency and mitigate compounding error. This is relevant if
  later experiments shift from sparse instants to temporally extended decisions.
- Haldar et al., [BAKU: An Efficient Transformer for Multi-Task Policy
  Learning](https://arxiv.org/abs/2406.07539), evaluates modular observation trunks
  and action heads, including action chunking, on LIBERO and related manipulation
  settings. It is a close benchmark reference but does not by itself establish causal
  decision sparsity.
- [Simulation-Pretrained Latent Action Space for Whole-Body Real-World Reinforcement
  Learning](https://arxiv.org/abs/2506.04147) studies a simulation-pretrained latent
  action interface for downstream RL. It supports the project's broader latent-action
  direction, but moving there before resolving local geometry would change the
  present estimand.
- [Long Horizon Latent Action Learning for General Robot
  Manipulation](https://arxiv.org/abs/2512.20166) is relevant to long-horizon latent
  action abstractions. It should be treated as a later comparison if EXP5 rejects a
  stable state-conditioned q subspace.

The research-line implication is conservative: recent latent/action-chunk methods
make a method pivot plausible, but EXP4's evidence first calls for a better causal
state and direction model, not immediate adoption of a larger learned policy.

## 14. Required outputs

EXP5 should save at minimum:

- reference and corrected-D metrics;
- state-descriptor schema and normalization manifest;
- cross-fitting and prototype manifests;
- exact match tables with distances/rejections;
- direction/radius manifests and joint-limit audits;
- zero controls;
- intervention- and per-step Parquet files;
- scalar state-alignment summaries;
- operator spectra, principal angles, and cross-radius linearity tables;
- held-out random-direction prediction results;
- LODO/LOTO and hierarchical inference;
- GPU audit and CPU/GPU equivalence;
- immutable raw hashes, failure examples, and visualizations;
- `reports/exp5_report.md` and `reports/next_exp_from5.md`.

## 15. Stop/go rule for latent or sparse RL

Proceed to sparse-decision or latent-action policy training only if EXP5 reaches
`state_conditioned_replicated_anisotropic_criticality`. If only subspaces replicate,
test subspace-constrained exploration before time sparsification. If neither scalar
nor subspace structure replicates, record the q-criticality mainline as unsupported
for a universal controller and preregister a genuinely new latent/action-chunk
estimand rather than relabeling the current result.
