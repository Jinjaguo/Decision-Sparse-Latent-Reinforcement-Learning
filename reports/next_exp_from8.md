# Next experiment from EXP8

## Recommendation

The next experiment should be:

```text
EXP9 — Action-Conditioned Hybrid Contact Response Distribution
```

EXP8 selected `continuous_geometry_insufficient`. Therefore the explicit reusable-local-operator mainline should be closed. EXP9 must not rename the failed contact-frame operator as a latent representation, and it should not repeat another cross-demo Jacobian fit with a larger network.

The new estimand should be the conditional distribution

```text
p(contact events, signed physical response, gap/wrench evolution
  | state/contact history, q perturbation, planned action chunk)
```

on the full intent-to-perturb population, including contact-mode changes. This directly targets the two failures exposed by EXP8:

1. exact-mode preservation was rare and task dependent (19.17% overall at H1; 4.44% through the remaining horizon);
2. survivor-conditioned linear operators were not reusable across demonstrations.

EXP9 remains an offline estimand-validation experiment. It must not train a scheduler, controller, MPC, VLA, or RL agent.

## Evidence driving the redesign

- Primary cross-demo top-1 was 0.39387, below Baseline B at 0.50230 and far below the per-row best frozen baseline envelope at 0.78506.
- H1 improvement over the best baseline was -0.32859, with a fully negative 95% CI [-0.44707, -0.22008].
- All task-level H1 estimates were negative.
- Held-out signed-vector relative error had median and P90 approximately 1.0.
- H3 improvement beyond Baseline B was -0.06891, 95% CI [-0.19012, 0.03635], BH q=0.84179.
- No continuous contact-frame feature showed significant incremental value. Removing the normal/tangent frame improved mean top-1 by 0.06914.
- The risk gate achieved sensitivity 0.90319 only with specificity 0.19630 and false-safe rate 0.80370.
- The remaining-horizon coverage-adjusted similarity was only 0.01902.

These results argue against estimating a single local linear map and then conditioning away contact switches. Contact switching should instead be a modeled outcome, and response uncertainty/multimodality should be part of the estimand.

## Related recent methods

These papers motivate components of EXP9 but do not establish that they will work in LIBERO:

- [Learning Contact Dynamics for Control with Action-conditioned Face Interaction Graph Networks](https://arxiv.org/abs/2509.12151) models action-conditioned contact motion and force/torque with an interaction graph. EXP9 should borrow the action-conditioned graph idea, not its control stage.
- [Contact-Aware Neural Dynamics](https://arxiv.org/abs/2601.12796) separates future contact probability from a conditional distribution over pose evolution. This is closely aligned with modeling contact switches as outputs rather than conditioning them away.
- [Koopman global linearization of contact dynamics for robot locomotion and manipulation](https://arxiv.org/abs/2511.06515) motivates a separately labeled lifted-hybrid baseline that can span contact changes. It must not be conflated with EXP8's failed explicit local operator.
- [FAWAM: Force-Aware World Action Models for Closed-Loop Contact-Rich Manipulation](https://arxiv.org/abs/2606.08555) motivates predicting future wrench/contact evolution. EXP9 should test only the offline prediction component; closed-loop correction remains forbidden.

## Core scientific questions

Primary:

```text
Can an action-conditioned hybrid distribution predict contact transitions and
multi-horizon signed physical responses across demonstrations better than
explicit-operator, nearest-state, and factorized risk/regression baselines?
```

Secondary:

```text
Does short reference history plus planned action chunk reduce false-safe mode
predictions while producing calibrated response uncertainty?
```

Mechanistic:

```text
Are failures best explained by discrete contact-event uncertainty, within-mode
nonlinear response, or interaction between the two?
```

## Cohort policy

EXP8 data may be used only for architecture development, debugging, normalization design, and power estimation. It cannot support the final EXP9 confirmation because EXP9 is proposed after observing EXP8 outcomes.

Before freezing EXP9:

1. audit all remaining unused successful trajectories;
2. target 10 new qualified demonstrations per task;
3. require same-runtime corrected-D exactness, finite snapshots, successful reference completion, and no overlap with EXP4–EXP8 confirmatory cohorts;
4. if any task has fewer than 10 candidates, stop the independent-confirmation claim and label the run as a development-only availability-limited pilot;
5. do not substitute previously analyzed demonstrations merely to reach 10.

If enough trajectories exist, use 30 new demos, 10 per task. Otherwise the report must state exact availability and may only estimate feasibility/power.

## Branch and intervention design

Retain the validated EXP8 execution substrate:

- 12 outcome-blind branches per demo;
- branch selection from reference state/history only;
- corrected-D snapshot restoration;
- three small radii 0.0003125, 0.000625, 0.00125;
- seven basis plus one held-out direction;
- both signs;
- no branch deletion based on outcomes;
- matched-zero controls before formal interventions.

Add an action-history/chunk design:

- input history: last 5 reference boundaries before the branch;
- action chunk: the next 5 frozen reference policy actions, plus the signed q perturbation;
- prediction horizons: 1, 3, 5, 10, and remaining;
- output contact events: named-pair add/drop, physical-group add/drop, and exact-mode preservation;
- output continuous sequence: signed gap, contact-frame relative velocity, normal/tangential force or torque, EEF-object relative motion, task-object state, and the existing normalized physical response vector.

The action chunk is an observed/frozen conditioning variable, not a learned macro-action policy.

## Models to freeze

### Baseline A — empirical and EXP8 baselines

- normalized-time nearest reference;
- physical-progress nearest reference;
- EXP5-style physical-state ridge;
- final EXP8 contact-frame KRR operator, evaluated exactly as frozen.

### Baseline B — factorized hybrid predictor

Two stages trained with demo-level cross-fitting:

1. contact-event classifier from current state/contact features and perturbation;
2. separate continuous response regressor conditioned on predicted event class.

This determines whether joint distribution modeling is necessary.

### Baseline C — lifted hybrid linear model

A compact Koopman-style lifted model with an explicitly frozen feature dictionary. This is a new nonlinear/lifted estimand and must never be described as evidence rescuing EXP8's explicit operator.

### Primary — action-conditioned contact graph mixture distribution

- graph nodes: robot end-effector, task object, support object, relevant environment bodies;
- edges: audited nearest physical contact pairs;
- edge history: gap, frame, relative velocity, force/torque, contact age, active flag;
- action conditioning: signed q perturbation and next five policy actions;
- temporal encoder over five reference boundaries;
- sparse mixture-of-experts or mixture-density head;
- one head predicts discrete contact events;
- one head predicts a distribution over signed continuous response/wrench trajectories;
- permutation invariance over same-group contact pairs;
- compact parameter budget frozen before confirmatory outcomes.

Use a small model first. The point is to test the hybrid distributional estimand, not to win by unconstrained capacity.

## Cross-fitting and leakage controls

- Entire demonstration is the independent unit.
- Five outer folds per task; no state, branch, direction, or horizon from a held-out demo may appear in training.
- Hyperparameters, early stopping, calibration, mixture count, and risk threshold use training folds only.
- Normalization is fit on training demos only.
- Contact-event labels from test folds are never inputs to the continuous predictor.
- Teacher-forced event labels may appear only as a separately labeled oracle diagnostic.
- The held-out q direction remains excluded from operator/response fitting.
- Final test predictions are written once and locked before hypothesis evaluation.

## Primary endpoints and gates

### H1 — hybrid response-distribution improvement

Population: all intent-to-perturb branches, not preserved-mode survivors.

Primary score: demo-averaged multivariate energy score at horizons 1, 3, and 5, normalized by zero-floor and reference response scale.

Pass only if:

- Primary improves over every frozen baseline by at least 10% relative energy score;
- demo-clustered 95% CI for improvement is entirely positive;
- all three task point estimates are positive;
- at least 2/3 task CIs have positive lower bounds.

### H2 — contact-event prediction

Report exact-mode preservation, physical-group add/drop, and named-pair add/drop separately.

Pass only if:

- macro AUROC demo-cluster CI lower bound >=0.80;
- macro AUPRC >=0.75;
- ECE <=0.05;
- sensitivity >=0.85 and specificity >=0.70 at a training-only threshold;
- overall false-safe rate <=0.30;
- every task specificity >=0.60.

### H3 — continuous response conditional on predicted events

Use predicted, not oracle, contact events in the primary population.

Pass only if:

- median signed-vector relative error <=0.35;
- P90 <=0.65;
- 90% predictive interval empirical coverage is 0.85–0.95;
- interval width improves over the factorized baseline without undercoverage.

Report oracle-event results only as a mechanism diagnostic.

### H4 — rollout locality and event compounding

For horizons 1, 3, 5, 10, remaining, report:

- event AUROC/AUPRC/ECE;
- energy score and CRPS;
- vector error median/P90/P95/max;
- contact/wrench sequence error;
- predictive interval coverage;
- task-level results;
- intent-to-perturb population size.

Do not condition the main curve on future mode preservation.

### H5 — incremental value of history and action chunk

Required ablations:

- no state history;
- no planned action chunk;
- no force/torque;
- no contact age;
- no exact pair identity;
- no physical-group identity;
- no nearest points/frame;
- deterministic single-head instead of mixture distribution;
- event-only and response-only factorization;
- shuffled action chunk as a negative control.

Require positive demo-clustered CI and BH q<0.05 for any claim of incremental value.

## Calibration and uncertainty

- Calibrate event probabilities on training folds only.
- Use proper scoring rules: log loss/Brier for discrete events and energy score/CRPS for continuous distributions.
- Report reliability curves and class-conditional ECE.
- Report empirical 50/80/90/95% interval coverage and width.
- Report false-safe rate with a “safe” definition frozen before outcomes.
- If no threshold meets sensitivity and specificity gates, retain continuous risk scores and declare the selective gate failed.

## Zero, raw-lock, and GPU policy

- Re-run matched-zero controls for the new cohort.
- Require exact corrected-D median/P95/max and terminal-pose gates before q interventions.
- Freeze all manifests, folds, model sizes, losses, metrics, seeds, and thresholds before formal outcomes.
- Partition intervention execution by immutable demo shards.
- Merge only after every shard passes exact coverage and safety gates.
- Hash and commit raw Parquet files before any formal fit.
- GPU is permitted for neural fitting only after float64/float32 equivalence or bounded mixed-precision audit on a frozen subset.
- MuJoCo remains the CPU source of truth.
- No post-hoc tolerance relaxation.

## Formal classifications

Use this priority order:

1. `hybrid_response_distribution_replicates`
   - H1, H2, and H3 pass.
2. `contact_event_model_only`
   - H2 passes; H1 or H3 fails.
3. `continuous_distribution_only`
   - H1/H3 pass but H2 fails; no risk/scheduler claim.
4. `lifted_hybrid_baseline_only`
   - lifted baseline passes while Primary does not; report this without relabeling EXP8.
5. `action_conditioning_insufficient`
   - cohort/support valid but H1–H3 fail.
6. `support_or_identifiability_failure`
   - independent cohort, zero, event prevalence, or output identifiability is inadequate.

Only classification 1 may authorize a later, separate offline scheduler-utility experiment. None authorizes online control or latent RL directly.

## Required artifacts

At minimum:

```text
exp9_cohort_manifest.json
branch_manifest.json
action_chunk_manifest.json
contact_event_schema.json
response_distribution_schema.json
crossfit_manifest.json
model_specs.json
statistical_analysis_plan.json
zero_controls.parquet
interventions.parquet
per_step_effects.parquet
raw_hash_manifest.json
contact_event_predictions.parquet
response_distribution_predictions.parquet
heldout_direction_predictions.parquet
horizon_rollouts.parquet
ablation_results.parquet
calibration_metrics.json
scientific_decision.json
failure_examples.json
```

## Highest-value first step

Do not immediately run another 17,280-intervention sweep. First run a strictly labeled retrospective feasibility analysis on locked EXP8 raw data:

1. define contact-event and distributional targets without modifying EXP8;
2. compare factorized, lifted, and small graph-mixture models with the existing demo folds;
3. test whether contact-event prevalence and sample size support the proposed gates;
4. estimate training time and select a compact architecture;
5. use those results only for EXP9 design/power, never as confirmation.

If this retrospective stage cannot beat Baseline B on held-out EXP8 demos or cannot improve risk specificity above 0.50 while retaining sensitivity 0.85, stop before collecting a new cohort. That is the cheapest decisive test of whether the new estimand is viable.
