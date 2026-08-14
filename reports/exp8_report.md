# EXP8 Report — Continuous Contact-Frame Response Field

Date: 2026-08-14

Status: completed

Formal classification: `continuous_geometry_insufficient`
Final analysis run: `exp8_s13_formal_analysis_r5_20260814`

## Executive conclusion

EXP8 completed the full independent-cohort, zero-control, 17,280-intervention, cross-demo analysis protocol. The simulator, corrected-D restoration, contact-frame measurements, GPU/CPU audit, raw-data lock, artifact completeness, and all execution gates passed.

The scientific hypothesis did not pass. The frozen continuous contact-frame model generalized worse than the frozen baselines, failed held-out signed-vector prediction, and produced an inadequate selective mode-risk gate. The formal priority rule therefore selects:

```text
continuous_geometry_insufficient
```

The strongest allowed claim is narrow but useful: continuous nearest-surface geometry, deterministic contact frames, relative velocity, contact age, force/torque, and action projection can be measured exactly and reproducibly in the tested MuJoCo runtime, but the tested permutation-invariant contact-frame KRR model does not define a reusable cross-demonstration local response operator. The explicit reusable-local-operator mainline should be closed. Offline scheduler evaluation, online control, and latent RL are not eligible.

## Required-question answers

| Question | Answer |
|---|---|
| Was an independent 30-demo cohort obtained? | Yes. 30/30 first selected candidates passed: 10 Drawer, 10 Bowl, 10 Stove; no rejected reference. |
| Did corrected-D remain exact? | Yes. Zero-control integration median, P95, maximum, and terminal-object-pose P95 were all exactly 0 over 360 branches and 11,766 suffix steps. |
| Were nearest points/contact frames reproducible? | Yes. 120 repeated boundaries had maximum feature range exactly 0. |
| Was tangent gauge deterministic? | Yes. The deterministic right-handed normal/tangent gauge passed audit and known-answer tests. |
| Were force/impulse fields valid? | Yes as force/torque, not impulse. MuJoCo 3.2.3 `mj_contactForce` semantics were audited; 1,758 reference boundaries had positive normal force. |
| How many branches/interventions completed? | 360 branches; exactly 17,280 unique signed interventions; 564,768 per-step rows; 246 terminal-success flips. |
| How much H1 preserved-mode support existed? | 69/360 branches (19.17%) and 23/30 demos. Drawer 45/120, Bowl 9/120, Stove 15/120 branches. |
| Did contact-frame conditioning improve cross-demo top-1 reuse? | No. Demo-level mean improvement over the best baseline was -0.32859, 95% CI [-0.44707, -0.22008]. |
| Did it beat every frozen baseline? | No. Primary mean top-1 was 0.39387; Baseline B was 0.50230; per-row best frozen baseline was 0.78506. |
| Did at least 2/3 tasks independently improve? | No. All three task estimates and all three task CIs were negative. |
| Did held-out vector prediction pass median and p90 gates? | No. Median relative error was 1.00000 and P90 was 1.00000; required <=0.35 and <=0.60. Demo-median rho was 0. |
| Which contact-frame features added incremental value? | None had statistically established incremental value. Removing most fields improved mean top-1; all formal removal variants remained non-significant versus Baseline B. |
| Did action-direction projection matter? | Only negligibly: removing it changed mean top-1 by -0.00102 relative to full; no significant incremental effect was established. |
| Did contact age matter? | Not positively: removing it raised mean top-1 by 0.01109. |
| Did force/impulse matter? | Not positively: removing force raised mean top-1 by 0.01051. Impulse was never claimed or used. |
| How did H1/H3/H5/remaining coverage-adjusted performance change? | 0.07549, 0.06529, 0.07023, 0.01902. It remained low and collapsed over the remaining horizon. |
| Did risk gate reach specificity >=0.70 and sensitivity >=0.85? | No. Sensitivity was 0.90319, but specificity was only 0.19630. |
| What was false-safe rate? | 0.80370 overall; Drawer 0.89593, Bowl 0.70929, Stove 0.86190. |
| Which formal classification was selected? | `continuous_geometry_insufficient`. |
| What is the strongest allowed claim? | Contact-frame measurements are reproducible, but the frozen explicit model does not support reusable cross-demo operators or a deployable risk gate. |
| Is offline scheduler utility evaluation eligible? | No. |
| Is online control eligible? | No. |
| Is latent RL eligible? | No. |
| What should EXP9 test? | An offline action-conditioned hybrid forward distribution over contact transitions, signed physical response, and wrench/contact evolution—not another explicit local Jacobian and not a controller. |

## Protocol and frozen design

### Independent cohort

- Drawer: `demo_31`–`demo_40`.
- Bowl: `demo_30`–`demo_39`.
- Stove: `demo_30`–`demo_39`.
- These episodes were unused by the previous confirmatory cohorts.
- All 30 first candidates completed successfully, remained finite, and passed corrected-D qualification.
- Merged run: `exp8_s2_independent_refs_20260814`.

### Branch and perturbation design

- 12 reference-only, outcome-blind branches per demonstration.
- 360 total branches, stratified by time, physical progress, contact mode, signed gap, gripper state, predicate state, and pair support.
- Radii: 0.0003125, 0.000625, and 0.00125 of joint range.
- Seven frozen orthonormal basis directions plus one independently frozen random held-out direction.
- Both signs at every branch/radius/direction.
- 8,640 unsigned direction/radius rows and 17,280 signed interventions.
- No joint-limit branch replacement and no branch deletion.
- Horizons: 1, 3, 5, and remaining continuation.

### Frozen models

- Baseline A: exact named contact mode plus boundary-margin class, with reference normalized-time candidate selection; separate normalized-time and physical-progress comparators were also retained.
- Baseline B: continuous EXP5-style physical-state features plus gap/normal-velocity matching through ridge regression.
- Primary: permutation-invariant pooled contact-pair features with RBF kernel ridge regression.
- Five outer folds split by entire demonstration. Hyperparameters were selected only in nested training folds.
- H1/H2 confirmatory support required all seven basis directions and both signs to preserve the exact mode.
- Risk thresholds were selected only from training-fold out-of-fold predictions, maximizing specificity subject to sensitivity >=0.85.

## Execution chronology

### 1. Baseline and cohort audit

- Baseline: 59/59 tests passed; `pip check` clean.
- Availability run `exp8_s0_cohort_audit_20260814`: 19 unused Drawer, 20 unused Bowl, and 20 unused Stove candidates were available.
- Reference generation:
  - `exp8_s1_refs_drawer_31_40_20260814`
  - `exp8_s1_refs_bowl_30_39_20260814`
  - `exp8_s1_refs_stove_30_39_20260814`
- Merge: `exp8_s2_independent_refs_20260814`, 30/30 qualified, 0 rejected.

### 2. Contact identity and contact-frame audit

- `exp8_s3_contact_identity_audit_20260814` observed 12 relevant Drawer pairs, 57 Bowl pairs, and 21 Stove pairs.
- Preserved negative run `exp8_s4_contact_frame_audit_20260814` stopped before q outcomes because a fixed 0.05 m `mj_geomDistance` query returned no nearest segment for far free-space Stove pairs.
- The prospective repair used a 0.05/0.2/1/5 m query ladder and never substituted body-center geometry.
- Corrected run `exp8_s4_contact_frame_audit_r1_20260814` passed:
  - 3,232 reference boundaries;
  - 120 repeated boundaries;
  - repeat maximum feature range 0;
  - active-pair range 0–16;
  - valid-pair range 12–57;
  - 1,758 positive-normal-force boundaries.
- Official MuJoCo 3.2.3 semantics were recorded for `mj_geomDistance`, `mj_jac`, `mjContact.frame`, and `mj_contactForce`.

### 3. Feature-support audit and freeze

- `exp8_s5_branch_selection_20260814`: 360 frozen branches.
- The first support run, `exp8_s6_frozen_contact_features_20260814`, was intentionally retained as a failure because “free space” was incorrectly tested as a wholly contact-free branch.
- The corrected preregistered estimand is pair-level valid inactive nearest-surface support. `exp8_s6_frozen_contact_features_r1_20260814` passed:

| Task | Active branches | Wholly free branches | Valid free-space pairs | Exact modes | Physical groups | Max contact age |
|---|---:|---:|---:|---:|---:|---:|
| Drawer | 91/120 | 29 | 1,304 | 28 | 2 | 43 |
| Bowl | 120/120 | 0 | 6,161 | 106 | 4 | 48 |
| Stove | 110/120 | 10 | 2,262 | 61 | 1 | 30 |

Bowl therefore supports inactive nearest pairs, not a claim of wholly contact-free branch coverage.

### 4. GPU equivalence

- Final run: `exp8_s7_gpu_equivalence_r1_20260814`.
- Device: NVIDIA GeForce RTX 4090; CUDA 11.3; PyTorch 1.11.0+cu113; float64.
- 11/11 components passed with no fallback and no tolerance relaxation.
- Largest absolute error: 2.13e-14 for the Gram matrix; KRR prediction error 2.66e-15; top-1 projector error 8.33e-16.
- MuJoCo physics itself remained CPU; GPU was used only for the audited numerical analysis path.

### 5. Dry and zero controls

- Dry run `exp8_s8_dry_20260814`: one branch, two interventions, 238 per-step rows, maximum non-arm mutation 0.
- Full zero run `exp8_s9_zero_controls_20260814`:
  - 360/360 branches;
  - 11,766 reference suffix steps;
  - 243.77 s;
  - integration median/P95/max exactly 0;
  - terminal object pose P95 exactly 0;
  - no failures.

### 6. Formal intervention sweep

- Ten immutable three-demo shards: `exp8_s10_formal_shard_00_20260814` through `exp8_s10_formal_shard_09_20260814`.
- Every shard finished with `status=completed` and `gate.passed=true`.
- Each shard contained 1,728 interventions; union 17,280.
- Aggregate per-step rows: 564,768.
- Aggregate success flips: 246.
- Aggregate shard compute time: 8,457.99 s (2.35 compute-hours); shards ran concurrently.
- Maximum non-arm mutation: 0.
- No branch or formal row was removed.

### 7. Raw lock and assembly

Raw run `exp8_s11_formal_raw_locked_r1_20260814` passed exact coverage, uniqueness, geometry schema, and failure gates. Maximum q-injection error was 2.22e-16.

| Raw artifact | SHA-256 |
|---|---|
| `interventions.parquet` | `31cd1ff86a9196c83d0c7312e4c1013fa80671db28ebf793e091a376d6540c9a` |
| `per_step_effects.parquet` | `58c293bd7cf6f389f2e221b1ff28efdd6c269a13148c0f66645877b1d58dd8c3` |
| `zero_controls.parquet` | `1c0353e49323e18f11114345e83e4ae15038fbeb5533fa87828406bd13dbaaff` |
| `zero_reference_steps.parquet` | `a99212b5dd4d52a27dc67e1e39b33b3b2329fd37d926a2722a9027a01fa29279` |

The hashes were committed as `a56ea38` before formal analysis. Assembly run `exp8_s12_horizon_assembly_20260814` reverified them and produced 69,120 horizon responses, 34,560 mode outcomes, 4,320 operators, and 4,320 held-out rows.

### 8. Analysis implementation failures retained

No failed analysis run wrote a final scientific decision. The following runs are preserved for audit:

- `exp8_s13_formal_analysis_20260814`: Arrow/Pandas nested arrays reached NumPy as `object`; fixed by explicit float64 materialization.
- `exp8_s13_formal_analysis_r1_20260814`: nested operator matrices required row-wise decoding.
- `exp8_s13_formal_analysis_r2_20260814`: H4 intent tried to reuse a missing conditional-fold parameter key; fixed to separately train intent and conditional models at each horizon, matching the frozen H4 definition.
- `exp8_s13_formal_analysis_r3_20260814`: a collinear conditional fold made normal-equation ridge singular; replaced with the mathematically equivalent augmented least-squares/SVD solve.
- `exp8_s13_formal_analysis_r4_20260814`: `direction_role` appeared in both risk merge inputs and was suffixed; it was added to the exact merge key. The join was verified 17,280 -> 17,280 with both roles intact.
- `exp8_s13_formal_analysis_r5_20260814`: completed and passed.

These repairs changed no cohort, raw datum, fold, feature definition, model family, hyperparameter grid, threshold, or scientific gate.

## Formal results

### H1 — cross-demo subspace reuse: failed

- Supported demos: 23/30; supported branches: 69/360.
- Demo-level mean improvement over best frozen baseline: -0.32859.
- Demo-clustered 95% CI: [-0.44707, -0.22008].
- Required: mean >=+0.15, CI lower >0, every task positive, at least 2/3 task CIs lower >0.

| Task | Demo-level mean improvement | 95% CI | Supported branches | Supported demos |
|---|---:|---:|---:|---:|
| Drawer | -0.37630 | [-0.49299, -0.25914] | 45 | 10 |
| Bowl | -0.19159 | [-0.38925, -0.02107] | 9 | 6 |
| Stove | -0.37786 | [-0.66166, -0.09791] | 15 | 7 |

Mean row-level top-1 similarities:

| Model/baseline | Mean top-1 |
|---|---:|
| Primary contact-frame model | 0.39387 |
| Baseline B | 0.50230 |
| Exact mode + margin | 0.43996 |
| Normalized time | 0.53799 |
| Physical progress | 0.43393 |
| Per-row best frozen baseline | 0.78506 |

The per-row best baseline is intentionally an evaluation envelope, not a deployable model. Even against Baseline B alone, the primary did not improve.

### H2 — held-out signed-vector prediction: failed

- Supported demos: 23/30.
- Held-out rows: 68.
- Demo-median response-norm Spearman rho: 0.0; required >=0.70.
- Demo-median vector relative error: 1.00000; required <=0.35.
- Overall vector error median: 1.00000.
- P90: 1.00000; required <=0.60.
- P95: 1.00002.
- Maximum: 1.74736.

| Task | Demo-median rho | Demo-median error | P90 | P95 | Max |
|---|---:|---:|---:|---:|---:|
| Drawer | -0.300 | 1.00000 | 1.00000 | 1.00003 | 1.74736 |
| Bowl | -0.500 | 0.92989 | 1.00000 | 1.00000 | 1.00000 |
| Stove | 0.000 | 1.00000 | 1.00000 | 1.06450 | 1.21499 |

Constant-input Spearman warnings occurred for demos with constant predicted norms. They were retained as evidence of non-informative prediction, not filtered.

### H3 — incremental value beyond Baseline B: failed

- Demo-level mean improvement: -0.06891.
- 95% CI: [-0.19012, 0.03635].
- One-sided permutation p: 0.84179.
- BH q: 0.84179.
- Required: CI lower >0 and BH q <0.05.

### Ablations

`delta_from_full` is ablated mean top-1 minus full-model mean top-1; positive values mean removing the field improved performance.

| Ablation | Mean top-1 | Delta from full | Mean improvement vs B | Permutation p |
|---|---:|---:|---:|---:|
| Remove nearest points | 0.39679 | +0.00292 | -0.10551 | 0.86403 |
| Remove normal/tangent frame | 0.46300 | +0.06914 | -0.03929 | 0.80230 |
| Remove relative velocity | 0.39510 | +0.00123 | -0.10720 | 0.84779 |
| Remove contact age | 0.40495 | +0.01109 | -0.09734 | 0.81830 |
| Remove force | 0.40438 | +0.01051 | -0.09792 | 0.83679 |
| Remove action projection | 0.39285 | -0.00102 | -0.10945 | 0.86703 |
| Remove EEF-object pose | 0.40980 | +0.01593 | -0.09250 | 0.73307 |

No removal variant had a positive CI lower bound or significant permutation p. Action projection was the only listed field whose removal slightly reduced mean top-1, but the magnitude was only 0.00102 and did not establish incremental value.

### H4 — horizon locality

| Horizon | Intent top-1 | Mode preservation | Conditional top-1 | Coverage-adjusted similarity |
|---|---:|---:|---:|---:|
| 1 | 0.45863 | 0.19167 | 0.39387 | 0.07549 |
| 3 | 0.40636 | 0.15278 | 0.42738 | 0.06529 |
| 5 | 0.37661 | 0.14444 | 0.48618 | 0.07023 |
| Remaining | 0.30024 | 0.04444 | 0.42793 | 0.01902 |

Task-specific full-direction mode preservation at the smallest radius:

| Task | H1 | H3 | H5 | Remaining |
|---|---:|---:|---:|---:|
| Drawer | 0.3750 | 0.3333 | 0.3250 | 0.1250 |
| Bowl | 0.0750 | 0.0333 | 0.0333 | 0.0083 |
| Stove | 0.1250 | 0.0917 | 0.0750 | 0.0000 |

The conditional survivor score does not rise with deployable support. The mandatory intent and coverage-adjusted views show poor and rapidly shrinking utility.

### H5 — selective mode-risk gate: failed

- AUROC: 0.64131.
- Demo-clustered 95% AUROC CI: [0.59424, 0.68355]; required lower bound >=0.75.
- AUPRC: 0.63532.
- ECE: 0.12076; required <=0.05.
- Sensitivity: 0.90319; required >=0.85, passed individually.
- Specificity: 0.19630; required >=0.70, failed.
- False-safe rate: 0.80370.
- PPV: 0.50619; NPV: 0.68974.

| Task | AUROC | Sensitivity | Specificity | False-safe rate |
|---|---:|---:|---:|---:|
| Drawer | 0.60919 | 0.91498 | 0.10407 | 0.89593 |
| Bowl | 0.65240 | 0.87351 | 0.29071 | 0.70929 |
| Stove | 0.54816 | 0.90744 | 0.13810 | 0.86190 |

High sensitivity was obtained by classifying too many unsafe branches as safe. This gate cannot be used for scheduling or control.

## Output audit and deliverables

Independent audit of `exp8_s13_formal_analysis_r5_20260814` passed:

- 20/20 required artifacts present;
- 16/16 required plots present;
- raw hashes valid;
- exactly 360 zero controls;
- exactly 17,280 interventions;
- classification valid;
- online control and latent RL explicitly forbidden.

Final artifact directory:

```text
runs/exp8_s13_formal_analysis_r5_20260814/artifacts
```

Final plot directory:

```text
runs/exp8_s13_formal_analysis_r5_20260814/plots
```

The required artifacts include contact frames, support audits, zero controls, locked raw interventions, per-step effects, horizon operators, operator matrices, all baseline/primary/held-out predictions, ablations, horizon locality, risk predictions/metrics, GPU audits, the decision file, failure examples, and the raw-hash manifest. `failure_examples.json` is empty.

## Exact principal commands

The conda interpreter was:

```powershell
C:\Users\Guoji\anaconda3\envs\libero-exp1\python.exe
```

Principal commands, abbreviated only where a regular shard index expands 0–9:

```powershell
python scripts\exp8\audit_independent_cohort.py --output runs\exp8_s0_cohort_audit_20260814\artifacts\cohort_availability.json

python scripts\exp8\generate_new_references.py --run-id exp8_s1_refs_drawer_31_40_20260814 --progress-audit-run runs\exp4_e4_0_1_runtime_audit_geom_20260814 --episodes 31 32 33 34 35 36 37 38 39 40 --task-names open_the_middle_drawer_of_the_cabinet
python scripts\exp8\generate_new_references.py --run-id exp8_s1_refs_bowl_30_39_20260814 --progress-audit-run runs\exp4_e4_0_1_runtime_audit_geom_20260814 --episodes 30 31 32 33 34 35 36 37 38 39 --task-names put_the_bowl_on_the_plate
python scripts\exp8\generate_new_references.py --run-id exp8_s1_refs_stove_30_39_20260814 --progress-audit-run runs\exp4_e4_0_1_runtime_audit_geom_20260814 --episodes 30 31 32 33 34 35 36 37 38 39 --task-names turn_on_the_stove
python scripts\exp8\merge_new_references.py --run-id exp8_s2_independent_refs_20260814 --source-runs exp8_s1_refs_drawer_31_40_20260814 exp8_s1_refs_bowl_30_39_20260814 exp8_s1_refs_stove_30_39_20260814

python scripts\exp8\audit_contact_frame.py --reference-run runs\exp8_s2_independent_refs_20260814 --identity-schema runs\exp8_s3_contact_identity_audit_20260814\artifacts\contact_identity_schema.json --run-dir runs\exp8_s4_contact_frame_audit_r1_20260814
python scripts\exp8\build_branch_manifest.py --contact-frames runs\exp8_s4_contact_frame_audit_r1_20260814\artifacts\reference_contact_frames.parquet --output runs\exp8_s5_branch_selection_20260814\artifacts\branch_candidates.json
python scripts\exp8\freeze_protocol.py --reference-run runs\exp8_s2_independent_refs_20260814 --branch-input runs\exp8_s5_branch_selection_20260814\artifacts\branch_candidates.json --contact-frames runs\exp8_s4_contact_frame_audit_r1_20260814\artifacts\reference_contact_frames.parquet --contact-audit runs\exp8_s4_contact_frame_audit_r1_20260814\artifacts\contact_frame_audit.json --identity-schema runs\exp8_s3_contact_identity_audit_20260814\artifacts\contact_identity_schema.json --output experiments\exp8_continuous_contact_frame
python scripts\exp8\build_contact_frame_features.py --contact-frames runs\exp8_s4_contact_frame_audit_r1_20260814\artifacts\reference_contact_frames.parquet --manifest-dir experiments\exp8_continuous_contact_frame\manifests --run-dir runs\exp8_s6_frozen_contact_features_r1_20260814
python scripts\exp8\validate_gpu_backend.py --run-dir runs\exp8_s7_gpu_equivalence_r1_20260814

python scripts\exp8\run_interventions.py --run-id exp8_s8_dry_20260814 --mode dry --reference-run runs\exp8_s2_independent_refs_20260814 --manifest-dir experiments\exp8_continuous_contact_frame\manifests --config experiments\exp8_continuous_contact_frame\configs\exp8.json --max-trajectories 1 --max-branches 1 --max-directions 1
python scripts\exp8\run_zero_controls.py --run-id exp8_s9_zero_controls_20260814 --mode zero --reference-run runs\exp8_s2_independent_refs_20260814 --manifest-dir experiments\exp8_continuous_contact_frame\manifests --config experiments\exp8_continuous_contact_frame\configs\exp8.json

python scripts\exp8\run_interventions.py --run-id exp8_s10_formal_shard_00_20260814 --mode full --reference-run runs\exp8_s2_independent_refs_20260814 --zero-run runs\exp8_s9_zero_controls_20260814 --manifest-dir experiments\exp8_continuous_contact_frame\manifests --config experiments\exp8_continuous_contact_frame\configs\exp8.json --trajectory-start 0 --max-trajectories 3
# Repeated for shard 01..09 with trajectory-start 3,6,...,27.

python scripts\exp8\merge_intervention_shards.py --run-id exp8_s11_formal_raw_locked_r1_20260814 --source-runs exp8_s10_formal_shard_00_20260814 ... exp8_s10_formal_shard_09_20260814 --zero-run runs\exp8_s9_zero_controls_20260814 --manifest-dir experiments\exp8_continuous_contact_frame\manifests
python scripts\exp8\assemble_horizon_operators.py --raw-run runs\exp8_s11_formal_raw_locked_r1_20260814 --manifest-dir experiments\exp8_continuous_contact_frame\manifests --output-run runs\exp8_s12_horizon_assembly_20260814
python scripts\exp8\analyze_formal.py --raw-run runs\exp8_s11_formal_raw_locked_r1_20260814 --assembly-run runs\exp8_s12_horizon_assembly_20260814 --feature-run runs\exp8_s6_frozen_contact_features_r1_20260814 --contact-frame-run runs\exp8_s4_contact_frame_audit_r1_20260814 --gpu-run runs\exp8_s7_gpu_equivalence_r1_20260814 --manifest-dir experiments\exp8_continuous_contact_frame\manifests --output-run runs\exp8_s13_formal_analysis_r5_20260814
python scripts\exp8\audit_outputs.py --run runs\exp8_s13_formal_analysis_r5_20260814
```

## Tests and environment health

- Baseline before EXP8: 59 passed.
- Targeted final EXP8 tests before repository-wide run: 17 passed.
- Added/retained known-answer coverage includes nearest surface points, signed gap, normal orientation, tangent gauge, action projection, relative point velocity, contact-force transformation, contact age, permutation invariance, physical identity, demo isolation, ridge determinism/stability, projector similarity, p90/P95, risk threshold, specificity/sensitivity, false-safe rate, ECE, nested Parquet arrays, and immutable run directories.
- GPU/CPU equivalence: 11/11 runtime cases passed.
- Final repository-wide run: 76/76 tests passed in 2.05 s.
- `pip check`: `No broken requirements found.`
- `compileall` for `scripts/exp8` and `src/decision_sparse_rl/metrics/exp8.py`: passed.
- Final output audit rerun: passed all criteria with 20/20 artifacts and 16/16 plots.

## Commits created during EXP8

```text
82e428b exp8: audit independent contact-frame cohort
3845347 exp8: freeze continuous contact-frame protocol
cc5deac exp8: freeze formal analysis pipeline
206b9e4 exp8: require full demo support for confirmatory gates
6221431 exp8: finalize frozen ablation and horizon estimands
c23b8a7 exp8: align formal robust feature normalization
a56ea38 exp8: lock formal raw hashes
2b9b528 exp8: materialize formal arrays as float64
9e01f0f exp8: decode nested parquet matrices
896a607 exp8: train horizon estimands independently
899a68a exp8: stabilize formal ridge solves
805065e exp8: preserve direction role in risk join
```

## Claim boundary and decision

Allowed:

- The independent 30-demo cohort and corrected-D substrate passed.
- Contact-frame measurements and GPU/CPU numerical results were reproducible.
- The frozen explicit contact-frame model failed cross-demo operator reuse and held-out vector prediction.
- The frozen risk model did not meet discrimination, calibration, or specificity requirements.
- The formal classification is `continuous_geometry_insufficient`.

Forbidden:

- Calling the failed explicit operator a latent representation.
- Claiming a reusable Jacobian/contact response field.
- Claiming a deployable risk gate.
- Starting an offline scheduler utility study from EXP8's failed gates.
- Online control, MPC, VLA integration, or latent RL.

The highest-value next action is the offline estimand-validation experiment specified in `reports/next_exp_from8.md`.
