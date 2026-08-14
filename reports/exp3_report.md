# EXP3 Report: Time-Indexed q Criticality in LIBERO

**Experiment:** EXP3 — Time-Indexed q Criticality  
**Execution date:** 2026-08-14  
**Formal classification:** **Partial support**  
**Primary formal run:** `runs/exp3_t6_full_20260814T022200`  
**Passing matched-zero run:** `runs/exp3_t5_zero_corrected_20260814T021800`

## 1. Executive conclusion

EXP3 completed the full preregistered causal sweep: 3 tasks, 3 demonstrations per
task, 12 frozen branch times per demonstration, 4 frozen PCG64 directions per
branch, and both perturbation signs. All 864 interventions and 45,656 continuation
steps passed the execution gates. The intervention changed only Panda arm q at the
branch boundary; the largest change in any non-arm MuJoCo integration component was
exactly zero. All joint-limit and finite-state checks passed, and no branch was
removed.

The primary result is evidence for temporal non-uniformity, but not for a stable,
replicated set of sparse decision times:

- the median demonstration-level top-20% effect mass was **0.4493**;
- its hierarchical bootstrap 95% CI was **[0.3579, 0.5861]**;
- the uniform 12-point null is **0.25**;
- however, within-task cross-demo median Spearman correlations were only
  **0.2727** (drawer), **0.4061** (bowl), and **0.1879** (stove);
- only **2 of 9** demonstrations passed the frozen direction/sign stability rule;
- the weakest leave-one-demo-out top-20 estimate was **0.4381**, below the strong
  support requirement of 0.45.

The frozen decision rule therefore returns **partial support**, not strong support.
The permitted claim is that small local q perturbations have temporally non-uniform
remaining-horizon effects in this pilot. The experiment does **not** establish a
task-general sparse set of decision times, an event-driven control mechanism, or a
benefit for sparse-decision RL.

## 2. Scientific question

EXP3 asked whether the causal consequence of the same small arm-configuration
perturbation is concentrated at a small subset of times along successful LIBERO
demonstrations.

The intervention was fixed before formal outcomes:

```text
delta_q[j] = sign * 0.005 * verified_joint_range[j] * unit_direction[j]
```

For every branch, four independent seven-dimensional directions were generated,
L2-normalized, and evaluated with signs -1 and +1. The experimental unit for
inference was the demonstration, not a simulator step.

EXP3 did not train a latent policy, event detector, reinforcement-learning agent,
SVM, or robot controller. It measured causal time-indexed sensitivity only.

## 3. Frozen scope

### 3.1 Tasks and demonstrations

| Task | Task ID | Demonstrations | Branches per demo | Interventions |
|---|---:|---:|---:|---:|
| `open_the_middle_drawer_of_the_cabinet` | 0 | 0, 1, 2 | 12 | 288 |
| `turn_on_the_stove` | 7 | 0, 1, 2 | 12 | 288 |
| `put_the_bowl_on_the_plate` | 8 | 0, 1, 2 | 12 | 288 |
| **Total** | — | **9** | **108** | **864** |

Every demonstration was a successful local same-runtime reference from
`exp2_r2_gripper_refs_20260814T011336`. Branch selection was inherited verbatim from
the outcome-blind EXP2 manifest: ten temporal quantiles plus one gripper-event slot
and one contact-count-change slot. Invalid drawer gripper events remained explicit
50% fallbacks and were never treated as real events.

### 3.2 Audited task semantics

The task-grounded channels were derived from the BDDL files and exact LIBERO
predicate implementation:

| Task | BDDL goal | Audited physical target bodies |
|---|---|---|
| Drawer | `Open(wooden_cabinet_1_middle_region)` | `wooden_cabinet_1_cabinet_middle` |
| Stove | `TurnOn(flat_stove_1)` | `flat_stove_1_button` |
| Bowl | `On(akita_black_bowl_1, plate_1)` | `akita_black_bowl_1_main`, `plate_1_main` |

The binary predicate channel was always `env.check_success()`, which evaluates the
parsed BDDL conjunction. Object identifiers were checked against the active MuJoCo
model rather than inferred only from task names.

## 4. Preregistration and manifest freeze

All required scientific decisions were frozen before the formal q sweep in commit
`ff8e6bf`.

The manifest set is:

1. `task_demo_manifest.json`;
2. `branch_manifest.json`;
3. `direction_manifest.json`;
4. `effect_channel_schema.json`;
5. `effect_normalization.json`;
6. `primary_metric_spec.json`;
7. `event_manifest.json`;
8. `statistical_analysis_plan.json`;
9. `scientific_decision_rule.json`.

The direction master seed was `4223443001079176503`. It was obtained from a SHA-256
derivation involving the scaffold commit and frozen branch-manifest hash. It is not
the EXP2 R5 seed (`20260814`). Each direction stores its seed spawn key, raw vector,
unit vector, unsigned delta, and two-sign joint-limit audit. A deterministic test
regenerates all 432 direction vectors exactly.

Key frozen manifest hashes:

| Manifest | SHA-256 |
|---|---|
| Branch | `686761a062f211fd0223c06550b03ad5ff74cc3e282f935b05f92794fd1d4502` |
| Direction | `570c4c127e159a9a789187f0fe284bdcb2cc42cbb7366613837c96b3e5bf390f` |
| Channel schema | `e7ed017c012f6f3d5418a1597bc48bda692f3be025c40e94ec1ee8550283c8c8` |
| Normalization | `906ca20e409a2878e1bbe0b39a6d46c00009d35b7d7a2e6b64a36ea222cdaa12` |
| Primary metric | `05df556491bd6cd3d626189ebfbca57cbb6526f2cb7780720e066923bca2ef3d` |
| Events | `6fb4d87d2e270eb8033b4ec553e85e6e36b98d943038ef6cecef6a25c9da565c` |
| Statistical plan | `22375c893db5e1d1d51084dcc628e1046dd52683b98e5069f2eca01b8ca4955d` |
| Decision rule | `4ea56a7ad1b5860bcd4c09690f59ca48b0f81ecfd9de4da3ff18bcf59c8cd8d0` |

## 5. Primary metric

Six physical channels remained separate in storage and were normalized using only
reference data or fixed physical maxima:

1. Panda arm-q L2;
2. Panda arm-qvel L2;
3. EEF position L2;
4. EEF SO(3) geodesic angle;
5. task-object position L2;
6. mean task-object quaternion geodesic angle.

At every future step, the primary effect was the equal-weight arithmetic mean of
these six normalized components. The intervention-level endpoint was the arithmetic
mean over all remaining steps. This duration-normalized mean controls mechanically
for the shorter horizon of late branches. The branch-level primary value was the
median across its eight direction/sign interventions.

Contact-pair difference, raw contact-count difference, task-predicate divergence,
terminal success flip, and integration-state L2 were retained as separate secondary
or diagnostic channels. They were not hidden inside the primary score.

The frozen meaningful-effect threshold was 0.01, interpreted as one percent of the
equal-weight normalized physical scale.

## 6. Execution chronology and gates

### T0: substrate audit

Run: `exp3_t0_audit_20260814T020528` — **PASS**.

- all required EXP2 inputs existed;
- the nine-reference gate passed;
- the corrected-D EXP2 gate passed;
- all nine trajectories had twelve unique branches;
- the RTX 4090 and driver 596.49 were recorded;
- baseline tests: 30 passed at that stage;
- dependency check: no broken requirements.

MuJoCo remained on the validated CPU path. GPU permission did not justify changing
the deterministic simulator backend.

### T1: corrected Condition-D regression

Passing run: `exp3_t1_condition_d_20260814T020613` — **PASS**.

This was stronger than a representative subset: it reran all 108 branches, three
matched pairs per branch.

| Quantity | Result |
|---|---:|
| Matched pairs | 324 |
| Continuation steps | 17,121 |
| Atomic-component rows | 222,573 |
| Maximum integration L2 | 0 |
| Maximum q/qvel error | 0 / 0 |
| Maximum controller field error | 0 |
| Maximum EEF-position error | 0 |
| Maximum terminal object-pose L2 | 0 |
| Final-success agreement | 324/324 |

Every task, early/middle/late phase, and contact/non-contact stratum passed.

The first T1 attempt, `exp3_t1_condition_d_20260814T020559`, was preserved as a
failed run. It did not execute the simulator because the sandbox denied robosuite's
fixed `C:\tmp\robosuite.log`. The rerun used approved filesystem access; there was no
scientific or code change between the attempts.

### T2–T4: direction, channel, normalization, metric, event, and analysis freeze

All 432 directions and 864 signed interventions were generated and checked against
joint limits before outcomes. The nine required manifests and their hashes were
committed. Known-vector, all-zero, orientation, aggregation, and direction-regeneration
tests passed.

### Non-scientific dry run

Run: `exp3_tdry_20260814T021501` — **PASS**.

One branch, one direction, both signs, and two zero continuations were used only to
exercise schemas and code paths. It produced 262 effect rows, no failure, and zero
non-arm mutation. These outcomes were excluded from formal analysis.

### T5: matched-zero gate

Passing run: `exp3_t5_zero_corrected_20260814T021800` — **PASS**.

| Gate item | Result |
|---|---:|
| Frozen branches | 108/108 |
| Zero continuations per branch | 2 |
| Saved reference steps | 5,707 |
| Maximum integration L2 | 0 |
| Maximum q and qvel L2 | 0 |
| Maximum EEF position/orientation difference | 0 / 0 |
| Maximum object position/orientation difference | 0 / 0 |
| Contact-pair/count difference | 0 / 0 |
| Predicate divergence | 0 |
| Terminal-success disagreement | 0 |

The first complete zero run, `exp3_t5_zero_20260814T021530`, was correctly stopped.
The underlying twins were bitwise equal in simulator state, but the original SO(3)
and quaternion helpers returned angles up to `1.13484e-7` and `2.98023e-8` radians
for identical, slightly non-orthonormal floating-point representations. An exact-array
identity short circuit fixed this numerical metric defect. Regression tests cover the
case, the failed run remains immutable, and no formal q outcome had been started.

### T6: full intervention sweep

Run: `exp3_t6_full_20260814T022200` — **PASS**.

| Gate item | Result |
|---|---:|
| Interventions | 864/864 |
| Per-step effect rows | 45,656 |
| Unique branches | 108/108 |
| Directions and signs at each branch | 4 × 2 |
| Maximum non-arm integration L-infinity change | 0 |
| Joint-limit-valid interventions | 864/864 |
| Finite continuations | 864/864 |
| Removed branches | 0 |
| Failure examples | 0 |
| Terminal success flips | 58 |

## 7. Primary temporal-concentration result

Top-k counts used the frozen ceiling rule. With 12 branch points, top 10%, 20%, and
30% mean the top 2, 3, and 4 points. For an all-zero curve, top mass and Gini are
zero, normalized entropy is one, and an explicit all-zero flag is set.

### 7.1 Per-demonstration concentration

| Task | Demo | Top-10 mass | Top-20 mass | Top-30 mass | Gini | Norm. entropy |
|---|---|---:|---:|---:|---:|---:|
| Drawer | 0 | 0.4948 | 0.6028 | 0.6661 | 0.4139 | 0.8672 |
| Drawer | 1 | 0.2478 | 0.3669 | 0.4761 | 0.1952 | 0.9751 |
| Drawer | 2 | 0.3091 | 0.4270 | 0.5282 | 0.2672 | 0.9526 |
| Bowl | 0 | 0.3126 | 0.4493 | 0.5683 | 0.3036 | 0.9410 |
| Bowl | 1 | 0.5162 | 0.5932 | 0.6586 | 0.4442 | 0.8320 |
| Bowl | 2 | 0.4019 | 0.4892 | 0.5744 | 0.3298 | 0.9175 |
| Stove | 0 | 0.2298 | 0.3365 | 0.4259 | 0.1295 | 0.9895 |
| Stove | 1 | 0.2252 | 0.3278 | 0.4193 | 0.1221 | 0.9903 |
| Stove | 2 | 0.4312 | 0.5102 | 0.5876 | 0.3435 | 0.8971 |
| **Median** | — | **0.3126** | **0.4493** | **0.5683** | **0.3036** | **0.9410** |

The hierarchical bootstrap 95% CI for the median top-20 mass was
`[0.3579, 0.5861]`, entirely above the uniform value 0.25. Thus the measured effect
mass is not consistent with a uniformly flat 12-point curve under the frozen pilot
analysis.

Task-level median top-20 masses were 0.4270 for drawer, 0.4892 for bowl, and 0.3365
for stove. Concentration is consequently task- and demonstration-dependent.

### 7.2 Effect magnitude and breadth

Across the 108 branch medians, the primary effect distribution was:

- minimum: 0.00656;
- P25: 0.01077;
- median: 0.01451;
- P75: 0.02138;
- maximum: 0.10222.

At the individual-intervention level, 634/864 = **73.38%** exceeded 0.01. At the
branch level, 97/108 = **89.81%** had at least half of their eight interventions
above 0.01. Sensitivity is therefore often measurable and broad, even though effect
mass is moderately concentrated. The saturation rule did not fire because fewer
than 95% of interventions exceeded the threshold.

The Spearman correlation between remaining horizon and branch criticality was
-0.2245. The primary mean already normalizes by continuation duration; this weak
negative association does not explain the concentration result by itself.

## 8. Direction and sign robustness

The frozen robustness statistic was the median pairwise Spearman correlation among
the eight direction/sign temporal curves within each demonstration. Stability
required at least 0.5.

| Task | Demo | All-pair median rho | Sign-pair median rho | Direction median rho | Stable? |
|---|---|---:|---:|---:|---|
| Drawer | 0 | 0.0734 | 0.3986 | 0.0944 | No |
| Drawer | 1 | 0.2902 | 0.6329 | 0.4545 | No |
| Drawer | 2 | 0.3846 | 0.6189 | 0.4406 | No |
| Bowl | 0 | 0.5804 | 0.6783 | 0.4720 | Yes |
| Bowl | 1 | 0.5909 | 0.9196 | 0.6084 | Yes |
| Bowl | 2 | 0.1958 | 0.6469 | 0.3776 | No |
| Stove | 0 | 0.0804 | 0.4056 | 0.1084 | No |
| Stove | 1 | -0.0315 | 0.7413 | -0.0699 | No |
| Stove | 2 | 0.2343 | 0.5979 | 0.1888 | No |

Across demonstrations, the median sign-pair rho was 0.6329, but the median
direction-only rho was 0.3776 and the all-pair median was 0.2343. Opposite signs of
the same direction are moderately consistent; different q directions often rank
time differently. This is a central reason not to treat a single scalar time curve
as a fully direction-invariant decision-criticality property.

## 9. Cross-demonstration replication

Cross-demo comparisons used only the ten common temporal-quantile branches, ordered
by nominal quantile. Event branches were excluded from this test.

| Task | Demo pair | Spearman rho |
|---|---|---:|
| Drawer | 0–1 | 0.2727 |
| Drawer | 0–2 | -0.1394 |
| Drawer | 1–2 | 0.3818 |
| Bowl | 0–1 | 0.4061 |
| Bowl | 0–2 | 0.2485 |
| Bowl | 1–2 | 0.9273 |
| Stove | 0–1 | 0.1879 |
| Stove | 0–2 | 0.6485 |
| Stove | 1–2 | -0.1152 |

The task medians were 0.2727, 0.4061, and 0.1879. No task reached the frozen 0.5
replication threshold. Individual pairs can agree strongly, but that agreement is not
consistent across all three demonstrations.

The nine leave-one-demo-out median top-20 estimates all remained above the uniform
null, but the minimum was 0.4381, short of the strong-support requirement of 0.45.

## 10. Event alignment

Events were frozen from unperturbed references only. Present-event counts were nine
for contact-count change, six for a real gripper sign change, and nine for first task
predicate success. Drawer 50% gripper fallbacks were explicitly excluded.

| Event | Median inside/outside ratio | Raw permutation p | BH q | Adjusted log coefficient | Adjusted p | Adjusted BH q |
|---|---:|---:|---:|---:|---:|---:|
| Contact-count change | 1.0351 | 0.2222 | 0.3333 | 0.0621 | 0.2852 | 0.4278 |
| Gripper sign change | 0.7575 | 0.8663 | 0.8663 | -0.4021 | 0.9993 | 0.9993 |
| Predicate first true | 1.3316 | 0.0190 | 0.0570 | 0.1626 | 0.1195 | 0.3584 |

The adjusted analysis includes a remaining-horizon term, early/middle/late phase,
and demonstration fixed effects, which also control task. Event labels were permuted
within demonstration. No event survived either the adjusted test or BH FDR 0.05.

The nominal unadjusted predicate event is a useful hypothesis for another study, but
it is not valid evidence for a frozen event mechanism here. Contact and gripper events
do not explain the observed temporal concentration.

## 11. Outcome relevance

There were 58 terminal-success flips:

| Task | Success flips | Unique affected branches |
|---|---:|---:|
| Drawer | 10 | 8 |
| Bowl | 48 | 18 |
| Stove | 0 | 0 |

Every bowl demonstration showed 14–18 flips; every drawer demonstration showed 2–4;
no stove intervention flipped terminal success. This confirms that the intervention
can matter behaviorally, but the aggregate association between the primary score and
terminal task outcomes was weak:

- branch criticality versus terminal object effect: Spearman 0.0785;
- branch criticality versus predicate-divergence fraction: Spearman 0.2660.

The primary metric describes whole-horizon physical divergence, not a calibrated
probability of task failure. Success flips should not be used as a post-hoc branch
selection rule.

## 12. Component behavior

Median branch-level normalized component means were:

| Component | Median |
|---|---:|
| Arm q | 0.00282 |
| Arm qvel | 0.04301 |
| EEF position | 0.01085 |
| EEF orientation | 0.00843 |
| Task-object position | 0.00446 |
| Task-object orientation | 0.00330 |

qvel is the largest typical normalized component. Because all components remain
available separately, later work can test whether this results from the reference-only
qvel denominator or represents a genuine dynamic amplification. The primary result
must not be restated as object-level criticality alone.

## 13. Sensitivity analyses

The frozen secondary checks did not reverse the qualitative conclusion:

- using the mean rather than median across the eight interventions increased the
  median demonstration top-20 mass from 0.4493 to 0.4869;
- using only the ten temporal-quantile branches gave a median top-20 mass of 0.3286,
  compared with its 10-point uniform null of 0.20;
- top-10 and top-30 median masses were 0.3126 and 0.5683;
- no curve was all zero;
- direction sensitivity remained the main replication weakness.

These checks support non-uniformity but do not satisfy the strong replicated-sparsity
rule.

## 14. Frozen decision rule evaluation

### Strong support checks

| Requirement | Observed | Pass? |
|---|---:|---|
| Median top-20 mass >= 0.50 | 0.4493 | No |
| Bootstrap lower CI > 0.25 | 0.3579 | Yes |
| At least 2/3 tasks with median cross-demo rho >= 0.50 | 0/3 | No |
| At least 6/9 direction/sign-stable demos | 2/9 | No |
| LODO minimum top-20 >= 0.45 | 0.4381 | No |

### Partial support check

The median top-20 mass was at least 0.40 and its hierarchical CI lower bound exceeded
0.25. This satisfies the frozen partial-support rule.

### Other classifications

- saturation: false, because threshold exceedance was 73.38%, not at least 95%;
- broad-sensitivity branch condition: numerically present at 89.81%, but the frozen
  priority assigns partial support first because significant moderate concentration
  is also present;
- no support: false.

**Final formal classification: PARTIAL SUPPORT.**

## 15. Direct answers to the scientific questions

1. **Are local q effects measurable above zero noise?** Yes. The matched-zero floor
   was exactly zero and all 864 interventions ran validly.
2. **Is effect magnitude temporally uniform?** No under the frozen pilot statistic;
   the top-20 mass CI is above the uniform null.
3. **Is there strong decision sparsity?** No. Concentration is moderate and fails the
   replication, robustness, and LODO requirements.
4. **Are critical times stable across q direction and sign?** Signs within the same
   direction are moderately stable, but different directions are not reliably stable.
5. **Are critical times replicated across demonstrations?** Not consistently. No task
   met the task-level 0.5 median-rank threshold.
6. **Do contact or gripper events explain criticality?** No. Neither event was enriched.
7. **Does task-predicate completion explain it?** There is an unadjusted nominal signal,
   but it fails BH and horizon/phase/task-adjusted inference.
8. **Does the primary score predict terminal outcome relevance?** Only weakly at the
   branch level, despite 58 individual success flips.
9. **Is sparse-decision RL justified now?** No. The result motivates replication and
   better progress alignment first.

## 16. Allowed and forbidden claims

### Allowed

- In this 3-task × 3-demo LIBERO pilot, fixed 0.5%-joint-range q perturbations had
  temporally non-uniform remaining-horizon physical effects.
- The demonstration-median top-20 effect mass was 0.449 with a hierarchical 95% CI
  above the uniform 0.25 null.
- Direction dependence and cross-demo variability prevent a strong replicated-sparsity
  conclusion.
- Contact and gripper event alignment was not supported.

### Forbidden

- that a universal sparse set of decision times has been identified;
- that criticality is direction-invariant;
- that contact transitions are decision bottlenecks;
- that learned event-triggered control will outperform dense control;
- that latent actions or sparse-decision RL are superior;
- that the result generalizes beyond these tasks, demonstrations, epsilon, policy,
  simulator stack, or same-runtime continuation design;
- that the 58 success flips establish causal task-failure probabilities at selected
  times.

## 17. Reproducibility

### Software and hardware

- Python 3.8.20;
- NumPy 1.22.4;
- PyArrow 17.0.0;
- MuJoCo 3.2.3;
- robosuite 1.4.0;
- LIBERO SHA `8f1084e3132a39270c3a13ebe37270a43ece2a01`;
- robosuite source SHA `fbee5844ff5632f5b5698e204ec5357ca50be0df`;
- formal execution project SHA `e3f7507297a43964853f149c7aa1e38e9097c148`;
- NVIDIA RTX 4090 detected, but the validated MuJoCo path executed on CPU.

The run correctly records the project as dirty only because of the user's pre-existing
`README.md`, `reports/README.md`, and `prompts/` changes. They were not modified or
staged as part of EXP3 commits.

### Main commands

```powershell
C:\Users\Guoji\anaconda3\envs\libero-exp1\python.exe scripts\exp2\validate_zero_twins.py --run-id exp3_t1_condition_d_20260814T020613 --reference-run runs\exp2_r2_gripper_refs_20260814T011336 --condition-d-only

C:\Users\Guoji\anaconda3\envs\libero-exp1\python.exe scripts\exp3\freeze_protocol.py --t1-run runs\exp3_t1_condition_d_20260814T020613 --reference-run runs\exp2_r2_gripper_refs_20260814T011336

C:\Users\Guoji\anaconda3\envs\libero-exp1\python.exe scripts\exp3\run_criticality.py --run-id exp3_t5_zero_corrected_20260814T021800 --mode zero --reference-run runs\exp2_r2_gripper_refs_20260814T011336

C:\Users\Guoji\anaconda3\envs\libero-exp1\python.exe scripts\exp3\run_criticality.py --run-id exp3_t6_full_20260814T022200 --mode full --reference-run runs\exp2_r2_gripper_refs_20260814T011336 --zero-run runs\exp3_t5_zero_corrected_20260814T021800

C:\Users\Guoji\anaconda3\envs\libero-exp1\python.exe scripts\exp3\analyze_criticality.py --run-dir runs\exp3_t6_full_20260814T022200
```

Final verification: **32 tests passed**, zero failed, zero skipped. `pip check` found
no broken requirements. Two pytest cache warnings reflect denied writes to the default
`.pytest_cache`; a dedicated writable `--basetemp` was used and all assertions passed.

### Key artifact hashes

| Artifact | SHA-256 |
|---|---|
| `interventions.parquet` | `1e55bee4e112d77c768c0ae2d59c5a26a6f99ff8a48aa894bdeb260142ea1595` |
| `per_step_effects.parquet` | `57afb903641e9faefdc8d84c9c6f5d7618aa2c6435de8cea13226af7448c2964` |
| `branch_summary.parquet` | `ac312b6c5ca7b6e818ad9ac337ae1130ab1dd02c179394048cc2edce06f08d26` |
| `zero_controls.parquet` | `fe5127fe4d631671024a6c227c1fd4dfb66d687dac05ca4d7517232a6ae2ccd3` |
| `scientific_decision.json` | `b27bdd4d67f4bad9374ad74fc111bbeca0718ea3873312b04aa4fcaf797940bc` |

## 18. Artifact inventory

The formal run contains:

- `zero_controls.parquet`;
- `interventions.parquet`;
- `per_step_effects.parquet`;
- `branch_summary.parquet`;
- `demo_concentration.parquet`;
- `direction_robustness.parquet`;
- `cross_demo_rank_agreement.parquet`;
- `event_enrichment.parquet`;
- `leave_one_demo_out.parquet`;
- root copies of direction, event, and normalization manifests;
- all frozen manifests under `artifacts/frozen_manifests/`;
- `statistical_results.json`;
- `scientific_decision.json`;
- `failure_examples.json` (empty);
- the twelve required plots.

Required plots:

1. `criticality_vs_normalized_time_per_demo.png`;
2. `criticality_components_per_demo.png`;
3. `direction_sign_variability.png`;
4. `effect_mass_concentration_curves.png`;
5. `topk_effect_mass_by_demo.png`;
6. `concentration_index_by_demo.png`;
7. `cross_demo_rank_agreement.png`;
8. `event_window_enrichment.png`;
9. `criticality_vs_remaining_horizon.png`;
10. `criticality_vs_terminal_object_effect.png`;
11. `criticality_vs_predicate_divergence.png`;
12. `success_flip_locations.png`.

## 19. Limitations

1. Only three demonstrations per task are available for replication inference.
2. All continuations replay the recorded open-loop action suffix; results need not
   equal closed-loop policy sensitivity.
3. Only one epsilon, 0.5% of joint range, was studied. EXP3 is not a local-linearity
   or multi-scale experiment.
4. Four random directions do not fully characterize a seven-dimensional sensitivity
   operator.
5. The equal-weight primary composite has a larger typical normalized qvel component;
   separate channel conclusions should be preferred over an object-only reading.
6. Normalized clock time may misalign equivalent physical progress across demos.
7. Event tests use coarse branch samples and only frozen reference events.
8. The tasks, reference policy, simulator versions, and exact restoration mechanism
   limit external validity.

## 20. Final disposition

EXP3 is complete. The experiment passes all causal-execution and zero-control gates
and yields a scientifically useful but deliberately limited result: temporal q
criticality is non-uniform, yet its location is not sufficiently direction-stable or
cross-demo-replicated to justify a sparse-decision controller.

The next experiment should remain on the causal-criticality main line and test
replication with more held-out demonstrations, more directions, and outcome-blind
task-progress alignment. It should not automatically proceed to latent RL or learned
event triggering.
