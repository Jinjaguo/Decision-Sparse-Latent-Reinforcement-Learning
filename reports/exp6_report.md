# EXP6 Report: Multi-Radius Convergence and Trust-Region Identification

**Experiment:** EXP6 — Multi-Radius Convergence and Trust-Region Identification of q Response

**Execution date:** 2026-08-14

**Status:** Complete

**Formal classification:** **`contact_mode_conditioned_convergence`**

**Locked raw run:** `runs/exp6_s11_formal_raw_locked_20260814`

**Formal analysis:** `runs/exp6_s12_formal_analysis_cpu_20260814`

## 1. Executive conclusion

EXP6 completed the full preregistered estimand-validation experiment on the frozen
30-demonstration EXP5 cohort. It used 240 outcome-blind branches, five calibrated
radii, seven orthonormal basis directions plus one held-out random direction, and
both perturbation signs. The final locked dataset contains:

- 19,200 formal q interventions;
- 774,560 future policy-step effect records;
- 1,200 radius-specific finite-radius response operators;
- 960 adjacent-radius comparisons;
- 9,600 exact named contact-mode plus/minus comparisons;
- 542 terminal-success flips;
- 16 required diagnostic plots.

The response was measurable far above the exact matched-zero floor, but it did not
converge to a generally stable local operator as radius decreased. For the primary
`0.000625` versus `0.00125` comparison:

- 0/30 demonstrations had median top-1 projector similarity at least 0.80;
- the hierarchical median top-1 95% interval was [0.3138, 0.5931], entirely below
  the required lower bound of 0.65;
- only 89/240 branches (37.08%) met the spectral-discrepancy threshold of 0.20;
- only 6/240 branches (2.50%) passed the full six-part adjacent-radius gate.

At the smallest validated radius (`0.0003125`), held-out response-norm ranking was
strong (`demo-median rho = 0.7381`), but the median signed-vector relative error was
0.7049, twice the allowed 0.35. This repeats the EXP5 pattern: coarse response
ranking can be useful even when the vector-valued local-linear interpretation is
not valid.

Only 7/240 branches (2.92%) had any resolved empirical trust region, all in the Bowl
task. Contact-mode divergence was associated with a higher convergence-failure
rate: 100% (165/165) for divergent branches versus 92% (69/75) for preserved
branches. The demonstration-clustered difference was +0.0778 with 95% bootstrap CI
[0.0222, 0.1389], permutation p=0.01525, and BH q=0.01525. Therefore H4 passed and,
under the frozen priority rule, the classification is
`contact_mode_conditioned_convergence`.

This is a mechanism result, not a controller result. Contact-mode preservation is
not sufficient for convergence—the preserved group still failed 92% of the time.
It only establishes that mode switching adds a statistically detectable failure
mechanism. Scheduler and latent-RL eligibility remain false.

## 2. Protocol, cohort, and outcome-blind branch subset

The controlling protocol was frozen in
`prompts/EXP6_CODEX_PROMPT_Multi_Radius_Convergence_Trust_Region.md`. EXP6 reused the
same 30 qualified corrected-D references as EXP5:

| Task | Demonstrations | Count |
|---|---|---:|
| Drawer | 10–16, 18, 19, 20 | 10 |
| Bowl on plate | 10–19 | 10 |
| Stove | 10–19 | 10 |
| **Total** | — | **30** |

Drawer demo17 remains preserved as the failed EXP5 qualification case.

Run `exp6_s1_branch_subset_20260814` selected eight unique branches per demo using
only unperturbed reference information: normalized time, task progress, exact
target–gripper contact state, gripper-command state, and exact predicate phase. It
did not read EXP5 q criticality, operator geometry, success flips, or held-out
prediction quality.

Reference coverage was:

| Branch stratum | Count |
|---|---:|
| Exact target–gripper contact | 134 |
| No target–gripper contact | 106 |
| **Total** | **240** |

The first formal manifest freeze correctly stopped before writing outputs because
two Stove demo11 branches were not valid for every frozen direction at the maximum
radius. The remedy was reference-only and deterministic: choose the nearest unused
boundary whose q state passed all frozen joint-limit checks. Action 47 was replaced
by 46 and action 48 by 49. No response outcome was used, no branch was deleted, and
all radii were retained. The failed pre-freeze attempt is preserved as
`exp6_s6_formal_freeze_20260814`; the passing freeze is
`exp6_s6_formal_freeze_rerun_20260814`.

## 3. Numerical-resolution calibration

The calibration protocol was committed before calibration outcomes. It used two
reference-only branches per task, all five candidate radii, four matched-zero
continuations per branch, and two repetitions of every signed intervention.

| Item | Result |
|---|---:|
| Calibration branches | 6 |
| Candidate radii | 5 |
| Repeated signed interventions | 960 |
| Calibration future-step records | 56,960 |
| Non-arm integration Linf maximum | 0 |
| q injection maximum absolute error | 2.21e-16 |
| Scalar repeat error | 0 |
| Signed-vector repeat error | 0 |
| Operator/spectrum repeat error | 0 |
| Direction/sign rank disagreements | 0 |
| Measured zero floor | exactly 0 |
| Smallest-radius minimum signal/resolution ratio | 1.59e9 |

Every radius passed all frozen numerical gates. Therefore the optional
`0.0003125` radius was admitted before formal outcomes, increasing the formal
budget from 15,360 to 19,200 interventions.

The first calibration-zero command was blocked before simulator execution because
the restricted process could not write robosuite's fixed `C:\tmp\robosuite.log`.
That failed run, `exp6_s3_calibration_zero_20260814`, was preserved. The unchanged
rerun `exp6_s3_calibration_zero_rerun_20260814` passed.

## 4. GPU/CPU equivalence decision

The RTX 4090 float64 audit `exp6_s7_gpu_equivalence_20260814` covered all three
tasks and all five radii. Most kernels agreed tightly:

| Quantity | Maximum absolute CPU/GPU difference | Frozen gate |
|---|---:|---:|
| Central-difference operator | 9.09e-13 | 1e-11 |
| Top-1/top-2 projector | 5.27e-15 | 1e-10 |
| Bootstrap quantiles | 0 | 1e-12 |
| Gram matrix | 1.19e-7 | 1e-8 / manifest absolute gate |
| Eigenspectrum | 1.79e-7 | 1e-8 / manifest absolute gate |

The Gram and eigenspectrum checks failed the frozen absolute tolerance. EXP6 did
not replace those limits with relative tolerances after seeing the result. In
accordance with the protocol, formal GPU analysis stopped and CPU remained the
sole source of truth. The failed GPU audit is retained as a required negative
artifact and the `gpu_cpu_equivalence.png` plot is explicitly labeled as failed.

## 5. Dry run and corrected-D formal zero gate

Run `exp6_s8_formal_dry_20260814` exercised one branch, all five radii, all eight
directions, and both signs:

- 80 interventions;
- 9,760 future-step rows;
- no non-arm change;
- no finite/joint-limit/injection failure.

Run `exp6_s9_formal_zero_20260814` then executed two zero continuations at all 240
formal branches. It preserved 9,682 reference future-step rows. Every repeated
difference was exactly zero, including integration state and terminal object pose;
all successes agreed and all arrays were finite.

## 6. Formal intervention execution and raw lock

The sweep was partitioned before execution into ten disjoint three-trajectory
shards, `exp6_s10_formal_shard_00_20260814` through
`exp6_s10_formal_shard_09_20260814`. Each shard completed exactly 1,920
interventions and passed its coverage gate.

The merge `exp6_s11_formal_raw_locked_20260814` verified:

- exactly 19,200 intervention rows;
- every row unique;
- exact equality to all frozen task/demo/branch/radius/direction/sign keys;
- 3,840 interventions at each radius;
- all q values within audited limits;
- maximum q injection error 2.220446049250313e-16;
- maximum non-arm integration Linf 0;
- all arrays finite;
- exact contact-pair fields present;
- no post-outcome branch deletion.

Locked hashes:

| Raw artifact | SHA-256 |
|---|---|
| `zero_controls.parquet` | `88de29ac4bbf8ba4e6ce9b15ae3d271288346031586df8c60899ca93842c4a98` |
| `zero_reference_steps.parquet` | `84fd133917618ac9c9050f8e71d658cfdd6d0be6213d40f96561b08406ac3ebd` |
| `interventions.parquet` | `36f30985823f69d9518942f6138d69f836da82f21cefcd828d6b28f47d869a59` |
| `per_step_effects.parquet` | `56e3c85fcdedbeb936c08d57213ebd7edeae8d646e9d039f2157c549c60d14cc` |

## 7. Radius-resolved operator results

The signed physical-output definition was unchanged from EXP5. Each operator used
the antithetic central difference over seven basis directions. These remain
**finite-radius response operators**, not Jacobians.

Median operator diagnostics by radius:

| Radius | Spectral norm | Frobenius norm | Leading share | Effective rank | Max sign asymmetry | Signal/floor |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0003125 | 44.633 | 49.268 | 0.794 | 1.947 | 0.949 | 1.63e9 |
| 0.000625 | 37.338 | 41.955 | 0.812 | 1.921 | 0.913 | 2.93e9 |
| 0.00125 | 36.004 | 39.633 | 0.796 | 1.951 | 0.897 | 4.85e9 |
| 0.0025 | 30.990 | 35.696 | 0.775 | 2.127 | 0.929 | 9.53e9 |
| 0.005 | 31.713 | 38.227 | 0.758 | 2.153 | 0.921 | 1.87e10 |

The large signal/floor ratios rule out numerical under-resolution. However, median
sign asymmetry near 0.9 at every radius is far above the trust threshold of 0.25,
showing that plus and minus responses are not behaving as a common local linear
map even at the smallest validated radius.

## 8. Adjacent-radius convergence

| Adjacent pair | Median top-1 | Median top-2 | Median spectral discrepancy | Full passes |
|---|---:|---:|---:|---:|
| 0.0003125–0.000625 | 0.424 | 0.383 | 0.250 | 7/240 (2.92%) |
| 0.000625–0.00125 | 0.435 | 0.398 | 0.285 | 6/240 (2.50%) |
| 0.00125–0.0025 | 0.308 | 0.379 | 0.384 | 2/240 (0.83%) |
| 0.0025–0.005 | 0.216 | 0.346 | 0.317 | 0/240 (0%) |

For the preregistered H1/H2 pair, task-median branch top-1 similarity was 0.425
Drawer, 0.576 Bowl, and 0.369 Stove. No demonstration reached the required median
0.80. Spectral-threshold pass fractions were 23.75% Drawer, 53.75% Bowl, and
33.75% Stove.

The smallest pair is somewhat more stable than the largest pair, but it is not
close to satisfying the frozen population-level convergence rule.

## 9. Held-out prediction

| Radius | Median vector error | Median actual norm | Median predicted norm |
|---:|---:|---:|---:|
| 0.0003125 | 0.705 | 13.284 | 16.344 |
| 0.000625 | 0.770 | 10.697 | 13.410 |
| 0.00125 | 0.612 | 10.174 | 11.406 |
| 0.0025 | 0.747 | 10.315 | 11.981 |
| 0.005 | 0.987 | 9.570 | 12.357 |

At the smallest radius, demonstration-median rank rho was 0.738 overall (0.595
Drawer, 0.893 Bowl, 0.726 Stove), exceeding the rank threshold. Vector error failed
overall and for every task: task medians were 0.861, 0.500, and 0.867. H3 therefore
failed.

## 10. Empirical trust regions

Only 7/240 branches resolved any adjacent-radius interval:

| Largest passing interval | Count |
|---|---:|
| 0.0003125–0.000625 | 1 |
| 0.000625–0.00125 | 4 |
| 0.00125–0.0025 | 2 |
| 0.0025–0.005 | 0 |
| Unresolved | 233 |

All seven resolved branches belonged to Bowl demos 10, 12, 13, 15, 16, 18, and
19. Drawer and Stove each resolved 0/80. This does not support a global or
task-general local q trust region.

## 11. Exact contact-mode analysis

The analysis preserved exact named target–gripper contact pairs at future steps
1, 3, 5, and 10. Direction-pair contact divergence increased with radius:

| Radius | Divergent direction pairs | Fraction |
|---:|---:|---:|
| 0.0003125 | 1,078/1,920 | 56.15% |
| 0.000625 | 1,131/1,920 | 58.91% |
| 0.00125 | 1,186/1,920 | 61.77% |
| 0.0025 | 1,244/1,920 | 64.79% |
| 0.005 | 1,261/1,920 | 65.68% |

For the primary adjacent pair, convergence failure was 100% when any exact contact
mode diverged and 92% when contact mode was preserved. Only Bowl contributed
within-task passing branches; Drawer and Stove failed regardless of contact-mode
stratum. The clustered H4 test still passed:

- mean demo-level failure-rate difference: +0.07778;
- bootstrap 95% CI: [0.02222, 0.13889];
- permutation p: 0.01525;
- BH q: 0.01525;
- eligible demo clusters: 30.

Thus contact switching is a supported additional failure mechanism, but ordinary
curvature/asymmetry inside preserved modes remains the dominant unresolved issue.

## 12. Formal hypothesis decisions

| Hypothesis | Frozen requirement | Result | Decision |
|---|---|---|---|
| H1 small-radius subspace | >=70% demos median top-1 >=0.80 and CI lower >0.65 | 0%; CI [0.3138, 0.5931] | Fail |
| H2 scale convergence | >=70% branches spectral discrepancy <=0.20 | 37.08% | Fail |
| H3 held-out prediction | rho >=0.65 and vector error <=0.35 | rho 0.7381; error 0.7049 | Fail |
| H4 contact explanation | positive clustered effect, CI excludes 0, BH q<0.05 | +0.0778; CI [0.0222, 0.1389]; q=0.01525 | **Pass** |

The frozen priority order assigns `contact_mode_conditioned_convergence` because H4
passes and the full H1/H2/H3 convergence rule does not.

## 13. Secondary outcome: terminal-success flips

Success flips increased sharply with radius:

| Radius | Flips | Fraction of signed interventions |
|---:|---:|---:|
| 0.0003125 | 5 | 0.13% |
| 0.000625 | 12 | 0.31% |
| 0.00125 | 25 | 0.65% |
| 0.0025 | 112 | 2.92% |
| 0.005 | 388 | 10.10% |

By task there were 217 Drawer, 312 Bowl, and 13 Stove flips. These were not used
for branch selection or radius admission.

## 14. Required output audit

`scripts/exp6/audit_outputs.py` passed every output criterion:

- 18/18 required formal artifacts present;
- 16/16 required non-placeholder plots present;
- raw copied artifacts still match their pre-analysis hashes;
- 19,200 intervention rows;
- 1,200 operator rows;
- 960 adjacent comparisons;
- 240 trust-region rows;
- 9,600 contact-mode pair rows.

## 15. Software and reproducibility

| Component | Version / identity |
|---|---|
| Python | 3.8.20 |
| NumPy | 1.22.4 |
| SciPy | 1.10.1 |
| PyArrow | 17.0.0 |
| MuJoCo | 3.2.3 |
| robosuite | 1.4.0 |
| LIBERO SHA | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| Formal physics | corrected-D CPU MuJoCo |
| GPU | RTX 4090 audit attempted; formal use rejected by gate |
| Calibration freeze commit | `097d1c2` |
| Formal freeze commit | `34611e4` |
| Pre-outcome analysis implementation | `7492005` |

All 56 tests passed after implementation and `pip check` reported no broken
requirements. The final post-report verification is recorded in the research log.

## 16. Failures and warnings retained

1. The sandbox-only calibration-zero launch failed before simulator execution due
   to robosuite's fixed log path; the unchanged rerun passed.
2. The first formal freeze stopped on joint-limit invalidity; two deterministic
   reference-only replacements were made before formal outcomes.
3. GPU formal analysis was rejected after its frozen absolute-equivalence gate
   failed; CPU results are authoritative.
4. robosuite emitted its standard missing-private-macro warning and Gym emitted its
   maintenance warning. Neither changed numerical gates.
5. EXP3/EXP4 summary paths named in the prompt did not exactly match the repository
   directory names; their controlling prompts, manifests, reports, and run evidence
   were used, and no missing result was invented.

## 17. Allowed scientific conclusion

The strongest allowed conclusion is:

> In the tested corrected-D LIBERO continuations, Panda arm-q response is precisely
> measurable but does not converge to a generally stable radius-invariant local
> operator over 0.0003125–0.005 joint-range perturbations. Exact contact-mode
> divergence adds a statistically detectable convergence-failure mechanism, while
> most mode-preserved branches also remain nonlinear or asymmetric. A small local
> trust region is observed only for seven Bowl states.

The experiment does **not** validate contact as a scheduler trigger, a global
Jacobian, a reusable sensitive q subspace, an oracle adaptive controller, or latent
RL. The next experiment must first test a hybrid contact-mode-conditioned response
field with independent contact-mode validation.
