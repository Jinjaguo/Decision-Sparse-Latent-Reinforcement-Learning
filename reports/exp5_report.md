# EXP5 Report: Cross-Fitted State-Conditioned Anisotropic q Criticality

**Experiment:** EXP5 — Cross-Fitted State-Conditioned Anisotropic Criticality

**Execution date:** 2026-08-14

**Status:** Complete after the user-authorized confirmatory-cohort amendment

**Formal classification:** **`finite_radius_nonlinearity_dominates`**

**Raw run:** `runs/exp5_s12_s13_q_full_locked_20260814`

**Final formal analysis:** `runs/exp5_s14_s20_formal_analysis_frozen_matching_20260814`

## 1. Executive conclusion

EXP5 completed the full amended protocol: 30 qualified same-runtime references,
five-fold outcome-blind state matching, 480 state-coverage branches, 16,896 q
interventions, 658,288 future-step effect records, 1,056 radius-specific response
operators, and all 16 required plots. No formal branch or intervention was deleted.

The result does **not** support a reproducible state-conditioned sensitive q
subspace. State matching failed to improve scalar replication over the better of
normalized time and EXP4 progress:

- median paired state improvement: **-0.03627** (required >= 0.15);
- hierarchical 95% CI: **[-0.27947, 0.03343]** (required lower bound > 0);
- permutation p and BH q: **0.9950**;
- state-matched scalar rho reached 0.60 in only Drawer, not 2/3 tasks;
- state-matched top-1 subspace similarity was 0.197 Drawer, 0.091 Bowl, and
  0.080 Stove, far below 0.70.

The main scientific result is strong radius dependence:

- only **21/480 branches (4.375%)** passed the frozen four-part local-linearity gate;
- only **1/30 demonstrations (3.33%)** had median small/main top-1 similarity >= 0.70;
- median main/large top-1 similarity was **0.1617**;
- median main/large relative spectral-norm discrepancy was **0.4369**;
- the response magnitude generally increased with radius, but the dominant q
  direction rotated substantially.

The held-out random-direction rank prediction was the one positive preregistered
diagnostic: the overall demonstration-median rho was **0.6265**, exceeding 0.60.
However, the median relative vector-prediction errors were 0.833 Drawer, 1.661 Bowl,
and 0.935 Stove, and this prediction criterion was only one of nine strong-rule
requirements. It cannot rescue the failed linearity and subspace gates.

The strongest allowed claim is:

> In these replayed LIBERO continuations, local Panda arm-q response remains
> anisotropic and causally measurable, but its finite-difference magnitude and
> dominant direction change strongly over the tested 0.0025–0.01 joint-range
> radii. The tested physical-state matcher does not make the scalar or subspace
> response sufficiently reproducible across demonstrations.

An oracle state-conditioned adaptive scheduler is **not eligible**. Latent RL is
**not eligible**. The next experiment should test radius convergence at smaller
radii before proposing a controller.

## 2. Protocol amendment and cohort

The original fixed demos 10–19 cohort produced 29/30 successful references. Drawer
`demo_17` was finite and exactly round-trippable but ended at drawer q=-0.021819,
short of the exact Open threshold q<-0.14. The user then explicitly amended EXP5:

1. retain every successful demos 10–19 reference;
2. fill only a task deficit by scanning unused demos from 20 upward;
3. accept only successful, finite, exact-round-trip references;
4. stop as soon as that task reaches ten references;
5. preserve all rejects and do not select using q outcomes.

Drawer `demo_20` was the first attempted replacement and passed. Its reference had
116 actions, final success true, all finite arrays, integration round-trip 0, and
controller round-trip 0. The immutable negative record for Drawer `demo_17` remains
in `exp5_s4_refs_16_17_20260814`.

The formal eligibility-conditioned cohort was:

| Task | Qualified demos | Count |
|---|---|---:|
| Drawer | 10–16, 18, 19, 20 | 10 |
| Bowl on plate | 10–19 | 10 |
| Stove | 10–19 | 10 |
| **Total** | — | **30** |

Run `exp5_s4_confirmatory_refs_amended_20260814` passed all reference gates:
30 references, exactly ten per task, every final success true, every array finite,
and maximum integration/controller round-trip errors both 0.

This changes the estimand from the original exact-index cohort to successful
same-runtime, ordered-eligibility-conditioned demonstrations. Results must not be
generalized to all public LIBERO demonstrations.

## 3. Outcome-blind freeze

The amended protocol and frozen manifests were committed as `e851236` before the
first q-intervention dry run. The freeze used only reference trajectories:

| Frozen item | Result |
|---|---:|
| Development demos | 21 (3–9, seven/task) |
| Confirmatory demos | 30 |
| Reference descriptor states | 5,385 |
| Crossfit folds | 5/task, two held-out demos/fold |
| State prototypes | 16/task/fold |
| Confirmatory branches | 480 (16/demo) |
| Direction rows across radii | 8,448 |
| Signed interventions | 16,896 |
| Large-radius subset | 96 branches (20%) |
| Frozen manifests | 23 plus hash index |

The physical descriptor contains normalized Panda q/qvel, EEF position and 6D
orientation, EEF linear/angular velocity, gripper opening and
`PandaGripper.current_action`, audited task contact/predicate fields, and exact
task-object geometry. Continuous scaling came from development references with
physical floors; binary channels received weight 0.25. Each held-out fold used a
shrinkage Mahalanobis precision fitted without that fold.

Prototype coverage accepted 270/480 branches under the frozen training-reference
threshold and preserved 210 threshold exceedances. All 480 branches were still
intervened on, as required. Primary state-matched inference excluded rejected
state pairs but retained their counts and reasons. This produced 827 accepted and
1,346 rejected path pairs across the three task families; some Bowl/Stove demo
pairs had fewer than three accepted points and therefore received the frozen zero
rank-correlation convention.

## 4. Direction and radius design

At every branch, EXP5 used seven deterministic QR-orthonormal directions plus one
independent held-out random unit direction, both signs. The coordinate system was
joint-range-scaled Panda q. Every proposed q remained within audited limits.

| Radius | Branches | Directions/signs | Interventions |
|---|---:|---:|---:|
| 0.0025 (`small`) | 480 | 8 x 2 | 7,680 |
| 0.0050 (`main`) | 480 | 8 x 2 | 7,680 |
| 0.0100 (`large`) | 96 | 8 x 2 | 1,536 |
| **Total** | — | — | **16,896** |

## 5. Environment and validation

| Item | Result |
|---|---|
| Conda environment | `libero-exp1` |
| Python | 3.8.20 |
| PyTorch | 1.11.0+cu113 |
| GPU | NVIDIA GeForce RTX 4090 |
| Formal analysis dtype/device | float64 / cuda:0 |
| Simulator | CPU MuJoCo/robosuite corrected-D path |
| Final tests | 47/47 passed |
| `pip check` | no broken requirements |

The original pytest invocation produced 45 passes and two fixture errors because
the system pytest temp directory was inaccessible. Re-running unchanged tests with
`runs/pytest_exp5_20260814` as `--basetemp` passed 47/47. This was not a test
assertion failure.

GPU run `exp5_s10_gpu_equivalence_20260814` passed float64 equivalence for central
operators, Gram matrices, singular spectra, and projector calculations across all
tasks and primary radii. Formal analysis used the RTX 4090 for Gram/eigendecomposition
and bootstrap quantiles. MuJoCo stepping remained on CPU.

## 6. Dry run and complete zero gate

Dry run `exp5_s11_dry_20260814` passed one branch, one direction, and both signs:
two interventions, 120 effect steps, all finite, joint-limit valid, and non-arm Linf
0.

Full matched-zero run `exp5_s11_zero_full_20260814` passed:

| Gate | Result |
|---|---:|
| Branches | 480/480 |
| Zero continuations | two per branch |
| Saved zero-reference suffix steps | 18,614 |
| Integration L2 median / P95 / max | 0 / 0 / 0 |
| Terminal object-pose P95 | 0 |
| Success agreement | 100% |
| Failures | 0 |

This is direct regression evidence that corrected Condition D remained exact on the
entire EXP5 branch set.

## 7. Formal q execution and raw lock

The sweep ran as 15 immutable two-demo shards. Every shard independently passed
expected coverage, both signs, every frozen radius/direction, finite arrays, joint
limits, and non-arm preservation. The merged gate passed:

| Quantity | Result |
|---|---:|
| Branches | 480 |
| Interventions | 16,896/16,896 |
| Future-step effect records | 658,288 |
| Unique frozen keys | 16,896 |
| Maximum non-arm Linf | 0 |
| Execution failures | 0 |
| Intervention-level success flips | 1,955 |

Raw SHA-256 locks, committed before inference as `dd34ad3`:

| Artifact | SHA-256 |
|---|---|
| `zero_controls.parquet` | `1ed4254218f3906b53f1c48383d098e872f537b341c59a728cf7c53df2897256` |
| `zero_reference_steps.parquet` | `afb75eb1b4ac0d301b5fd9942ab23c11ba39cd0beffd648d8ebab48d8784162d` |
| `interventions.parquet` | `0a7101937ac22f30c9b8cda1e583734d5e1f2461539fe17546f1945b668d5129` |
| `per_step_effects.parquet` | `8965264781dd098fa970093d7bed22b9301588c13af7a55ffb3528626d0539a0` |

## 8. Radius-specific scalar and operator summaries

Median `S_RMS` by task and radius:

| Task | 0.0025 | 0.005 | 0.01 subset |
|---|---:|---:|---:|
| Drawer | 0.01522 | 0.03629 | 0.04880 |
| Bowl | 0.02681 | 0.04902 | 0.08139 |
| Stove | 0.02185 | 0.03993 | 0.06978 |

Median operator geometry:

| Task/radius | Spectral norm | Leading share | Effective rank |
|---|---:|---:|---:|
| Drawer 0.0025 / 0.005 / 0.01 | 32.71 / 39.04 / 28.25 | 0.801 / 0.791 / 0.746 | 1.98 / 2.02 / 2.21 |
| Bowl 0.0025 / 0.005 / 0.01 | 47.66 / 72.65 / 36.93 | 0.771 / 0.753 / 0.776 | 2.14 / 2.15 / 2.10 |
| Stove 0.0025 / 0.005 / 0.01 | 71.56 / 57.21 / 53.72 | 0.839 / 0.832 / 0.796 | 1.71 / 1.82 / 1.91 |

Anisotropy is therefore present at every tested radius, but anisotropy alone does
not mean the same physical q direction is sensitive across radius or trajectory.

## 9. Cross-radius linearity

The frozen local-linearity gate required all of:

- small/main top-1 similarity >= 0.70;
- relative spectral discrepancy <= 0.35;
- sign asymmetry <= 0.40;
- held-out relative prediction error <= 0.50.

Pass rates were:

| Task | Passing branches |
|---|---:|
| Drawer | 2/160 (1.25%) |
| Bowl | 11/160 (6.875%) |
| Stove | 8/160 (5.0%) |
| **Total** | **21/480 (4.375%)** |

Median small/main top-1 similarities were 0.174 Drawer, 0.139 Bowl, and 0.317
Stove. Median relative spectral discrepancies were 0.379, 0.477, and 0.387.
At 0.01, median main/large top-1 similarity was 0.162 overall and median relative
spectral discrepancy was 0.437. Thus 0.0025 was not a demonstrably stable local
limit relative to 0.005; 0.01 increased the evidence for finite-radius behavior.

Because the gate failed broadly, these matrices are called **finite-radius response
operators**, not Jacobians.

## 10. State-conditioned scalar replication

Primary inference used the frozen development scaling, the symmetric average of the
two held-out folds' reference-only precision matrices, monotone matching with the
0.25 temporal window, and frozen rejection flags.

| Task | Time rho | Progress rho | State rho | State improvement |
|---|---:|---:|---:|---:|
| Drawer | 0.8206 | 0.8206 | 0.7576 | -0.0010 |
| Bowl | 0.4971 | 0.4853 | 0.2000 | -0.2500 |
| Stove | 0.2147 | 0.2412 | 0.0000 | -0.0882 |

Across all demo pairs, median improvement over the better EXP4 baseline was
-0.03627, CI [-0.27947, 0.03343], permutation p=0.9950, BH q=0.9950. LODO and LOTO
stability requirements both failed. State matching therefore did not improve
cross-demo scalar replication under cross-fitting.

## 11. Subspace replication

State-matched median physical-q projector similarities:

| Task | Top-1 | Top-2 |
|---|---:|---:|
| Drawer | 0.1968 | 0.2762 |
| Bowl | 0.0906 | 0.2374 |
| Stove | 0.0803 | 0.2112 |

No task reached the top-1 threshold 0.70, whereas the strong rule required at least
two. Top-2 results were also low. Principal q directions were highly state-,
trajectory-, and radius-dependent even after physical-state matching.

## 12. Held-out direction prediction

Per-task demonstration-median rank rho and vector relative error:

| Task | Prediction rho | Relative vector error |
|---|---:|---:|
| Drawer | 0.7426 | 0.8330 |
| Bowl | 0.5191 | 1.6610 |
| Stove | 0.5868 | 0.9354 |

The overall demo-median rho was 0.6265, passing its isolated 0.60 threshold. This
means the seven-direction operator often ranked branch response magnitude usefully,
especially in Drawer. It did not accurately reproduce the signed held-out response
vector and failed the 0.50 relative-error component of the linearity gate at most
branches.

## 13. Terminal relevance

At the main radius, branch-level terminal-consequence Spearman correlations were:

| Task | Scalar `S_RMS` rho | Spectral-norm rho | Branches with any success flip |
|---|---:|---:|---:|
| Drawer | 0.6635 | 0.6343 | 99 |
| Bowl | 0.8496 | 0.8757 | 141 |
| Stove | 0.4427 | 0.1921 | 23 |

Neither scalar nor spectral sensitivity uniformly dominated: spectral norm was
slightly better for Bowl, while scalar `S_RMS` was better for Drawer and especially
Stove. These are secondary associations under replayed open-loop continuation, not
evidence that a scheduler would improve control.

## 14. Formal decision

| Strong requirement | Result |
|---|---|
| State improvement >= 0.15 | fail (-0.0363) |
| Bootstrap lower bound > 0 | fail (-0.2795) |
| 2/3 task state rho >= 0.60 | fail (1/3) |
| 2/3 task top-1 >= 0.70 | fail (0/3) |
| 70% demos cross-radius top-1 >= 0.70 | fail (3.33%) |
| Held-out prediction median rho >= 0.60 | **pass (0.6265)** |
| LODO positive | fail |
| LOTO stable | fail |
| Confirmatory BH-FDR 0.05 | fail (q=0.9950) |

Only one of nine requirements passed. Because only 4.375% of branches satisfied
the preregistered local-linearity gate, the first applicable classification in the
frozen priority order is **`finite_radius_nonlinearity_dominates`**.

## 15. Analysis implementation incidents

Three analysis directories were intentionally preserved:

1. the first import attempt stopped because pandas was absent;
2. the second computed derived tables but stopped while serializing NumPy booleans;
3. a completed diagnostic run revealed that pairwise matching had not yet enforced
   frozen development scaling/precision and rejection flags.

The final run used a new directory, verified raw hashes first, enforced the frozen
matching definition, recomputed every derived result, produced 16 non-placeholder
plots, and passed. No raw artifact or threshold was altered. `pandas==2.0.3` was
installed in the conda environment; final `pip check` remained clean.

## 16. Required artifacts and plots

The raw run contains zero controls, interventions, per-step effects, signed vectors
inside each intervention/effect row, frozen manifests, failure examples, and the raw
hash lock. The final analysis run contains scalar summaries, 1,056 operators and
matrices, subspace similarities, cross-radius linearity, state-match tables,
crossfit results, held-out prediction, terminal relevance, GPU audits, the frozen
scientific decision, analysis hashes, and an empty failure list.

All 16 required plots were generated, including state-distance, alignment,
scalar-replication, top-1/top-2, spectra, eigenvalue share, cross-radius consistency,
sign asymmetry, held-out prediction, terminal relevance, task specificity, LODO/LOTO,
and GPU equivalence figures.

## 17. Exact principal commands

```powershell
# Complete the amended Drawer reference cohort
python scripts/exp5/generate_confirmatory_references.py `
  --run-id exp5_s4_drawer_completion_demo20_20260814 `
  --episodes 20 --task-names open_the_middle_drawer_of_the_cabinet

# Merge and gate 30 qualified references
python scripts/exp5/merge_confirmatory_references.py `
  --run-id exp5_s4_confirmatory_refs_amended_20260814 `
  --source-runs <five original shards> exp5_s4_drawer_completion_demo20_20260814

# Freeze state matching, branches, directions, radii, and decision rule
python scripts/exp5/freeze_protocol.py `
  --run-id exp5_s1_s9_protocol_freeze_amended_20260814 `
  --development-reference-run runs/exp4_e4_3_heldout_refs_20260814 `
  --confirmatory-reference-run runs/exp5_s4_confirmatory_refs_amended_20260814

# GPU equivalence and tests
python scripts/exp5/validate_gpu_backend.py --run-id exp5_s10_gpu_equivalence_20260814
python -m pytest -q --basetemp runs/pytest_exp5_20260814 -p no:cacheprovider
python -m pip check

# Complete zero gate
python scripts/exp5/run_zero_controls.py `
  --run-id exp5_s11_zero_full_20260814 `
  --reference-run runs/exp5_s4_confirmatory_refs_amended_20260814

# Fifteen immutable q shards used trajectory-start 0,2,...,28
python scripts/exp5/run_interventions.py `
  --run-id exp5_s12_q_00_01_20260814 `
  --reference-run runs/exp5_s4_confirmatory_refs_amended_20260814 `
  --zero-run runs/exp5_s11_zero_full_20260814 `
  --trajectory-start 0 --max-trajectories 2

# Exact merge and raw lock
python scripts/exp5/merge_intervention_shards.py `
  --run-id exp5_s12_s13_q_full_locked_20260814 `
  --zero-run runs/exp5_s11_zero_full_20260814 `
  --source-runs <all 15 q shards>

# Final frozen-matching analysis
python scripts/exp5/analyze_exp5.py `
  --run-id exp5_s14_s20_formal_analysis_frozen_matching_20260814 `
  --raw-run runs/exp5_s12_s13_q_full_locked_20260814 `
  --development-run runs/exp5_s1_s9_protocol_freeze_amended_20260814 `
  --gpu-run runs/exp5_s10_gpu_equivalence_20260814
```

## 18. Claim boundary and next action

Supported:

- corrected-D and matched-zero branching remain exact on all 480 EXP5 branches;
- q sensitivity is anisotropic and predicts some held-out magnitude rankings;
- response geometry changes strongly across the tested finite radii;
- the frozen state matcher does not improve cross-demo replication.

Not supported:

- a radius-invariant Jacobian or sensitive q subspace;
- state-conditioned replicated anisotropic criticality;
- universal sparse decision times;
- an oracle adaptive scheduler;
- sparse-policy or latent-RL benefit.

The single highest-value next action is a smaller-radius convergence experiment
with 0.000625, 0.00125, 0.0025, and 0.005 joint-range perturbations on a frozen,
high-zero-margin subset, explicitly estimating numerical signal-to-zero-floor and
subspace convergence before any controller experiment.
