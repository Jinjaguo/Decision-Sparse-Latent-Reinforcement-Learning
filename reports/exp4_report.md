# EXP4 Report: Replicated, Direction-Resolved, Progress-Aligned q Criticality

**Experiment:** EXP4 — Replicated, Direction-Resolved, Progress-Aligned Criticality

**Execution date:** 2026-08-14

**Formal classification:** **Replicated non-uniformity without aligned sparse times**

**Primary formal run:** `runs/exp4_e4_12_full_20260814`

**Passing matched-zero run:** `runs/exp4_e4_11_zero_20260814`

## 1. Executive conclusion

EXP4 completed the preregistered held-out replication: 3 LIBERO tasks, 7 new
demonstrations per task, 12 outcome-blind branches per demonstration, a complete
seven-direction orthonormal basis plus one held-out random direction per branch, and
both perturbation signs. All **4,032 interventions**, **252 branches**, and **211,808
future-step effect records** passed the execution gates. No branch was removed. The
largest non-arm integration-state change at the intervention boundary was exactly
zero.

The experiment independently replicated temporal non-uniformity:

- held-out demonstration-median top-20% effect mass: **0.4507**;
- hierarchical bootstrap 95% CI: **[0.4360, 0.5560]**;
- uniform 12-branch null: **0.25**.

It did not establish a replicated set of sparse, progress-aligned decision times:

- the median improvement from physical-progress alignment over normalized time was
  only **0.0437**, 95% CI **[-0.0333, 0.1749]**, one-sided cluster-permutation
  **p = 0.2549**;
- only **5/21 demonstrations (23.8%)** met the frozen direction-robustness rule;
- held-out random-direction agreement had median Spearman **0.4336**;
- the weakest leave-one-demo-out concentration estimate was **0.4491**, narrowly
  below its 0.45 gate;
- the observed median top-20 mass was below the stronger 0.50 target.

The strongest defensible claim is therefore narrower than decision sparsity:

> In held-out successful demonstrations of three tested LIBERO tasks, small local
> Panda arm-q perturbations produce non-uniform remaining-horizon physical effects,
> but the most sensitive locations are not sufficiently invariant to direction,
> demonstration, or the preregistered scalar progress alignment to support a
> universal sparse-time controller.

EXP4 did not train a latent policy, sparse controller, event detector, or RL agent.

## 2. Scientific question and experimental unit

The primary question was whether EXP3's moderate temporal concentration would
replicate on demonstrations 3–9 and become more reproducible after replacing four
random projections with a full direction basis and aligning trajectories by audited
physical task progress.

The intervention remained:

```text
delta_q[j] = sign * 0.005 * verified_joint_range[j] * unit_direction[j]
```

The experimental unit for confirmatory inference was the demonstration. Simulator
steps and individual directions were repeated measurements, not independent sample
units.

## 3. Frozen experimental scope

| Task | Task ID | Held-out demos | Branches | Interventions |
|---|---:|---:|---:|---:|
| `open_the_middle_drawer_of_the_cabinet` | 0 | 3–9 | 84 | 1,344 |
| `turn_on_the_stove` | 7 | 3–9 | 84 | 1,344 |
| `put_the_bowl_on_the_plate` | 8 | 3–9 | 84 | 1,344 |
| **Total** | — | **21** | **252** | **4,032** |

Each branch used seven QR-generated orthonormal directions in joint-range-scaled
coordinates and one independent PCG64 random direction. Both signs were run for all
eight directions. The random direction was excluded from the primary seven-basis
`S_RMS` aggregate and retained as a genuine held-out diagnostic.

The formal direction seed was `129054049736580372`, distinct from EXP3. Across all
252 bases, maximum orthogonality error was `1.11e-15`, determinants were positive,
and every one of the 4,032 signed perturbations was within the audited joint limits.

## 4. Runtime progress audit

The task variables were resolved from the active environment and MuJoCo model, not
only from object-name strings:

| Task | Runtime state used for progress | Exact audit result |
|---|---|---|
| Drawer | middle-drawer joint coordinate | joint `wooden_cabinet_1_middle_level`, id 14, qpos address 38, range `[-0.16, 0.01]`, open when `q < -0.14` |
| Stove | button joint coordinate | joint `flat_stove_1_button`, id 16, qpos address 40, range `[-0.005, 2.1]`, on when `q >= 0.5` |
| Bowl | reach/lift/transport/place physical features | bowl and plate vertical bounds from the union AABB of active group-0 collision geoms |

Two failed audits were preserved. The first addressed the outer wrapper instead of
the inner environment. The second assumed generic bowl/plate top and bottom sites,
which these runtime objects do not expose. The passing audit uses collision geometry
and does not manufacture missing sites.

Progress was evaluated in original temporal order with first-crossing interpolation;
the analysis never sorted non-monotone progress values. Raw progress was non-monotone
in 7/7 references for every task. After frozen clipping, drawer had no backward
steps, stove had 4 small backtracks across two demonstrations (maximum 0.00115), and
bowl retained 30 backtracks across all seven demonstrations (maximum 0.11594). This
is an important limitation of the scalar progress representation.

## 5. Reference and deterministic-branching gates

### 5.1 Held-out references

Run `exp4_e4_3_heldout_refs_20260814` passed:

- exactly 21 successful references, demos 3–9 for all three tasks;
- every snapshot finite;
- maximum MuJoCo integration-state round-trip error: **0**;
- maximum controller round-trip error: **0**;
- reference action counts: drawer 121–158, stove 77–96, bowl 80–105;
- every pre-policy boundary stored the frozen task-progress channels.

The known public-HDF5 legacy-state mismatch remained visible as a diagnostic and was
not confused with the corrected same-runtime substrate.

### 5.2 Corrected Condition D

The full corrected-D regression executed **756 matched pairs**: 252 branches times
three repeats. It produced 39,714 paired continuation steps and 516,282 atomic
component comparisons. Integration, q, qvel, controller, EEF, terminal object pose,
and success discrepancies were all exactly zero.

The reused EXP2 validator initially returned a non-passing status only because it
hard-coded the old nine-demonstration cohort. The reconciler changed only that
obsolete count criterion, required the exact EXP4 cohort, and hashed the immutable
source metrics and Parquet files. No numerical gate was relaxed.

### 5.3 Matched-zero gate

Run `exp4_e4_11_zero_20260814` evaluated two independent zero continuations at every
branch, storing 13,238 reference-suffix steps. Integration median, P95, and maximum
were all **0.0**; terminal object-pose P95 was **0.0**. This gate completed in 253.8
seconds and authorized the formal intervention sweep.

## 6. Outcome-blind freeze and primary metric

The scaffold was committed as `33e7075c14ebc29de04ae13e301c7aa5dd0fd8cd` before
formal outcome generation. Fourteen manifests and their hash index were then frozen
against that SHA under
`experiments/exp4_replicated_progress_criticality/manifests`.

The primary branch score was frozen as the RMS across seven sign-paired basis
responses. Each response is the remaining-horizon mean of the same six separately
normalized physical channels used by EXP3:

1. Panda arm-q L2;
2. Panda arm-qvel L2;
3. EEF position L2;
4. EEF SO(3) geodesic angle;
5. task-object position L2;
6. mean task-object orientation geodesic angle.

The finite-difference operator uses a signed normalized physical-output vector. Raw
MuJoCo integration state remains a diagnostic and is not part of that physical
operator. Contact, predicate divergence, and success flips remain secondary outcomes.

## 7. Full execution

Run `exp4_e4_12_full_20260814` completed in **2,037.38 seconds** (33.96 minutes):

| Gate | Result |
|---|---:|
| Expected branches | 252/252 |
| Interventions | 4,032/4,032 |
| Interventions per branch | 16/16 |
| Per-step rows | 211,808 |
| Joint-limit-valid | 4,032/4,032 |
| Finite outcomes | 4,032/4,032 |
| Maximum non-arm boundary mutation | 0 |
| Removed branches | 0 |
| Terminal success flips | 417 |

The raw output hashes were locked before analysis:

| Artifact | SHA-256 |
|---|---|
| `interventions.parquet` | `21bf8fe958303236c0321078f8a41a89eb3776e0678c999728e2c961f1458eb3` |
| `per_step_effects.parquet` | `6016ae5bb73570948505a2f533f160e20955dd4985d2d057c81ceeff4cebd6c7` |
| `zero_controls.parquet` | `8344175d2e2f96532087ce81958f922e24e69356120b3a374f6b82dcc9b59a3c` |

## 8. Primary concentration result

The branch-level `S_RMS` distribution was:

| Statistic | Value |
|---|---:|
| Minimum | 0.00576 |
| P25 | 0.01561 |
| Median | 0.02317 |
| P75 | 0.04160 |
| P95 | 0.08562 |
| Maximum | 0.16868 |

The held-out demonstration-median top-20 mass was **0.45067**, with hierarchical
95% CI **[0.43599, 0.55598]**, safely above the uniform 0.25 null. This independently
replicates non-uniformity. Its magnitude is essentially identical to EXP3's 0.4493,
so the pilot concentration magnitude replicated. It does not pass the stronger 0.50 target,
and leave-one-demo-out robustness was marginal: the minimum was **0.44913**.

Task medians were:

| Task | Median top-20 mass |
|---|---:|
| Drawer | 0.44758 |
| Bowl on plate | 0.51133 |
| Stove | 0.40880 |

Leave-one-task-out estimates were 0.46341 without drawer, 0.44727 without bowl, and
0.50942 without stove. The result is therefore not driven by only one task, although
its magnitude varies materially by task.

## 9. Progress alignment result

All three tasks had positive within-task cross-demo rank agreement under both
coordinate systems:

| Task | Normalized-time median rho | Progress median rho | Difference |
|---|---:|---:|---:|
| Drawer | 0.68485 | 0.75195 | +0.06710 |
| Bowl on plate | 0.79394 | 0.86623 | +0.07229 |
| Stove | 0.58788 | 0.71738 | +0.12950 |

All three tasks exceeded the preregistered task-level progress-rho threshold of 0.50,
so the rule requiring at least two tasks passed. Drawer, bowl, and stove nevertheless
differed in concentration, alignment gain, operator spectrum, and success-flip rate;
the positive threshold count is not evidence of one universal curve.

The paired demonstration-level median improvement was only **0.04372**, with 95% CI
crossing zero and one-sided permutation p = **0.25494**. Thus physical progress
retained high absolute agreement but did not improve enough over time to pass the
frozen `>= 0.15` alignment rule.

Additional agreement measures did not tell a universal improvement story. Progress
ICC exceeded time ICC for bowl (0.2476 vs 0.0247) and stove (0.4524 vs 0.1138), but
decreased for drawer (0.1286 vs 0.1725). Mean top-20 Jaccard improved for stove
(0.4127 vs 0.3333), but decreased for drawer and bowl.

## 10. Direction resolution and local operator

Only **5/21 demonstrations** had direction-rank robustness at least 0.50:

| Task | Stable demos | Median direction robustness | Median random-direction agreement |
|---|---:|---:|---:|
| Drawer | 3/7 | 0.30070 | 0.37762 |
| Bowl on plate | 1/7 | 0.24476 | 0.43357 |
| Stove | 1/7 | 0.11189 | 0.46154 |

The median basis-direction coefficient of variation was **0.44392**. Sign asymmetry
had median **0.18476**, P95 **0.75264**, and maximum **0.93184**, showing that a
single finite-radius central response is frequently far from perfectly symmetric.

The local signed operator was strongly anisotropic:

- spectral norm median 23.61, P95 142.70, maximum 382.35;
- leading-eigenvalue share median 0.7549;
- task-median leading-eigenvalue shares: drawer 0.8447, bowl 0.6652, stove 0.7736.

This suggests locally dominant directions, but those directions do not reproduce
reliably enough across demonstrations to define a global low-dimensional q trigger.
The complete basis therefore did not materially improve the frozen robustness rate:
23.8% passed in EXP4 versus 22.2% (2/9) in EXP3. A scalar branch criticality remains
useful for concentration summaries, but it is not an adequate description of the
local causal response geometry.

The balanced variance decomposition assigned 87.44% to residual/interactions, 5.99%
to progress position, 4.76% to demonstration within task, 1.60% to task, 0.17% to
direction, and 0.03% to sign. The small marginal direction share does not contradict
the poor direction-rank replication: most variability lies in interactions among
state, direction, demonstration, and continuation dynamics.

## 11. Physical channels and terminal relevance

Median intervention-level normalized effects were:

| Channel | Median | P95 | Maximum |
|---|---:|---:|---:|
| arm q | 0.00265 | 0.01342 | 0.05376 |
| arm qvel | 0.03945 | 0.26731 | 1.55576 |
| EEF position | 0.01138 | 0.05230 | 0.21814 |
| EEF orientation | 0.00853 | 0.03169 | 0.13633 |
| task-object position | 0.00200 | 0.28444 | 3.72396 |
| task-object orientation | 0.00219 | 0.05039 | 0.50588 |

Arm velocity dominates the typical response. Object-position effects have a low
median but a very heavy tail. There were 417 success flips: 183 drawer, 213 bowl,
and 21 stove. Nevertheless, primary criticality correlated only weakly with terminal
object displacement (rho **0.1533**) and predicate divergence (rho **0.2965**).
Therefore success flips are important outcomes but do not validate the scalar
criticality ranking by themselves.

## 12. Event analysis

No frozen event survived adjusted testing:

| Event | Median enrichment | Adjusted coefficient | p | BH q |
|---|---:|---:|---:|---:|
| Contact-count change | 1.0761 | +0.1385 | 0.0617 | 0.1852 |
| Gripper-sign change | 0.7477 | -0.3911 | 1.0000 | 1.0000 |
| Predicate first true | 1.2601 | -0.0780 | 0.7646 | 1.0000 |

Contact, gripper, and predicate landmarks therefore remain unsuitable as causal
control triggers under the present evidence.

## 13. GPU analysis and equivalence audit

The simulator remained on its validated deterministic CPU path. Formal numerical
analysis used the available RTX 4090 with PyTorch `1.11.0+cu113`, CUDA 11.3, float64,
4,000 hierarchical bootstrap resamples, and 4,000 cluster permutations. GPU
operations included basis aggregation, Gram/eigenspectrum calculation, progress
interpolation, bootstrap, and rank calibration. No fallback was used.

Two development failures were preserved rather than hidden:

1. PyTorch 1.11 does not support the newer stable-argsort API; tie-free rank logic
   replaced that unsupported call without changing the frozen statistic.
2. CPU/GPU bootstrap equivalence then correctly failed because `torch.median` uses
   a lower middle element for an even sample count whereas NumPy averages the two
   middle values. The GPU implementation was corrected to the explicit arithmetic
   median; the failed audit remains in
   `gpu_cpu_equivalence_failed_bootstrap_median.json`.

The final equivalence gate passed without loosening tolerance: aggregation error
`3.47e-18`, concentration error `2.22e-16`, Gram error `2.66e-15`, singular-value
error `1.82e-14`, bootstrap error zero, interpolation error zero, and exact rank and
interpolation-source identities.

The optional pooled EXP3+EXP4 analysis was not used for the formal conclusion. This
keeps the independently held-out EXP4 result separate from the legacy development
cohort and avoids increasing apparent precision after inspecting the new outcome.

## 14. QA, tests, and artifact audit

The final environment audit reported PyTorch `1.11.0+cu113`, an NVIDIA GeForce RTX
4090 with 24,564 MiB, and driver 596.49. `pip check` reported no broken requirements.
The final automated suite passed **42/42 tests**.

Tests cover all 252 basis regenerations, all 2,016 direction records and joint-limit
audits, all 21 progress/milestone records, GPU equivalence and no-fallback rules, raw
hash mutation detection, concentration edge cases, and protocol schema invariants.

The independent output auditor passed every criterion: exact counts, 14 frozen
manifests, 18 required plots, 252 valid operator rows, no analysis fallback, matching
raw hashes, and an empty failure-example list. The plots were also visually inspected
for readable labels, legends, and axes.

## 15. Failures and corrections retained in the record

| Stage | Failure | Resolution and scientific effect |
|---|---|---|
| Runtime audit 1 | wrapper did not expose object sites | addressed inner env; no outcomes existed |
| Runtime audit 2 | bowl/plate lacked assumed sites | used exact active collision-geom AABBs; no outcomes existed |
| Condition-D summary | obsolete nine-demo count | reconciled count only; all immutable numerical errors remained zero |
| GPU analysis 1 | unsupported stable argsort in torch 1.11 | tie-free exact ranking; no formal result produced before fix |
| GPU equivalence 2 | different even-sample median conventions | explicit arithmetic median; tolerance unchanged; failed audit preserved |
| Protocol test | test depended on JSON key order | corrected test to address named milestones; data unchanged |
| Parquet summary | ICC/Jaccard columns omitted due first-row schema inference | added explicit null columns before writing and reran analysis; raw intervention files unchanged |

None of these corrections selected branches, directions, progress features, or
thresholds using q-intervention outcomes. Raw intervention hashes remained unchanged.

## 16. Reproduction map

Principal executable files:

- `scripts/exp4/audit_runtime_progress.py`
- `scripts/exp4/generate_heldout_references.py`
- `scripts/exp4/build_branch_candidates.py`
- `scripts/exp4/reconcile_condition_d_gate.py`
- `scripts/exp4/freeze_protocol.py`
- `scripts/exp4/run_criticality.py`
- `scripts/exp4/analyze_criticality.py`
- `scripts/exp4/audit_outputs.py`

Principal evidence directories:

- `runs/exp4_e4_0_1_runtime_audit_geom_20260814`
- `runs/exp4_e4_2_condition_d_reconciled_20260814`
- `runs/exp4_e4_3_heldout_refs_20260814`
- `runs/exp4_e4_11_zero_20260814`
- `runs/exp4_e4_12_full_20260814`

The exact formal commands are recorded in each run directory. The final raw and
derived data, statistical decision, GPU audit, CPU/GPU equivalence report, artifact
audit, and all 18 plots are under
`runs/exp4_e4_12_full_20260814/artifacts`.

## 17. Claim boundary and decision

The frozen strong rule required all seven conditions. Only two passed: the bootstrap
lower bound exceeded the uniform null, and at least two tasks had progress-aligned
median rho above 0.50. The remaining five failed.

Allowed claims:

- temporal non-uniformity independently replicated in the held-out 21-demo cohort;
- local q response is anisotropic and often sign-asymmetric;
- the audited progress coordinate retains substantial task-wise rank agreement.

Disallowed claims:

- universal or task-general sparse decision times;
- progress alignment causally resolves temporal misalignment;
- a single joint-space direction or low-dimensional global action latent is stable;
- contact, gripper, or predicate events are validated triggers;
- EXP4 demonstrates improved sample efficiency or task success for sparse RL.

The next experiment should test state-conditioned local response geometry and
finite-radius linearity before any sparse-decision policy is trained.
