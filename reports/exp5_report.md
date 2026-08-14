# EXP5 Report: Confirmatory Reference Gate Failure

**Experiment:** EXP5 — Cross-Fitted State-Conditioned Anisotropic q Criticality

**Execution date:** 2026-08-14

**Status:** **Stopped at the confirmatory-reference hard gate**

**Formal classification:** Not legally available; no confirmatory q outcome exists

## 1. Executive conclusion

EXP5 did not proceed to state-conditioned matching, matched-zero controls, or the
planned 16,896 q interventions. The frozen protocol required every new demonstration
10–19 in all three tasks to produce a successful same-runtime local reference. One
required trajectory, `open_the_middle_drawer_of_the_cabinet/demo_17`, ended with
task failure. The eligible cohort was therefore 29/30 overall and only 9/10 for the
drawer task.

The failure is substantive, not a missing file, NaN, snapshot serialization error,
or joint-identifier ambiguity. All 30 dataset episodes existed. The failed local
reference contained 195 finite boundaries; integration and controller snapshot
round trips were exactly zero. However, the local continuation moved the audited
middle-drawer joint only to `-0.021819`, far short of the exact open threshold
`q < -0.14`. The historical dataset trajectory reaches `-0.158130` and records
reward/done at its final step.

The protocol explicitly states:

```text
If fewer than 10 demos qualify for any task: STOP.
Do not silently substitute another demo.
Do not reuse EXP4 demos.
Do not replace the task.
```

That stop rule was followed. No confirmatory intervention outcome was generated or
inspected, no threshold was changed, and no formal EXP5 classification was selected.

## 2. Intended scientific question

EXP5 was designed to test whether the stable causal object behind EXP3/EXP4 is a
state-conditioned scalar sensitivity field, a reproducible local sensitive
subspace, a finite-radius nonlinear response geometry, or a trajectory-specific
effect. It proposed:

- reference-only physical state descriptors;
- deterministic five-fold cross-fitting;
- 16 state-coverage branches per confirmatory demo;
- seven orthonormal basis directions plus one held-out random direction;
- radii 0.0025 and 0.005 at all 480 branches;
- radius 0.01 on a frozen 20% subset;
- 16,896 planned interventions after exact corrected-D and zero gates.

None of these q-outcome questions can be answered because the prerequisite cohort
does not exist under the specified same-runtime reference rule.

## 3. Required-input audit

The complete EXP5 prompt, `PROJECT.md`, EXP1–EXP3 experiment records, EXP1–EXP4
reports, `next_exp_from4.md`, the evidence table, the latest research log, and the
passing EXP2–EXP4 snapshot/intervention/analysis implementations were inspected.

The requested `experiments/exp4_replicated_progress_criticality/EXP4.md` file is not
present in the repository. Its controlling prompt, fourteen frozen manifests,
complete report, code, run evidence, and raw-data manifest were present and used.
No missing runtime identifier was guessed from that absent summary file.

The working tree already contained user changes to `README.md`, `reports/README.md`,
and `prompts/`; these were preserved and excluded from EXP5 commits.

## 4. Baseline environment

| Item | Result |
|---|---|
| Conda environment | `libero-exp1` |
| Python | 3.8.20 |
| PyTorch | 1.11.0+cu113 |
| CUDA available | yes |
| GPU | NVIDIA GeForce RTX 4090 |
| Dataset episodes | 50 per selected task |
| `pip check` | no broken requirements |
| Corrected baseline tests | 47 passed |

The first baseline test invocation produced 40 passes and two fixture setup errors
because the explicitly selected `.tmp` parent did not exist. After creating that
task-local parent, the unchanged tests passed 47/47. This was a test-harness path
issue, not a scientific failure.

## 5. EXP5-0 availability audit

Run: `exp5_s0_availability_20260814` — **PASS**.

The HDF5 audit checked every development episode 3–9 and confirmatory episode 10–19
for all three frozen tasks:

| Gate | Result |
|---|---:|
| Development episodes | 21/21 |
| Confirmatory episodes | 30/30 |
| Files present | 51/51 |
| `states`/`actions` length agreement | 51/51 |
| Valid episode length | 51/51 |
| Episode model XML present | 51/51 |

Thus the stop was not caused by missing demonstrations.

## 6. Reference-generation execution

### 6.1 Failed sandbox attempt

Run `exp5_s4_confirmatory_refs_20260814` failed before importing the simulator
because robosuite 1.4.0 attempted to open its fixed `C:\tmp\robosuite.log` path and
the sandbox denied access. No trajectory was simulated.

### 6.2 Interrupted monolithic attempt

Run `exp5_s4_confirmatory_refs_corrected_20260814` began after filesystem approval
but exceeded the outer execution-session duration while processing the 30-reference
monolithic job. It stopped after writing partial drawer artifacts and never emitted a
metrics/gate file. Those partial artifacts were not reused.

### 6.3 Immutable reference shards

The job was divided into five non-overwriting shards, each covering two exact episode
indices across all three tasks:

| Run | Episodes | References | Gate |
|---|---|---:|---|
| `exp5_s4_refs_10_11_20260814` | 10–11 | 6 | pass |
| `exp5_s4_refs_12_13_20260814` | 12–13 | 6 | pass |
| `exp5_s4_refs_14_15_20260814` | 14–15 | 6 | pass |
| `exp5_s4_refs_16_17_20260814` | 16–17 | 6 | **fail** |
| `exp5_s4_refs_18_19_20260814` | 18–19 | 6 | pass |

All five shards completed and retained their exact commands, environment, Git state,
stdout/stderr, metrics, reference arrays, boundaries, controller snapshots, and
failure examples.

## 7. Confirmatory-reference results

| Task | Successful references | Required | Action-count range |
|---|---:|---:|---:|
| Drawer | **9/10** | 10 | 126–195 |
| Stove | 10/10 | 10 | 77–103 |
| Bowl on plate | 10/10 | 10 | 84–112 |
| **Total** | **29/30** | **30** | — |

Across all generated references:

- every saved snapshot array was finite;
- maximum integration-state round-trip error was 0;
- maximum controller round-trip error was 0;
- only drawer `demo_17` failed final task success.

The four passing shards are valid negative/engineering evidence, but they cannot be
combined into a reduced confirmatory cohort because the sample design forbids
post-availability deletion.

## 8. Drawer demo_17 failure diagnosis

The failed reference is located under
`runs/exp5_s4_refs_16_17_20260814/artifacts/references/`
`open_the_middle_drawer_of_the_cabinet/demo_17`.

| Diagnostic | Same-runtime local continuation | Historical HDF5 record |
|---|---:|---:|
| Action count | 195 | 195 |
| Minimum audited drawer q | -0.021819 | -0.158130 |
| Final audited drawer q | -0.021819 | -0.158130 |
| Open threshold | `< -0.14` | `< -0.14` |
| Nonzero reward count | 0 | 1 |
| Final reward/done | 0 / false | 1 / true |
| Snapshot/controller round-trip max | 0 / 0 | not applicable |

The public legacy replay diagnostic for this episode had median state L2 0.04695,
P95 2.25925, and maximum 5.27784. The failure is consistent with the known historical
open-loop replay portability problem: restored local snapshots are exact once
captured, but replaying the historical action sequence in the current runtime does
not necessarily reproduce the historical state trajectory.

This interpretation is evidence-supported but not yet a complete causal diagnosis.
The first divergence time, contact transition, controller history difference, and
whether the failure repeats in isolated serial runs remain to be established.

## 9. Code implemented before the stop

The following non-outcome scaffold was added:

- EXP5 experiment/config directory;
- generic episode-list support in the audited EXP4 reference generator;
- generic multi-radius support in the shared continuation runner;
- confirmatory HDF5 audit and reference wrapper;
- EXP5 zero/intervention entry points;
- state-matching/operator math helpers for robust scaling, shrinkage covariance,
  Mahalanobis costs, constrained monotone matching, deterministic prototype
  selection, central operators, projectors, subspace similarity, and BH FDR;
- exact synthetic unit tests for those mathematical helpers.

These modules were tested, but no EXP5 state descriptor, matcher, prototype set,
branch set, basis, radius set, linearity threshold, or statistical decision rule was
formally frozen. The scaffold must not be mistaken for completed preregistration.

## 10. Stages not executed

Because the confirmatory reference gate failed, the following were not run:

- formal development descriptor and baseline output generation;
- confirmatory five-fold fitting and matching;
- 480-branch prototype coverage freeze;
- corrected-D regression on those branches;
- GPU equivalence on the final descriptor/matcher/operator implementation;
- dry q intervention;
- full matched-zero gate;
- radii 0.0025, 0.005, or 0.01 interventions;
- operator assembly, linearity, subspace, random-direction, terminal, or event
  inference;
- formal EXP5 classification.

Interventions completed by radius are therefore:

| Radius | Planned | Completed |
|---|---:|---:|
| 0.0025 | 7,680 | 0 |
| 0.0050 | 7,680 | 0 |
| 0.0100 calibration | 1,536 | 0 |

## 11. Protocol deviations

The availability audit was correctly executed before references. However, the first
confirmatory-reference attempt occurred before completing the planned development
descriptor/matching outputs and their freeze. This deviated from the exact stated
order. It did not leak q outcomes—none existed—and cannot make the failed reference
gate pass. Work stopped immediately once the completed reference shards established
the gate failure.

The monolithic reference attempt was also too large for the outer execution-session
limit. Splitting by predeclared episode ranges changed only execution packaging, not
the cohort, task, actions, snapshot schema, or success gate.

## 12. GPU status

The RTX 4090 was detected and available, but no formal EXP5 GPU workload was legally
eligible after the reference gate failed. Consequently there is no EXP5 CPU/GPU
equivalence result and no GPU formal-analysis claim. MuJoCo reference stepping used
the validated CPU path.

## 13. Answers required by the EXP5 protocol

| Question | Answer |
|---|---|
| Did demos 10–19 exist? | Yes, 30/30. |
| Did all pass same-runtime validation? | No; drawer/demo_17 failed final success. |
| Confirmatory demos completed? | 30 generated, 29 successful. |
| Branches/interventions completed? | 0 formal branches; 0 q interventions. |
| Did corrected Condition D remain exact? | Not tested on an EXP5 branch set because none could be frozen. Immediate saved-snapshot round trips were zero. |
| Did state matching improve scalar replication? | Not legally testable. |
| Did cross-fitting replicate? | Not legally testable. |
| Did top-1/top-2 subspaces replicate? | Not legally testable. |
| Was 0.0025 more stable than 0.005? | Not legally testable. |
| Fraction passing linearity gate? | Not defined; no frozen gate/outcomes. |
| What happened at 0.01? | No calibration interventions were run. |
| Did held-out direction prediction work? | Not tested. |
| Was GPU used formally? | No; only detected. |
| Did CPU/GPU equivalence pass? | Not run. |
| Formal classification? | None; stopped prerequisite. |
| Is an oracle adaptive scheduler eligible? | No. |
| Is latent RL eligible? | No. |

## 14. Claim impact

EXP5 neither supports nor contradicts state-conditioned anisotropic criticality. It
failed before measuring that estimand.

Strongest allowed new statement:

> Under the exact EXP5 cohort rule, 29 of 30 requested demonstrations generated
> successful same-runtime references, but drawer demo_17 did not; therefore the
> confirmatory state-conditioned q-criticality experiment was not executable without
> violating its frozen no-substitution rule.

Claims still forbidden include universal sparse times, state-conditioned
replication, stable sensitive subspaces, finite-radius linearity, held-out direction
prediction, oracle scheduler eligibility, and latent-RL benefit.

## 15. Highest-value next action

Run a dedicated reference-reconciliation experiment on drawer `demo_17`, with no q
interventions. It should locate the first historical/local divergence, test isolated
repeatability, audit contact/controller/gripper evolution, and decide prospectively
whether the exact cohort can be repaired or whether a new, explicitly selection-
audited cohort protocol is required.
