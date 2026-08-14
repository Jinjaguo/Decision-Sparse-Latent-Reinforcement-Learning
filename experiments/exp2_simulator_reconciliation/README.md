# EXP2 simulator reconciliation

- `EXP2.md` records the experiment-local objective and protocol source.
- `configs/zero_twin_gate.json` freezes the formal restoration thresholds.
- `manifests/` contains runtime-derived schemas and frozen branch choices.
- Formal outputs are immutable directories under `runs/`.

No perturbation result is admissible unless the R2 local-reference gate and R4
zero-twin gate both pass.

## Final status

EXP2 completed on 2026-08-14.

- Final local references: `exp2_r2_gripper_refs_20260814T011336` — passed, 9/9
  successful with exact immediate snapshot round trips.
- Initial formal Conditions A–D: `exp2_r4_formal_20260814T002437` — all failed;
  retained as negative diagnostic evidence.
- Diagnostic Condition E: `exp2_r4_condition_e_20260814T010207` — failed and
  localized the remaining omission to Python-side runtime state.
- Corrected Condition D: `exp2_r4_condition_d_gripper_20260814T011457` — passed all
  preregistered criteria with exactly zero error across 324 pairs and 17,121 paired
  continuation steps.
- R5 smoke: `exp2_r5_q_smoke_20260814T012633` — passed; 48/48 q interventions were
  measurable above an exact zero matched-twin floor.

The selected state is MuJoCo `mjSTATE_INTEGRATION` plus explicit controller, robot,
environment-timing, and `PandaGripper.current_action` state. See
[`../../reports/exp2_report.md`](../../reports/exp2_report.md) for the complete
analysis and [`../../reports/next_exp_from2.md`](../../reports/next_exp_from2.md) for
the proposed full time-indexed criticality experiment.
