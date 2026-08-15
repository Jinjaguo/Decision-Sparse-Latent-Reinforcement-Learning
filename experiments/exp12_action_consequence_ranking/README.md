# EXP12 reproducibility map

The controlling prompt is
`prompts/EXP12_CODEX_MASTER_PROMPT_Autonomous_Action_Consequence_Coordinator_to_EXP42.md`.
The user's later instruction extends the autonomous stopping boundary to EXP62;
that discrepancy changes only the outer iteration budget, not EXP12.

Protocol files under `configs/` and `manifests/` are frozen before formal model
scoring. Immutable outputs are written beneath `runs/`.

