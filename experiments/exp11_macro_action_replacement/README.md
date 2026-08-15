# EXP11 — Structured Macro-Action Replacement

EXP11 replaces the earlier one-boundary joint-state impulse with a causal
future-action intervention.  At a corrected-D reference boundary it replaces
the next `K` recorded actions, then resumes the original continuation:

`A_rep[t:t+K] = A_ref[t:t+K] + alpha B_k`.

Stage 0 is reference-only. Stage 1 uses episodes 41–42 for calibration. Stage
2 uses the remaining independent episodes 43–49 and is explicitly an
availability-limited pilot (7 demonstrations per task).  All selection,
family, action-fidelity, model, cross-fitting, conformal, and decision rules
are written and hashed before their corresponding outcomes.

Canonical immutable runs and final conclusions are indexed in
`reports/exp11_report.md`.
