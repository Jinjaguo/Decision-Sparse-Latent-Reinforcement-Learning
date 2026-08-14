# EXP2 — Full-State Matched-Twin Simulator Reconciliation

This directory is the experiment-local execution record for the protocol in
`prompts/EXP2_CODEX_PROMPT_Simulator_Reconciliation.md`.

The experiment addresses the failed EXP1 replay prerequisite. It audits the exact
MuJoCo 3.2.3 / robosuite 1.4.0 runtime, implements explicit simulator and
controller/robot snapshots, generates same-runtime local references for the frozen
3-task × 3-demonstration pilot, and tests zero-perturbation matched twins before any
q intervention is permitted.

The stage order and hard stops are R0 runtime audit, R1 snapshot round trips, R2
local-reference gate, R3 frozen branch times, R4 restoration Conditions A–D, and R5
only if the pre-registered zero-twin gate passes. The source prompt is controlling
if this summary and the full protocol differ.

