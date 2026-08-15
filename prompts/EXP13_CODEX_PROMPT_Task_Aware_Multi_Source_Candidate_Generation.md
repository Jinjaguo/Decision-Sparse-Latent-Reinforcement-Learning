# EXP13 Codex Prompt — Task-Aware Multi-Source Candidate Generation

## Mission

Solve one module: create executable candidate action chunks that provide real oracle improvement opportunities across tasks. EXP12 showed ranking signal but only 10/84 groups had a candidate better than nominal, all on Stove. Do not train a new coordinator in EXP13.

## Frozen evidence

- EXP12 primary cohort: 1,036 candidates, 84 groups, 21 demos.
- Candidate sets: 56×17 and 28×3.
- Opportunity >0.05: Drawer 0/28, Bowl 0/28, Stove 10/28.
- Safe setwise selectors, richer-future, and nominal all had top-1 0.881 and opportunity capture 0.
- Bowl I-A failed the EXP11 task-family clipping gate and is not a valid primary source.

## Primary question

Can a diverse but audited candidate generator make genuinely better actions available often enough for a consequence selector to matter?

## Stage 0 — Source and semantics audit

Read EXP11 action semantics and corrected-D code. Audit exact progress channels:

- Drawer: `wooden_cabinet_1_middle_level`, goal `q < -0.14`.
- Stove: `flat_stove_1_button`, goal `q >= 0.5`.
- Bowl: exact On predicate plus bowl-to-plate planar geometry and vertical/support contact.

Freeze candidate chunks before any target-state outcomes. Candidate generation may use reference trajectories and training demos, never target-demo counterfactual results.

## Candidate families to implement and compare

### G1 — Multi-channel analytic modes

Use DCT, spline and smooth-pulse temporal modes, but do not restrict every branch to the nominal chunk's single highest-variance channel. Select up to three task×phase channels using calibration/reference-only action semantics and training-demo evidence. Test small sign-paired amplitudes that respect OSC bounds.

### G2 — Temporal warp and phase edits

Include index shifts ±1/±2, local speed-up/slow-down, chunk compression/expansion with exact gripper sign handling, and phase-landmark aligned chunks. Do not continuously interpolate gripper commands.

### G3 — Training-only task×phase residual modes

Fit leave-demo-out residual SVD/PLS-like modes from successful training references. Compare rank-1/2 modes, robust normalized modes, and task-specific versus shared modes. Target-demo trajectories may supply the current state and nominal chunk but not basis training.

### G4 — Cross-demo action library

Retrieve chunks from other demos of the same task using compact physical, object-centric and contact-aware context distances. Compare nearest-one, diverse top-k, and phase-constrained retrieval. Every retrieved chunk must be executed from the exact target state; do not assume action-library transfer works.

### G5 — Task-progress-directed residuals

Use calibration-only finite differences to estimate which action-channel/time patterns improve audited task progress while preserving success/contact. Compare ridge direction, sign-constrained direction, and trust-region clipped direction. Freeze directions before formal target outcomes.

### G6 — Predictor-guided local proposals

Use only EXP12 training-fold consequence models to score a finite pool of analytic/residual candidates, then execute the top diverse proposals on held-out demos. No test-demo outcome may enter proposal search. Compare greedy top score, uncertainty-penalized score, and diversity-regularized top-k.

### G7 — Exact gripper timing

Where a real sign transition lies near the branch, test advance/delay by one and two steps. Report support by task. Do not fabricate gripper candidates where no transition exists.

### G8 — Restricted composed candidates

After single-family calibration, compose at most two individually valid low-amplitude modes, such as progress-directed arm residual plus exact gripper timing. Use a small frozen composition budget; do not run a combinatorial search.

## Stage 1 — Calibration authorization

Use EXP11 calibration references (episodes 41–42) only. Execute multiple candidate families from same corrected-D states. For each task×family report:

- requested/executed mismatch on unsaturated channels;
- clipped-chunk fraction;
- all-state finiteness;
- exact matched-zero replay;
- task progress range and terminal success;
- coarse contact divergence and force/wrench validity;
- consequence diversity;
- oracle improvement opportunity versus nominal.

Authorize at most four families per task. A family must pass fidelity (`clipped_chunk_fraction <= 0.10`, unsaturated mismatch P95 below audited tolerance), have non-degenerate consequences, and create at least one calibration improvement opportunity without excessive failures. Selection is calibration-only and hashed before formal execution.

## Stage 2 — Development-formal execution

Use episodes 43–49 as held-demo development evaluation. This is not a new independent paper-confirmation cohort. Every learned basis/retrieval/guided proposal must be leave-demo-out. Use four existing EXP11 branch states per demo so comparisons are matched.

Limit the final set to nominal plus no more than 12 authorized candidates per branch. Preserve all failed/adverse candidates. Keep exact same-state zero controls and raw schemas.

## Primary metrics

- valid candidate fraction;
- candidate-set consequence diversity;
- proportion of groups with oracle improvement >0.05;
- opportunity by task and phase;
- median/P90 oracle improvement gap;
- terminal failure rate;
- catastrophic contact rate;
- clipping and execution fidelity;
- source-family contribution to oracle-best actions.

Do not rank families by macro effect alone.

## Success rule

Candidate generation is solved only if:

- at least 30% of all formal groups have valid oracle improvement >0.05;
- at least two of three tasks individually reach 20%;
- valid candidate fraction is at least 90%;
- terminal failure and catastrophic-contact tails are reported and bounded;
- no single target demo was used to train its own generator.

Partial success is task-specific and must trigger a new EXP for the missing task/source. Failure does not stop the research program.

## Required ablations

- each G1–G8 family separately where supported;
- task-specific versus shared modes;
- context metric for library retrieval;
- guided versus unguided proposal;
- diversity regularization on/off;
- single versus composed candidates;
- candidate count 4/8/12;
- with and without clipped rows in intent-to-propose diagnostics, while primary remains fidelity-valid.

## Outputs

Create `experiments/exp13_candidate_generation/`, immutable runs, tests, manifests, raw locks, plots, `reports/exp13_report.md`, `reports/next_exp_from13.md`, and the next prompt. Required artifacts include source support, calibration authorization, candidate plans, per-step raw, candidate summaries, opportunity tables, family ablations, raw hashes, scientific decision and failure examples.

## Stop/proceed

If the success rule passes, EXP14 retrains and tests the coordinator on the improved candidate sets. If only one task passes, EXP14 targets proposal coverage for the other tasks. If no local family works, EXP14 changes proposal abstraction to task-space targeted chunks or an external policy/library. Do not stop unless the complete project succeeds or EXP62 has been completed.

