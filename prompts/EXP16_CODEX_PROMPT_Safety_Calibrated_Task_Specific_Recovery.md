# EXP16 Codex Prompt — Safety-Calibrated Task-Specific Recovery

## Mission

Turn EXP15's partial reference-free recovery into a cross-task candidate source. Preserve the hidden-target-future rule and feedback macro-policy interface. Fix the uncalibrated force stop, improve Drawer/Stove policy structure, retain Bowl success, and rerun calibration/formal gates. Do not train the final consequence coordinator yet.

## Frozen evidence

- EXP15 calibration: safe-candidate availability 15/24 (62.5%), decision demand 14/24 (58.3%), demand recovery 5/14 (35.7%), valid candidate fraction 93.98%.
- Bowl: all nine routes succeeded on all 8 branches; default demand was zero.
- Drawer: availability 4/8; conservative route was best at 4/8.
- Stove: availability 3/8; most routes hit the absolute 200 N stop before success. The threshold was not calibrated against successful expert force trajectories.
- No route passed the frozen combination of success, clipping and <=10% safety-stop authorization, so formal was not run.

## Stage 0 — Expert-relative safety audit

Before candidate outcomes, replay successful expert upper-bound suffixes from calibration states and record per-step force/torque, contact mode, signed gap and task progress. Freeze task×phase safety envelopes using robust P99/P99.5 plus a documented margin. Safety is a relative exceedance over successful contact-rich behavior, not an arbitrary global force number. Retain an absolute emergency ceiling and compare both rules.

Do not use expert future actions in candidates. The safety audit is an isolated evaluation/calibration path, hashed before candidate execution.

## Routes to implement and compare

### S1 — Safety-envelope variants

Compare absolute-only, task-level expert-relative, task×phase expert-relative, duration-aware exceedance, and force-plus-contact-mode safety stops. Freeze the primary rule on calibration experts before candidate outcomes.

### S2 — Task-specific object/progress features

Use drawer joint/body displacement, bowl-to-plate planar/vertical error, stove button orientation/progress, EEF-object geometry, gripper/contact state and recent progress slope. Compare shared full-state distance with task-specific standardized distances.

### S3 — Progress-monotone retrieval

Retrieve only source states whose task progress is at or slightly ahead of current progress, with a bounded lookahead window. Compare nearest state, weighted-k, diverse trajectory and progress-stratified retrieval.

### S4 — Persistent source-trajectory tracking

Maintain source-demo identity and monotone index, with hysteresis before switching sources. Compare free nearest-neighbor switching, one-source persistence, top-two beam tracking, and dynamic-time-alignment-like local windows.

### S5 — Task-aware conservative ensembles

Retain EXP15's conservative median and add trimmed mean, medoid chunk, risk-weighted vote, and task-specific route ensembles. Preserve exact gripper signs and action bounds.

### S6 — Feedback retargeting with phase guards

Apply EEF/object-frame correction only in compatible phases and only below a calibrated magnitude. Drawer correction must preserve handle contact; Bowl correction must preserve grasp/support phase; Stove correction must allow intended button contact. Compare gains and guarded versus unguarded correction.

### S7 — Action continuity and chunk execution

Add action-rate limits, exponential smoothing, overlap blending and replan hysteresis. Compare per-step replanning, 2/5/10-step execution, and adaptive replan triggered by progress/contact divergence.

### S8 — Recovery-specialist ensemble

Create task-specific stalled-progress, lost-contact and overshoot specialists from training rollouts. Include the frozen EXP15 default and best routes as controls. Compose at most two audited components and cap candidate count.

## Cohorts and leakage

Use EXP8 independent demos for policy/library construction. Calibration is episodes 41–42; formal development is episodes 43–49. Target-demo future actions remain accessible only to isolated expert evaluation and never to policy features, retrieval, fitting, route selection or online updates. Hash code/config/library before candidate outcomes.

## Success rule

Keep the EXP15 recovery gates:

- at least 70% formal safe-candidate availability overall;
- at least 60% availability in at least two tasks;
- at least 30% decision demand against the frozen non-oracle default;
- at least 60% recovery among demand groups;
- at least 90% executable/finite chunks;
- zero target-future leakage.

Additionally, primary safety stops must be justified by the expert envelope, and results must be reported under both the primary relative rule and the frozen 200 N sensitivity rule.

## Required ablations

- S1–S8 separately;
- absolute versus expert-relative safety;
- shared versus task-specific features;
- unconstrained versus progress-monotone retrieval;
- free switching versus persistent/beam source tracking;
- nearest, weighted, median, trimmed, medoid and risk-weighted aggregation;
- retargeting gains and phase guards;
- smoothing/rate limits/replan interval;
- task specialists and compositions;
- candidate count 4/8/12;
- expert-relative quality, default demand and oracle recovery jointly.

## Stop/proceed

If EXP16 passes formal recovery availability, EXP17 trains the action-consequence coordinator and tests actual selection utility. If one task fails, EXP17 imports a stronger policy source or task planner for that task while retaining passed modules. If demand is insufficient, EXP17 creates prospectively frozen policy-noise/state-disturbance demand without exposing expert future actions. Continue automatically; stop only when the complete project succeeds or EXP62 has been completed and the next experiment would be EXP63.

