# EXP14 Codex Prompt — Object-Centric Task-Space Candidate Planning

## Mission

Solve the cross-task candidate-availability bottleneck. EXP13 established that diverse local action-space edits create valid improvement opportunities on Stove, but not on Drawer or Bowl. EXP14 must change the proposal abstraction: express desired object/EFF geometry and task progress first, then convert those intentions into executable OSC action chunks. Do not train a new coordinator yet.

## Frozen evidence

- EXP13 calibration: 589 candidates, 24 groups, 98.64% fidelity-valid; opportunity was Drawer 0/8, Bowl 0/8, Stove 4/8.
- EXP13 formal: 308 candidates over the 28 authorized Stove groups and all 84 groups in the denominator; opportunity was 12/84 overall, Drawer 0/28, Bowl 0/28, Stove 12/28.
- Formal candidate validity was 100%; terminal failure was 2.92%; catastrophic-contact proxy was 0.97%.
- Local action-space diversity is therefore not the main missing ingredient. The missing abstraction is task-directed geometry for contact acquisition, object transport, and progress completion.

## Primary question

Can object-centric and task-space proposal mechanisms produce fidelity-valid action chunks that beat the successful nominal continuation on at least two tasks often enough for consequence coordination to become meaningful?

## Stage 0 — Semantics, geometry, and OSC audit

Freeze and test the exact OSC mapping before outcomes are inspected. Translation inputs in `[-1,1]` map to at most 0.05 m controller deltas and rotation inputs to at most 0.5 rad. Verify realized EEF direction, magnitude, clipping, exact gripper signs, same-state replay, and controller-state restoration.

Use only audited task channels:

- Drawer: middle-drawer joint progress, cabinet/drawer body frame, EEF-to-drawer geometry, grasp/contact state, and the successful pull direction inferred without target-demo counterfactual outcomes.
- Bowl: bowl and plate positions, planar bowl-to-plate error, height/support relation, EEF-to-bowl geometry, grasp/contact state, and exact predicate.
- Stove: button joint progress and button-relative EEF geometry; preserve the successful EXP13 action-space families as controls.

All learned geometry, inverse maps, retrieval indices, or search models must exclude the target demo.

## Proposal routes to implement and compare

### T1 — Large-horizon phase skips and action lookahead

Replace the weak ±1/±2 temporal edits with frozen lookahead offsets 5/10/20, compressed later chunks, phase-landmark jumps, and goal-tail chunks. Preserve exact gripper signs. This tests whether the nominal chunk is simply too conservative locally.

### T2 — Direct EEF waypoint controllers

Construct position/orientation waypoint chunks in task space and map each desired delta through the audited OSC scale. Compare proportional, tapered, and two-stage approach/act profiles. Bound every command before simulation and report desired-versus-realized EEF displacement.

### T3 — Object-frame directional bases

Generate small finite bases aligned with EEF-to-target, target-to-goal, vertical, support-normal, drawer-pull, and button-tangent directions. Use sign-paired and trust-region amplitudes. These are geometric proposals, not arbitrary channel perturbations.

### T4 — Cross-demo geometric retargeting

Retrieve successful EEF/object-relative trajectories from other demos of the same task, transform them into the target branch's object frame, and convert the retargeted waypoints to OSC commands. Compare phase-only, physical-context, and object-geometry retrieval; use diverse top-k and exclude the target demo.

### T5 — Local inverse-response proposals

Fit leave-demo-out regularized maps from short action summaries to realized EEF/object progress, then solve a bounded inverse problem for desired progress. Compare linear ridge, robust weighted least squares, low-rank PLS, and nearest-local inverse models. Reject ill-conditioned solutions rather than silently clipping them.

### T6 — Task-semantic finite-state skills

Implement explicit finite proposal templates, selected from observed reference-side state only:

- Drawer: approach handle, close/hold, pull along audited drawer axis, and settle.
- Bowl: approach bowl, grasp/hold, lift, translate over plate, lower, and release/settle.
- Stove: approach button, press/rotate, and hold, plus EXP13 controls.

Each branch receives multiple applicable templates and conservative variants; do not use target outcomes to choose among them.

### T7 — Predictor-guided low-dimensional search

Define a small coefficient space over T1–T6 primitives. Use training-fold consequence models to score a finite Latin-hypercube/CEM-like pool, uncertainty-penalize it, diversity-filter it, and execute a frozen top-k. Search is offline and finite; no simulator outcome from the target demo may feed back into the search.

### T8 — Restricted hybrid compositions

Compose at most two individually fidelity-valid primitives, such as retargeted approach plus drawer pull, or bowl transport plus exact release timing. Compare single primitives, two-stage skills, and ensemble source coverage under the same candidate budget.

## Stage 1 — Calibration

Use only episodes 41–42 and the corrected-D states. Execute broad T1–T8 candidate pools. For each task×route report:

- OSC desired-versus-realized direction and magnitude;
- clipping, finite-state, zero-replay, and wrench validity;
- exact terminal success and task-native progress;
- contact divergence and catastrophic tails;
- consequence diversity and oracle opportunity over nominal;
- support across phases and demos.

Authorize at most five routes per task. Authorization requires at least one valid opportunity, at least 80% terminal success, at least 90% fidelity-valid candidates, non-degenerate consequence diversity, and no unexplained replay failure. Freeze authorization and candidate-budget rules before formal outcomes.

## Stage 2 — Development-formal evaluation

Use episodes 43–49, four existing branch states per demo, with all learned/retrieved components leave-demo-out. Keep nominal plus at most 16 authorized proposals per branch. Count all 84 groups in the denominator, including a task with no authorized proposal. Preserve adverse candidates and all raw trajectories.

## Primary metrics and success rule

Report valid candidate fraction, opportunity rate overall/by task/by phase, median/P90 oracle gap, task progress, terminal failure, catastrophic contact, clipping, source contribution, and candidate-count sensitivity.

Candidate generation is solved only if all hold:

- at least 30% of all 84 formal groups have a fidelity-valid oracle improvement greater than 0.05;
- at least two of three tasks individually reach 20%;
- at least 90% of proposed candidates are fidelity-valid;
- terminal failure and catastrophic-contact tails are bounded and fully reported;
- no target demo trains, retrieves, or tunes its own generator.

Do not relax these thresholds after seeing results.

## Required ablations

- T1–T8 separately where supported;
- action-space EXP13 controls versus task-space routes;
- position-only versus position+orientation;
- object frame versus world frame;
- phase-only versus physical versus object-context retrieval;
- direct waypoint versus inverse-response versus finite-state skill;
- guided versus unguided and uncertainty penalty on/off;
- single versus composed primitives;
- chunk horizon 10 versus 20;
- candidate budget 4/8/12/16;
- primary fidelity-valid versus intent-to-propose diagnostics.

## Outputs and honesty rules

Create `experiments/exp14_task_space_candidates/`, immutable runs, unit tests, frozen manifests, raw hashes, plots, `reports/exp14_report.md`, `reports/next_exp_from14.md`, and the next prompt. Failed runs remain preserved. Separate protocol failures, implementation failures, and scientific failures. Do not claim task-space planning works from EEF motion alone; it must create task-native oracle improvement while respecting outcome/contact tails.

## Stop/proceed

If the success rule passes, EXP15 trains and tests the action-consequence coordinator on the improved sets. If one task remains missing, EXP15 targets that task with sequential/receding-horizon proposals or an external successful-policy library. If task-space open-loop chunks fail broadly, EXP15 changes to closed-loop skill rollout and adaptive replanning. Continue automatically; stop only when the complete project succeeds or EXP62 has been completed and the next experiment would be EXP63.
