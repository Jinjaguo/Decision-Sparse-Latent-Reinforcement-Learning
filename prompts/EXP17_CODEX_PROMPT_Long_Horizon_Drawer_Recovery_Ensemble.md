# EXP17 Codex Prompt — Long-Horizon Drawer Recovery Ensemble

## Mission

Close the remaining EXP16 formal recovery gap without weakening any gate. Preserve the target-future-free interface, expert-relative safety envelopes, passed Bowl/Stove modules, and frozen non-oracle default. Replace the globally short 80-step horizon with a prospectively fixed long recovery horizon and compare multiple Drawer-capable policy structures.

## Frozen evidence

- EXP16 formal: safe availability 78.57%, decision demand 47.62%, valid candidates 100%.
- Bowl availability 100%; Stove 89.29% with 86.36% demand recovery.
- Drawer availability 46.43%; only 3/18 default-demand branches recovered.
- Drawer failed branches are concentrated at early branch times. Their successful target upper bounds have 89–125 steps remaining, while every selectable policy was terminated at 80 steps.
- Later Drawer branches with 40–74 remaining steps are commonly recovered; no Drawer safety stop occurred in the formal route breakdown.

## Routes

Compare all of the following under a target-independent fixed maximum horizon (primary 140 steps, sensitivity 100/120/160):

1. frozen EXP16 physical default;
2. weighted object/contact kNN with k=3/5/9;
3. conservative median and medoid with k=7/9;
4. low/medium/high action smoothing;
5. progress-monotone retrieval with advance 1/2 and source persistence;
6. short-cycle 2/5-step chunk tracking versus per-step feedback;
7. source-trajectory beam of top two persistent demos;
8. Drawer-specific stalled-progress recovery that switches retrieval source only after a frozen no-progress dwell.

Keep exact gripper signs, expert-relative taskwise safety, absolute 1000 N emergency ceiling, and no target future leakage. Use EXP8 demos for every library/model.

## Success rule

Unchanged from EXP16: at least 70% overall safe availability; at least two tasks at 60%; at least 30% default demand; at least 60% demand recovery; at least 90% valid chunks; no leakage. Formal results, not calibration, decide success.

## Ablations

Report horizon 80/100/120/140/160, k, aggregate type, smoothing, replan interval, monotone/persistent source, early versus late branches, task-specific versus shared routes, candidate count 4/8/12, safety envelope versus 200 N sensitivity, and time-to-success tails.

If EXP17 passes, EXP18 trains the action-consequence coordinator on the validated recovery candidate sets. If Drawer still fails, EXP18 imports/trains a dedicated Drawer policy source while preserving Bowl/Stove. Continue automatically until project success or EXP62 completion.

