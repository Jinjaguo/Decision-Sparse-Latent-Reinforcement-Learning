# EXP23 Codex Prompt — Risk-Aware Consequence Arbitration with New Confirmation

## Mission

EXP22 achieved 82.14% safe success and exactly 60% demand recovery, missing the 75% headroom gate by two safe branches. Drawer reached 78.57%, Bowl 100%, while Stove retained eight force stops. EXP23 must recover complementary branches while preventing repeated high-force contact. It must not tune on EXP22 formal demos 43–49 and call the same cohort confirmation.

## Cohorts and leakage boundary

Use demos 30–40 plus 41–42 for expanded calibration and mechanism selection. The prospectively named demos 50–56 were audited before candidate execution and found outside the dataset's available index range. Replace them, before inspecting any EXP23 candidate outcomes, with the ordered common available cohort demos 21–27. These 21 trajectories already pass the EXP7 same-runtime reference gate, have no overlap with the demos 30–40 action library, and have not supplied EXP15–22 recovery outcomes. Select four reference-only landmarks per demo. Target-demo future actions and expert suffixes are evaluation-only. Exclude the target demo from retrieval neighbors and feature scaling whenever it overlaps the independent library.

## Required mechanism portfolio

Implement multiple structural routes:

1. Repeated-force guard: after force approaches the expert-relative envelope, retract against the previous commanded motion, reset retrieval memory, and move to a complementary consequence mode.
2. Soft force scaling: attenuate Cartesian commands before the stop threshold without changing the frozen safety definition.
3. Goal-consequence guarded route: prioritize independent-demo successor states near demonstrated goals, then fall back to smooth and response-aligned actions.
4. Smooth-response guarded route: use low-variance contact approach first, then empirical response alignment and progress-biased retrieval.
5. Current-state phase/risk arbitration: estimate coarse progress only from current state and independent references, then choose a mode order; never use target trajectory length or future action.
6. Task-specific guarded portfolio: retain diverse cycling for Drawer, a short stable route for Bowl, and force-aware goal/smooth/response coordination for Stove.
7. Progress-stall plus force guard and an unguarded phase control, so the contribution of force intervention is identifiable.
8. Analyze guard counts, force peaks, safe success, regressions, route complementarity, task/phase robustness, and whether guards merely avoid stops or actually finish tasks.

Calibration may compare guard fractions, retract gains, scaling factors, mechanism order, progress bands, stall windows, and 280–320 step budgets, but formal thresholds remain unchanged. Freeze the smallest primary within one safe success of the best expanded-calibration route.

## Success rule

On untouched recovery outcomes from demos 21–27, the frozen primary must improve safe success by at least 10 points over the 140-step default, recover at least 60% of default-demand branches, capture at least 75% of candidate-oracle headroom, not worsen safety-stop rate, and be non-inferior in at least two tasks. Use a confirmation-set oracle only for evaluation, report EXP17/EXP22 transfer separately, and audit all source hashes and target-future isolation.

If unsuccessful, write EXP23 evidence and continue to a broad EXP24 experiment. Stop only on success or after EXP62.
