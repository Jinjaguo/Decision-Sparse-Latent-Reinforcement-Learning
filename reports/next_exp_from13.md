# Next experiment from EXP13

## Current evidence

EXP13 compared eight candidate-source families rather than a single parameter change. Calibration executed 589 candidates; formal evaluation executed 308 authorized Stove candidates and counted all 84 cross-task groups. Formal opportunity was 12/84 overall: Drawer 0/28, Bowl 0/28, Stove 12/28. Candidate execution itself was healthy: validity 100%, terminal failure 2.92%, catastrophic-contact proxy 0.97%, and exact zero replay passed.

## Remaining bottleneck

The remaining gap is representational. Local perturbations, temporal edits, residual bases, retrieval, progress directions, predictor guidance, gripper timing, and restricted compositions can accelerate Stove, but they do not encode the contact acquisition and object-goal geometry needed by Drawer and Bowl. More amplitudes or a larger ranking network cannot create those missing actions.

## Next concrete module

EXP14 changes the proposal interface from local action-space edits to object-centric task-space planning. It compares large phase jumps, direct EEF waypoint control, object-frame bases, cross-demo geometric retargeting, inverse response models, task-semantic finite-state skills, predictor-guided low-dimensional search, and restricted compositions. EXP13 Stove candidates remain controls.

## Decision rule

Keep the EXP13 success rule unchanged: at least 30% opportunity over all 84 groups, at least 20% in two tasks, and at least 90% fidelity-valid candidates with bounded failure/contact tails. Missing-task groups stay in the denominator. If EXP14 passes, train the coordinator next; if it does not, move to sequential closed-loop skill rollouts or a new policy/library source rather than stopping.
