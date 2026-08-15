# Next experiment from EXP14

## Current evidence

EXP14 implemented eight task-space routes and executed 632 calibration candidates. Drawer and Bowl again had zero candidate better than the expert nominal by 0.05; Stove calibrated at 5/8 opportunity groups. Formal execution of 308 authorized Stove candidates produced 14/84 opportunities overall, all Stove (14/28). Candidate fidelity was 100%, so numerical clipping was not the explanation.

## Why the benchmark must change

The target-demo continuation is a successful expert upper bound and is not available at real deployment. In calibration, Drawer nominal quality was exactly 6.0 and the best successful candidate exceeded it by only 0.0019. Bowl nominal was 5.91–5.96 and the best gap was 0.0389. Requiring another candidate to exceed this expert by 0.05 makes the cross-task availability gate structurally near-impossible and rewards Stove overshoot rather than realistic recovery.

## Next module

EXP15 removes the target-demo future from candidate inputs and selectable actions. It remains only an evaluation ceiling. Candidate sources become reference-free feedback macro-policies trained/retrieved from independent demos. Each emits a short action chunk, observes the new state, and replans. The primary comparison is against a frozen non-oracle default, with explicit decision-demand and safe-recovery rates.

## Why this preserves the story

The project remains action-centric consequence coordination: the selector will still compare candidate actions through Contact, Motion, Outcome and Uncertainty. The change creates a deployment-realistic candidate set and a real selection problem instead of asking a policy to outperform an unavailable expert continuation. If recovery candidates pass, the next experiment trains the coordinator; if not, the next experiment strengthens the policy/planner source.

