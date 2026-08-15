# EXP10 — Phase-Conditioned Macro-Action Trajectory Response

EXP10 tests six retrospective routes on immutable EXP8/EXP9 data: multi-scale
macro actions, phase-conditioned experts, compact trajectory latents, direct
terminal consequences, teacher-free coarse-regime sequences, and probabilistic
world-action trajectory models. Routes authorize independently. New simulator
collection is forbidden unless at least one route passes its frozen Stage-A gate.

The completed result is `no_macro_scale_structure`: no route authorized Stage B.
This experiment does not train a scheduler, controller, MPC system, VLA, or RL
agent.
