# EXP_R5: leave-one-demo-out robustness

EXP_R5 refits the EXP_R4 scalar and factorized baselines ten times, holding
out one entire demo across all three tasks per fold. Feature normalization,
labels, and model fitting exclude the held-out demo. The retrieval-only score
is label-free. All folds retain zero-step and unsafe candidates.
