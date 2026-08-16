# EXP_R4: offline factorized-action baselines

This experiment consumes only the corrected EXP_R3 formal artifacts. It
compares a label-free retrieval score, a scalar success-and-safety utility
model, and a factorized model with separate success and unsafe heads.

The fixed demo-level split is demos 0--5 for training, 6--7 for validation,
and 8--9 for test, shared across all three tasks. The script fails if any
post-outcome field appears in the pre-outcome table or if pre-outcome hashes
are not unique. It reports tie-aware top-1/top-3 utility and mean average
precision, with branch bootstrap intervals.

The simulator is not imported. If CUDA is available, only the offline MLP
training uses it.
