# EXP6 implementation

Execution order is fixed: reference-only branch selection, calibration manifest freeze, repeated zero/intervention calibration, resolution decision, formal manifest freeze, GPU/CPU equivalence, full matched-zero, full radius sweep, immutable raw hash lock, then GPU analysis and reporting.

Run directories are immutable under `runs/`; calibration manifests and formal manifests are separate so optional-radius admission cannot depend on formal outcomes.
