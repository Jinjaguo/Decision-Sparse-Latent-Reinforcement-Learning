# EXP6 implementation

Execution order is fixed: reference-only branch selection, calibration manifest freeze, repeated zero/intervention calibration, resolution decision, formal manifest freeze, GPU/CPU equivalence, full matched-zero, full radius sweep, immutable raw hash lock, then GPU analysis and reporting.

Run directories are immutable under `runs/`; calibration manifests and formal manifests are separate so optional-radius admission cannot depend on formal outcomes.

Final source-of-truth runs:

- raw lock: `exp6_s11_formal_raw_locked_20260814`;
- CPU analysis: `exp6_s12_formal_analysis_cpu_20260814`;
- classification: `contact_mode_conditioned_convergence`;
- GPU formal analysis: rejected by the frozen absolute-equivalence gate.
