# EXP_R11 report: frozen factorized selector confirmation

## Freeze

The factorized selector was trained once on the corrected R3 development
matrix (demo_0--9, 540 candidates) and frozen before confirmation outcomes.
The model state hash is
`f381828cb910cd902f6f5219310502e1f170e60f1545d087fbd51a9c7c984235`.
The confirmation cohort is the pre-existing EXP26 frozen branch manifest for
episodes 28--29: 24 unique branches, three tasks, and nine routes.

## Confirmation result

| selector | success | safe success | safety-stop |
|---|---:|---:|---:|
| frozen factorized selector | 0.917 | 0.917 | 0.042 |
| fixed D physical control | 0.625 | 0.625 | 0.125 |
| per-branch oracle | 0.958 | 0.958 | 0.000 |

The deterministic top-1 selector had one safety stop among 24 selected
branches; this is reported by the separate summary rate and is not hidden.
The zero-determinism gate passed with 24/24
controls. The confirmation evaluation used the frozen model hash, no
threshold tuning, no target-future action access, and no outcome-derived route
prior.

## Boundaries

The result supports a narrower claim: a frozen pre-action factorized selector
generalizes beyond the ten development demos and improves over the D control
on this 24-branch confirmation cohort. It does not prove universal safety,
because unsafe labels are sparse and the oracle headroom is nonzero. The
factorized advantage is supported by R5/R6 cross-demo and ablation results,
while R7--R9 show that safety calibration and unsafe recall remain the main
limitation.

After EXP_R11 completion, C: was checked once: 1,180.06 GB free of 1,862.02
GB. E: was not inspected.
