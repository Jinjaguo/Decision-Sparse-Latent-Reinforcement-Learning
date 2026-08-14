# EXP7: Contact-mode-conditioned local response

EXP7 tests whether a reproducible short-horizon local q-response operator exists after conditioning on an audited unilateral contact mode and distance-to-boundary margin.

The formal independent cohort contains 30 successful corrected-D same-runtime references (10 per task), 360 frozen branches, three radii, seven fitted directions plus one held-out direction, and both signs: 17,280 interventions total.

Final classification: `within_mode_nonsmoothness_persists`.

This classification coexists with strong one-step state-local results: H1 and H3 passed, H2 and H4 failed, and the separate H5 mode-preservation predictor passed its frozen readiness rule. The failure is cross-demonstration reuse: exact discrete mode plus the coarse margin class did not make the one-step operator transferable across demonstrations.

See [EXP7.md](EXP7.md), the frozen files in `manifests/`, and `reports/exp7_report.md`.
