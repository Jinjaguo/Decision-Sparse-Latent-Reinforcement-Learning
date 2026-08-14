# EXP7 formal record

## Question

Within a fixed audited contact mode, away from a contact boundary, does the corrected-D LIBERO substrate admit a reproducible short-horizon local response operator?

## Frozen design

- Three tasks: middle drawer, bowl on plate, and stove.
- Independent demonstrations: Drawer 21–30; Bowl and Stove 20–29.
- 12 reference-only balanced branches per demonstration, 360 total.
- Radius fractions: 0.0003125, 0.000625, 0.00125 of joint range.
- Seven fitted orthonormal q directions plus one held-out random direction.
- Both signs, for 17,280 interventions.
- Horizons: H1, H3, H5, and remaining.
- Exact mode is the set of audited, named runtime geom pairs in four possible groups.
- Signed gap uses MuJoCo `mj_geomDistance`; no body-center proxy is allowed.
- The primary H1 analysis requires both signs to preserve the matched-zero exact mode.

## Locked raw result

- 17,280 unique interventions; exact frozen-set equality passed.
- 1,094,928 per-step rows.
- 145 terminal success flips.
- Maximum non-arm integration change: 0.
- Maximum q-injection error: 2.220446049250313e-16.
- All formal execution gates passed.

## Scientific result

| Hypothesis | Result |
|---|---|
| H1: preserved-mode H1 convergence | Pass; 30/30 demo medians passed, top-1 hierarchical 95% CI [0.99849, 0.99992] |
| H2: boundary margin explains convergence | Fail; mean interior advantage 0.04169, 95% CI [-0.00856, 0.11765], BH q=0.5 |
| H3: held-out prediction | Pass; demo-median rho 0.98788, vector error 1.64e-6 |
| H4: same mode+margin enables cross-demo reuse | Fail; improvement -0.03736, 95% CI [-0.06554, -0.01040] |
| H5: next-step mode preservation predictability | Pass separately; AUROC 0.89783, demo-cluster CI [0.87495, 0.92000], ECE 0.02880 |

The frozen priority therefore yields `within_mode_nonsmoothness_persists`. In practical terms, the response is smooth and predictable at one fixed simulator state when contact mode is preserved, but exact mode plus a 1 mm margin bin is not a sufficient state abstraction for reusing that operator across demonstrations.

EXP7 does not authorize scheduling, control, MPC, VLA integration, or latent RL.
