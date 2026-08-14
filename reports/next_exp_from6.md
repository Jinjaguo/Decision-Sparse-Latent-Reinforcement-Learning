# Next Experiment from EXP6: Hybrid Contact-Mode-Conditioned Response Fields

**Proposed experiment ID:** EXP7

**Proposed title:** Contact-Mode-Conditioned Local Response and Boundary-Margin Validation

**Status:** proposal only; do not execute automatically

**Source classification:** `contact_mode_conditioned_convergence`

## 1. Why this is the next experiment

EXP6 ruled out numerical under-resolution but did not validate a universal local q
operator:

- 0/30 demos met the small-radius top-1 criterion;
- only 37.08% of branches met the small-radius spectral criterion;
- median held-out vector error was 0.7049 at the smallest radius;
- only 7/240 branches had any resolved trust region;
- median antithetic asymmetry remained near 0.9 at every radius.

The one positive explanatory result was exact contact-mode conditioning. Primary
convergence failure was 100% under mode divergence and 92% under mode preservation;
the demonstration-clustered +7.78 percentage-point difference had 95% CI
[2.22, 13.89] points and BH q=0.01525.

Therefore the next question is not “can we use contact as a trigger?” It is:

> After explicitly representing unilateral contact modes and distance to their
> boundaries, is the within-mode short-horizon q response more convergent and more
> reproducible than the unconditional remaining-horizon response?

This remains on the project's causal decision-sparsity mainline. It changes the
response model from one smooth field to a preregistered hybrid field, but does not
train a scheduler, latent policy, or RL controller.

## 2. Literature check on 2026-08-14

The proposed pivot is consistent with recent primary work while remaining narrower
than those control systems:

1. Suh, Pang, Zhao, and Tedrake, *Dexterous Contact-Rich Manipulation via the
   Contact Trust Region* (2025), argue that ordinary ellipsoidal Taylor trust
   regions are inconsistent with unilateral contact and introduce a contact-aware
   trust region. This is the closest conceptual precedent for EXP7, but EXP7 should
   validate an estimand before attempting MPC:
   <https://arxiv.org/abs/2505.02291>.
2. *Smoothly Differentiable and Efficiently Vectorizable Contact Manifold
   Generation* (2026) targets smooth differentiable contact-manifold construction:
   <https://arxiv.org/abs/2602.20304>. It is an alternative diagnostic backend if
   the exact MuJoCo hybrid analysis cannot identify stable within-mode fields; it
   must not be mixed into the primary corrected-D estimand.
3. *Robust Rigid Body Assembly via Contact-Implicit Optimal Control with Exact
   Second-Order Derivatives* (2026) uses smoothed collision and contact resolution
   to obtain derivative information: <https://arxiv.org/abs/2601.22849>. This
   supports a future smoothing-homotopy comparison, not a post-hoc repair of EXP6.
4. ARCH (CoRL 2025) explicitly uses hierarchical hybrid learning for long-horizon
   contact-rich assembly: <https://proceedings.mlr.press/v305/sun25b.html>. Its
   hierarchy is relevant only after EXP7 establishes that the proposed contact
   modes are predictable and carry reusable response information.

Recent action-chunk work such as PACE (<https://arxiv.org/abs/2606.00537>) offers a
possible later alternative if the q-operator mainline closes. It defines a new
causal action-chunk execution-horizon estimand and must not be presented as a direct
continuation of the current q-Jacobian claim.

## 3. EXP7 scope and exclusions

EXP7 should estimate a **piecewise, contact-mode-conditioned response field**. It
must not:

- learn a scheduler or event trigger;
- train latent RL, sparse RL, MPC, or a VLA policy;
- select branches using EXP6 operator norms, trust passes, or success flips;
- delete branches whose perturbations change mode;
- call a finite-difference operator a Jacobian unless its within-mode convergence
  gate passes;
- change simulator/contact physics inside the primary experiment.

## 4. Independent cohort

Use a new same-runtime corrected-D cohort rather than recycling EXP6 outcomes as
confirmatory evidence:

- 10 newly qualified successful trajectories per task;
- 30 demonstrations total;
- preserve all failed qualification attempts;
- freeze ordered replacement rules before collection;
- require exact corrected-D round trips and successful terminal replay.

If the public dataset cannot supply ten new successful demonstrations for a task,
declare the cohort shortfall and run a labeled replication cohort that reuses the
EXP5/EXP6 demonstrations. Do not silently call reused data independent.

## 5. Reference-only mode schema

At every unperturbed boundary preserve:

- exact named geom-pair set;
- target–gripper pair set;
- target–environment pair set;
- gripper–environment pair set;
- exact task predicate;
- gripper command/state;
- signed geom distance or separation margin for preregistered task-relevant pairs;
- normal relative velocity for those pairs;
- normalized time and physical progress.

Define a contact mode as a tuple of exact preregistered pair-group occupancies, not
as raw contact count. Rare exact identities may be pooled only by a frozen physical
group mapping created from reference trajectories.

## 6. Boundary-margin calibration

Before formal outcomes, calibrate whether signed contact separation is stable and
repeatable under corrected-D restoration. Freeze three reference-only margins:

```text
interior mode:       |gap| >= m_far
near boundary:       m_near <= |gap| < m_far
boundary ambiguous:  |gap| < m_near
```

Derive `m_near` from repeated-zero geometric precision and `m_far` from a fixed
physical threshold, not from q-response convergence. Record exact values and units.

## 7. Branch design

Select 12 branches per demo (360 total) using only reference features, balanced as
far as the task permits over:

- contact/no-contact;
- interior/near/ambiguous boundary margin;
- early/middle/late progress;
- negative/neutral/positive gripper command;
- predicate false/true phase.

Use deterministic nearest-unused replacements. Preserve empty strata and report
them rather than changing quotas after seeing responses.

## 8. Perturbation design

Use the three smallest EXP6-validated radii:

```text
0.0003125 x joint range
0.0006250 x joint range
0.0012500 x joint range
```

At each branch/radius retain seven orthonormal plus one held-out direction and both
signs. Reuse the exact EXP6 output vector, but compute separate operators over:

```text
one policy step
first 3 steps
first 5 steps
remaining horizon (secondary historical comparison)
```

The key change is horizon stratification. A mode-preserved one-step response is a
cleaner hybrid-system local estimand than a remaining-horizon response that may
cross several later mode boundaries.

Formal budget:

```text
360 branches x 3 radii x 8 directions x 2 signs
= 17,280 interventions
```

plus at least two matched-zero continuations per branch.

## 9. Mode outcomes must remain outcomes

For every plus/minus pair classify, without deletion:

1. both signs preserve the reference mode;
2. both signs enter the same new mode;
3. signs enter different modes;
4. only one sign changes mode.

Fit/aggregate within-mode operators only in a cross-fitted analysis whose inclusion
rule is frozen. Report the unconditional intent-to-perturb result on all branches
alongside the conditional result so conditioning cannot hide failure cases.

## 10. Primary hypotheses

### H1 — Within-mode one-step convergence

Among branches where both signs preserve the reference mode for one step, at least
70% of demonstrations must have median `0.0003125–0.000625`:

```text
top-1 similarity >= 0.80
top-2 similarity >= 0.75
spectral discrepancy <= 0.20
sign asymmetry <= 0.25
```

The hierarchical 95% CI lower bound for demo-median top-1 must exceed 0.65.

### H2 — Boundary-margin mechanism

Convergence must be higher in interior-mode branches than near/ambiguous boundary
branches with a demo-clustered CI excluding zero and BH q<0.05.

### H3 — Mode-conditioned held-out prediction

At the smallest radius and one-step horizon:

```text
demo-median rank rho >= 0.65
median signed-vector relative error <= 0.35
```

### H4 — Cross-demonstration reuse

Within the same frozen contact-mode group and boundary-margin stratum, matched
top-1 subspace similarity must exceed the time/progress baseline by at least 0.15,
with hierarchical CI lower bound above zero.

### H5 — Contact-mode predictability

A reference-only, cross-fitted mode-transition predictor must reach preregistered
calibration and discrimination thresholds on held-out demonstrations. Prediction
is evaluated separately from response convergence and is not used for control.

## 11. Decision rule

Freeze this priority before outcomes:

1. `within_mode_short_horizon_operator_converges`
2. `boundary_margin_explains_hybrid_nonsmoothness`
3. `contact_modes_explanatory_but_not_predictable`
4. `within_mode_nonsmoothness_persists`
5. `contact_schema_not_identifiable`
6. `no_support`

The first classification requires H1, H3, and H4. H2 alone cannot authorize a
controller. H5 must pass independently before any later contact-aware scheduler.

## 12. Controls and hard gates

Retain all EXP6 gates:

- exact corrected-D matched zero;
- q injection tolerance <=1e-15;
- non-arm integration Linf <=1e-12;
- exact joint limits;
- finite arrays;
- complete directions/signs/radii;
- raw lock before analysis;
- CPU source of truth unless a newly frozen scale-aware GPU equivalence protocol
  passes before formal analysis.

For GPU, preregister both absolute and relative error criteria appropriate to the
radius-scaled Gram magnitude. This is a new audit design, not a reinterpretation of
the failed EXP6 absolute gate.

## 13. Required ablations

Compare:

- exact named mode versus contact count;
- target–gripper mode versus all contact groups;
- one-step versus 3-step versus 5-step versus remaining horizon;
- interior versus near-boundary versus ambiguous;
- mode-conditioned versus normalized time and task progress;
- exact MuJoCo hybrid response versus a separately labeled smooth-contact
  diagnostic on a small frozen subset, if technically feasible.

The smooth-contact diagnostic must never replace or pool with corrected-D outcomes.

## 14. Interpretation and fallback

If within-mode short-horizon convergence passes, EXP8 may test state/mode
replication inside the validated trust region. Only after both convergence and
contact-mode prediction pass may an oracle contact-aware scheduler be considered.

If boundary margin explains failure but mode prediction fails, continue measurement
work; do not train control.

If within-mode one-step response still fails despite precise mode and margin
conditioning, close the universal q-response/Jacobian scheduler mainline. A later
project may study causal action-chunk intervention or phase-aware replanning, but it
must declare a new estimand, new baselines, and new claim boundary.

Latent RL remains ineligible throughout EXP7.
