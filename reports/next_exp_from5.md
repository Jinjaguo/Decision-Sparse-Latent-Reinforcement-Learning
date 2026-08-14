# Next Experiment from EXP5: Radius-Convergence of Local q Response

**Proposed experiment ID:** EXP6

**Proposed title:** Multi-Radius Convergence and Trust-Region Identification of q Response

**Status:** proposal only; do not execute automatically

**Source classification:** `finite_radius_nonlinearity_dominates`

## 1. Why this is the next experiment

EXP5 found anisotropic response operators, but almost no evidence that they were
local linear objects over the tested radii:

- 21/480 branches (4.375%) passed the four-part linearity gate;
- small/main top-1 similarities were 0.174 Drawer, 0.139 Bowl, and 0.317 Stove;
- only 1/30 demos had median cross-radius top-1 similarity >=0.70;
- main/large top-1 similarity was 0.162;
- main/large spectral discrepancy was 0.437;
- held-out response rank prediction was encouraging (rho 0.626), but vector errors
  remained near or above the actual response norm.

State matching cannot solve an estimand that changes substantially with radius.
Before changing the representation, training a scheduler, or starting latent RL,
EXP6 should determine whether a stable local response emerges at smaller radii or
whether contact-rich q response is intrinsically nonsmooth at the accessible scale.

## 2. Primary question

> As Panda arm-q perturbation radius approaches zero above the exact matched-zero
> floor, do response magnitude and physical-q principal subspaces converge?

EXP6 separates:

1. finite-radius curvature;
2. contact-mode discontinuity;
3. numerical under-resolution at very small q offsets;
4. genuinely trajectory-specific directional response.

## 3. Frozen cohort and branch design

Reuse no EXP5 outcome to select “good-looking” branches. Construct a new
outcome-blind subset from reference state strata:

- all three tasks;
- ten qualified EXP5 demos per task, retaining the disclosed Drawer replacement;
- eight reference-only prototypes per demo, 240 branches total;
- stratify prototypes over normalized time, task progress, contact state, gripper
  state, and predicate phase;
- freeze the subset before any new-radius outcome.

The reduction from 16 to eight branches per demo buys four or five radii without
reducing demo coverage. Preserve the full 480-branch EXP5 data only as historical
comparison; do not use its response outcomes to choose the EXP6 subset.

## 4. Radius ladder

Primary proposed ladder:

```text
0.000625 x joint range
0.001250 x joint range
0.002500 x joint range
0.005000 x joint range
```

Optional 0.0003125 may be added only if a pre-outcome numerical calibration shows
its antithetic signal is resolvable above the exact zero floor.

At every branch/radius, retain:

- seven orthonormal basis directions;
- one held-out random direction;
- both signs;
- matched-zero continuations;
- the exact EXP5 signed physical output vector.

Planned primary budget:

```text
240 branches x 4 radii x 8 directions x 2 signs = 15,360 interventions
```

This is slightly smaller than EXP5 while directly answering the failed gate.

## 5. Trust-region estimand

Do not assume one global radius is valid. For each branch, estimate the largest
radius interval satisfying all of:

- adjacent-radius top-1 similarity >=0.80;
- adjacent-radius top-2 similarity >=0.75;
- relative spectral discrepancy <=0.20;
- antithetic sign asymmetry <=0.25;
- held-out vector relative error <=0.35;
- response norm at least 100 times the measured zero-floor upper bound.

Call this interval the **empirical local trust region**. If no adjacent pair passes,
record trust-region radius as unresolved rather than forcing a Jacobian.

## 6. Contact-mode analysis

EXP5 did not preregister radius convergence by contact transition. EXP6 should do so
without turning contact into an outcome-selected event:

- freeze contact/no-contact and gripper-state strata from the unperturbed reference;
- measure whether plus/minus perturbations change named contact mode within the
  first 1, 3, 5, and 10 future steps;
- compare smooth operator convergence within unchanged contact mode against branches
  with contact-mode divergence;
- retain exact contact pairs and predicate trajectories separately from continuous
  response vectors.

This can distinguish classical curvature from hybrid-system mode switching.

## 7. Numerical resolution calibration

Before the full sweep, run a preregistered calibration on two branches per task and
all proposed radii:

- at least four repeated matched-zero continuations;
- repeat each signed intervention twice;
- verify q injection accuracy and non-arm preservation;
- compare effect norm to zero-floor and repeated-intervention variance;
- require deterministic direction/sign rankings at the chosen dtype and runtime.

If 0.000625 is not reliably resolvable, stop and report a lower-bound limitation.
Do not silently drop the radius after observing the formal cohort.

## 8. Primary hypotheses and decision rule

### H1: radius convergence

At least 70% of demonstrations must have median adjacent-radius top-1 similarity
>=0.80 between 0.000625 and 0.00125, with hierarchical 95% CI lower bound >0.65.

### H2: scale convergence

At least 70% of branches must have relative spectral discrepancy <=0.20 over the
same radius pair.

### H3: held-out prediction

Demo-median held-out rank rho >=0.65 and median vector relative error <=0.35.

### H4: contact explanation

The convergence failure rate must be substantially higher when antithetic
perturbations diverge in contact mode, with a demo-clustered confidence interval
excluding zero and BH-FDR <0.05.

Formal classifications, in priority order:

1. `small_radius_local_operator_converges`;
2. `contact_mode_conditioned_convergence`;
3. `numerical_resolution_prevents_local_limit_test`;
4. `nonsmooth_response_persists_below_exp5_radius`;
5. `no_support`.

## 9. Interpretation tree

If `small_radius_local_operator_converges`:

- revisit state matching using only branches/radii inside their empirical trust
  region;
- test an oracle trust-region-aware sensitivity predictor;
- still do not train latent RL directly.

If `contact_mode_conditioned_convergence`:

- model a hybrid response field indexed by audited contact mode;
- use mode-specific subspaces rather than a universal q subspace;
- preregister contact-mode transition prediction before controller use.

If numerical resolution prevents the test:

- improve precision/runtime instrumentation or use a deterministic differentiable
  simulator diagnostic;
- do not infer nonsmoothness from an unresolved signal.

If nonsmoothness persists below EXP5 radii:

- close the q-Jacobian/subspace mainline for universal scheduling;
- define a new action-chunk or latent-action causal estimand;
- require a fresh experiment rather than relabeling finite-radius response as a
  local derivative.

## 10. Required controls and artifacts

Required controls:

- complete corrected-D regression on every EXP6 branch;
- at least two zero twins per branch, four on calibration branches;
- exact q injection and non-arm Linf <=1e-12;
- CPU/GPU float64 equivalence;
- immutable per-radius shards and raw lock before analysis;
- no branch deletion after outcomes.

Required primary artifacts:

- radius-resolved interventions and signed responses;
- repeatability and signal-to-floor tables;
- adjacent-radius operator and projector comparisons;
- branch trust-region estimates;
- contact-mode transition tables;
- held-out direction predictions;
- hierarchical bootstrap/permutation outputs;
- failure examples, raw hashes, GPU audit, decision JSON;
- `reports/exp6_report.md` and `reports/next_exp_from6.md`.

## 11. Claim boundary

EXP6 is a local-estimand validation experiment. Even a pass would establish only
that a measurable, radius-convergent response operator exists in part of the tested
state space. It would not by itself establish sparse decision times, policy benefit,
sample-efficiency improvement, an adaptive scheduler, or latent RL.

The research mainline should remain:

```text
validated simulator substrate
-> radius-convergent causal response
-> state/contact-conditioned replication
-> oracle scheduler or subspace controller
-> learned sparse/latent method only after oracle benefit
```
