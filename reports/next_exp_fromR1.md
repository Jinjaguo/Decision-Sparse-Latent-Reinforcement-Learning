# Next experiment after EXP_R1: EXP_R2 matched internal action-selection baselines

## Decision

EXP_R2 should compare retrieval-only, scalar verifier/value, and factorized consequence selection on exactly the same 60 branch states and seven candidates per state, while preserving tie-aware evaluation and post-action leakage controls.

Do not start a router. EXP_R1 validated the benchmark substrate but did not establish that factorization has independent value.

## Why this is necessary

EXP_R1 found complete candidate coverage for all 60 states and safe headroom on all 20 default-demand states, so the next bottleneck is candidate ranking rather than candidate availability. It also found an 0.8397 tie-pair fraction under the conservative safe-success label. A forced total ordering would create artificial signal; EXP_R2 must use partial-order/tie-aware labels.

## Pre-outcome protocol

Before reading any EXP_R2 formal outcomes, freeze:

- the 60 branch groups and exact seven-candidate matrix;
- demonstration-level train/calibration/test split, with no row-level split;
- admissible current-state and candidate-action fields;
- forbidden post-action fields;
- label key `(safe_success, not_safety_stop, success)` and tie policy;
- model capacity and optimization budget;
- cluster-bootstrap seed and resampling unit;
- write-once prediction artifact and hashes.

The EXP_R1 data are already consumed EXP27 formal data. They can be used for development and matched baseline construction, but not called independent confirmation.

## Baselines

### Retrieval-only

Use the same state feature and candidate action database, but do not use successor/post-action consequence fields as a decision signal. Because the current EXP27 artifact is a route-level continuation matrix rather than a clean pre-action candidate feature table, first audit whether admissible pre-action state/action features vary within each branch. If they do not identify candidates, report the degeneracy rather than injecting post-action information.

### Scalar verifier/value

Fit a small matched scalar model to predict the frozen safe-first utility or a calibrated probability of safe success. Use the same admissible inputs, candidate rows, encoder capacity, seeds, and training budget as the factorized model. Test at least a continuous scalar, binary safe-success, and pairwise scalar ranker within one coherent experiment.

### Factorized consequence

Use only repository-supported, measurable consequence fields. Start with explicit factor heads for goal, response, smoothness/action discontinuity, and force/contact risk only where the source artifacts actually provide the field. If a field is post-action only, it may be a training target but not an online input. Compare shared-encoder multi-head and independent-head variants only if the data support them.

## Required negative controls and analyses

- route identity only;
- action-norm-only and constant/default fallback;
- shuffled consequence targets within demonstration;
- task-ID-only diagnostic;
- leave-one-task-out or leave-one-demo-cluster-out analysis where support allows;
- tie-aware pairwise accuracy;
- top-1 with tie credit;
- regret, catastrophic selection, safe-success selection, and oracle-headroom capture;
- demo-cluster bootstrap intervals;
- calibration and abstention/fallback behavior.

If retrieval-only matches factorized selection, narrow the claim. If scalar matches factorized selection, factorization has no demonstrated ranking advantage and the paper should pivot toward specialization or a simpler verifier. If factorized gains appear only in force/contact strata, preserve that narrower claim rather than claiming universal superiority.

## Success and falsification

Support requires a consistent factorized advantage over both matched baselines on at least one primary ranking/safety metric, replicated across demos or failure regimes, without leakage and without destroying nominal/default selections.

Falsification includes:

- no candidate-identifying signal in admissible pre-action fields;
- scalar and factorized models tied after capacity matching;
- shuffled consequence controls performing similarly;
- gains driven by one demo/task or by post-action leakage;
- forced ranking changing only ties and not recovery utility.

If the admissible features are degenerate, EXP_R2 should become a data-instrumentation experiment rather than a model-tuning experiment. Do not proceed to EXP_R3 router work until this is resolved.

## Next implementation route

1. Add a frozen split/manifest builder for the 60 groups.
2. Add a candidate feature audit that explicitly separates pre-action fields from post-action labels.
3. Implement tie-aware metrics and cluster bootstrap.
4. Implement matched retrieval-only, scalar, and factorized baselines.
5. Audit outputs and write `reports/EXP_R2_report.md` before considering router work.
