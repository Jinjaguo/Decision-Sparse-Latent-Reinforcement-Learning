# EXP12 — Offline Counterfactual Action-Consequence Ranking

## Question

Given multiple action chunks executed from the same corrected-D physical state,
can small consequence predictors rank the preferable candidate without predicting
the complete future world?

## Frozen scope

EXP12 is a retrospective, availability-limited ranking study over the immutable
EXP11 formal replacement cohort. It does not collect simulator data and does not
run online control. The experimental unit is a complete demonstration; candidates
from a held-out demonstration never enter model fitting or calibration.

The primary cohort includes only task×family cells that passed the independent
EXP11 execution-fidelity re-audit. Bowl I-A remains in an intent-to-replace
diagnostic but is excluded from primary ranking. The exact successful reference
continuation is inserted as the nominal candidate for every same-state group.

## Ranking axes

- R1: terminal predicate success.
- R2: preservation of the successful reference's coarse contact evolution.
- R3: task-native terminal motion progress using the frozen EXP4 semantics.
- R4: a diagnostic lexicographic-compatible composite that gives terminal success
  priority, then contact preservation and motion progress.

R2 is a reference-preservation safety proxy, not a universal statement that every
contact change is harmful. R4 is never used as a test-time input.

## Model families

Compact physical, object-centric, contact-aware, and short-history encodings are
compared. Consequence-only coordinators include single-specialist, pairwise and
multi-specialist combinations, hard filtering, fixed weighted, learned small,
lexicographic, and uncertainty-abstaining selectors. A fuller 20-step
EEF/object/contact/predicate future predictor is the richer-future baseline.

## Gates

The module is promising only if a consequence coordinator improves same-state
ranking over nominal, deterministic random, and the strongest single specialist,
with demo-level evidence and no unacceptable catastrophic-selection tail. All axes
are reported separately; no all-or-nothing label is permitted.

