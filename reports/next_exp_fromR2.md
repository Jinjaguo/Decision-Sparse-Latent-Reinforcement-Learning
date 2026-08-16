# Next experiment after EXP_R2: EXP_R3 pre-action instrumentation and matched candidate substrate

## Decision

EXP_R3 should instrument a new same-state candidate dataset before any selector model is trained. It must make the decision-time interface explicit and remove the zero-step / route-level ambiguity found in EXP_R2.

## Scientific question

Can every candidate in an exactly restored branch be represented by the same admissible pre-action state, an explicit candidate action/chunk and an immutable retrieval provenance record, while all measured action consequences remain labels only?

## Required new data

For each branch and candidate, write before executing the candidate:

- corrected-D restore hash and simulator/controller snapshot identifiers;
- pre-action EEF/object/contact/current state fields;
- current task-progress fields available online;
- candidate route/expert ID;
- candidate action chunk before force guard and after force guard;
- retrieval row IDs, retrieved distances, aggregation and smoothing parameters;
- candidate action norm and clipping intent;
- a write-once pre-outcome row hash.

After execution, append separately:

- success and safety labels;
- physical progress response;
- force/contact response;
- terminal outcome and continuation length;
- matched-zero replay audit fields.

Zero-step success must still write its pre-action candidate rows with `executed=false` and explicit `predicate_already_true=true`. Do not drop these rows.

## Controls

- exact same-state restore twice before any candidate outcome is inspected;
- matched-zero continuation for every branch or an equivalent audited determinism gate;
- no target-future action access;
- target-demo exclusion from retrieval neighbors and scaling;
- candidate failure/unsafe retention;
- raw branch IDs and restore hashes;
- no post-action field in the pre-outcome selector table;
- route identity, state-only, and candidate-action-only negative controls.

## Design and stopping rule

Use a new immutable EXP_R3 calibration/formal data run. Freeze the candidate plan and pre-outcome hashes before any candidate rollout. Use the same three-task development scope first; do not consume the future untouched confirmation set.

If the pre-action fields are still identical or retrieval provenance cannot be reproduced, stop model development and report another instrumentation failure. If the interface is valid, EXP_R4 should run the pre-registered matched retrieval-only, scalar and factorized baselines with demo-level splits and tie-aware metrics.

## What must not happen

- Do not use post-action force or physical progress as online selector input.
- Do not fit a scalar or factorized model just because the route matrix has outcomes.
- Do not call route priors retrieval-only.
- Do not delete zero-step or unsafe candidates.
- Do not open untouched independent confirmation before the offline comparison and expert specialization stages are complete.
