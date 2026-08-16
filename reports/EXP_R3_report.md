# EXP_R3 report: instrumented candidate consequences

## Decision

The corrected EXP_R3 interface is valid for offline baseline comparison.
The valid formal run is
`runs/exp_r3_s1_instrumented_candidate_matrix_20260816_r4`.

## Result

- 60 corrected-D branches across the three development tasks.
- 9 frozen routes: one physical-chunk control and V0--V7 consequence routes.
- 540 candidates and 540 pre-outcome records, with one-to-one branch/route
  coverage.
- 60 matched-zero controls; maximum twin integration and EEF-position
  differences were both 0.0, with no predicate divergence.
- 540 unique pre-outcome row hashes.
- 180 candidates were already successful at the restored state and were
  retained as explicit zero-step rows; 360 rows contained an explicit action.
- All recorded states were finite. Unsafe outcomes were retained rather than
  filtered.

The pre-outcome table contains state, contact/force-at-restore, candidate
action, retrieval provenance, and immutable row hash only. It does not contain
`predicate_after_execution`; success, safety, physical progress, and force
response remain outcome labels in the separate summary/step tables.

## Code audit and invalid run

Two implementation defects were found and fixed during execution:

1. The first R3 implementation allowed Numba to use an inaccessible default
   cache path, causing multi-minute import stalls and CPU time comparable to
   the earlier 31,543-second anomaly. R3 now allocates a run-local Numba
   cache. Resume also allocates a fresh LIBERO config directory because the
   source bootstrap is intentionally write-once.
2. The first formal rerun (`..._r2`) copied the post-action predicate into the
   pre-outcome row. The next rerun (`..._r3`) removed that field but still
   mutated the `executed` indicator after the route. Both runs are preserved
   as invalid evidence. The final `..._r4` run was regenerated from scratch;
   it writes both fields required by the selector interface before `env.step`
   and passes schema/timing assertions.

No infinite loop was found. Rollouts have finite route limits, branch-level
atomic checkpoints, and progress events. The simulator remains single-process
for exact same-state determinism; GPU/multi-core execution is reserved for
the offline EXP_R4 baselines.

## Resource audit

After EXP_R3 completion, C: was checked once: 1,180.41 GB free of 1,862.02
GB, above the required 700 GB threshold. E: was not inspected.

## Next experiment

EXP_R4 will use only the corrected formal R3 artifacts. It will compare
pre-registered retrieval-only, scalar-consequence, and factorized-action
baselines using demo-level splits, tie-aware metrics, and leakage assertions.
No untouched confirmation set will be opened.
