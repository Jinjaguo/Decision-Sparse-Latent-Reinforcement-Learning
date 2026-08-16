# EXP_R3: instrumented factorized-action candidate matrix

EXP_R3 measures action consequences only after freezing an admissible,
pre-action candidate interface. It reuses the frozen EXP27 branch manifest
and route set, restores each corrected-D branch, writes the candidate row
before execution, then appends outcome data in separate tables.

The valid formal run is:

`runs/exp_r3_s1_instrumented_candidate_matrix_20260816_r4`

It contains 60 branches, 9 routes, 540 candidates, 540 pre-outcome rows, 60
matched-zero controls, and one atomic checkpoint per branch. The pre-outcome
table has 540 unique write-once hashes and intentionally contains no
`predicate_after_execution` or other post-action outcome field. Zero-step
successes are retained with `executed=false` and
`predicate_already_true=true`.

The earlier `..._r2` and `..._r3` formal runs are retained as invalid audit
evidence: `..._r2` leaked `predicate_after_execution` into the selector table,
and `..._r3` wrote the `executed` indicator after the route. They must not be
used for model fitting or scientific conclusions.

Runtime safety includes a per-run Numba cache, per-process LIBERO config on
resume, atomic JSON checkpoints, and progress events. The simulator is kept
single-process because parallel MuJoCo environments would weaken the exact
matched-state determinism gate; offline EXP_R4 analysis may use parallel CPU
or GPU computation.
