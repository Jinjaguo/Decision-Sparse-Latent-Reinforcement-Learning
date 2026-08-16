# EXP_R1 — Exact Expert Reconstruction and Same-State Counterfactual Benchmark

EXP_R1 is the first infrastructure and validity experiment of the post-EXP27
campaign. It reconstructs the exact implementation meaning of the EXP27
mechanisms and converts the immutable EXP27 formal route matrix into an explicit
same-state candidate-set benchmark.

The benchmark contains 60 restored branch states and seven candidates per state.
It retains all candidates, including failed and unsafe continuations. No selector
is trained in EXP_R1. Post-action outcomes are labels only and are explicitly
marked as forbidden selector inputs.

The source is the consumed EXP27 formal run. Therefore this experiment validates
the benchmark substrate and causal bookkeeping; it is not an independent
confirmation set and does not support a new closed-loop generalization claim.
