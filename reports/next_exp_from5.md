# Next Experiment from EXP5: Confirmatory Reference Reconciliation

**Proposed experiment ID:** EXP6

**Status:** proposal only; do not execute automatically

**Source result:** EXP5 stopped at the confirmatory-reference hard gate

## 1. Why EXP6 is necessary

EXP5 did not fail its state-conditioning hypothesis. It never reached that test.
Drawer `demo_17` exists in the dataset and historically ends in success, but the
current same-runtime replay ends with the middle drawer at `q=-0.021819` instead of
the historical `q=-0.158130`; the exact success threshold is `q < -0.14`.

Substituting another demo, dropping the trajectory, or pooling EXP4 demos would turn
an a priori 30-demo confirmation into a post-hoc selected cohort. EXP6 should resolve
the substrate first.

## 2. Primary question

Why does the historical drawer `demo_17` action sequence fail to produce a successful
same-runtime local reference while the other 29 requested EXP5 trajectories pass?

EXP6 must distinguish:

1. deterministic historical/current-runtime divergence;
2. an omitted initialization, controller, robot, gripper, or timing field before
   local reference capture;
3. model/XML or asset reconstruction differences;
4. contact-sensitive numerical bifurcation;
5. a genuinely unusable public demonstration for same-runtime causal branching.

## 3. Scope

Primary diagnostic target:

```text
task: open_the_middle_drawer_of_the_cabinet
episode: demo_17
```

Controls:

- successful neighboring drawer demos 16 and 18;
- one short/easy successful drawer demo from 10–19;
- the already passing stove and bowl demo17 references as task controls.

Do not run any q perturbation, state matching, operator estimation, or policy training.

## 4. Repetition and determinism

Run the exact drawer demo17 local-reference construction at least five times in
isolated serial processes. Freeze process environment and seed handling. Compare:

- full integration states;
- qpos/qvel;
- controller fields;
- `PandaGripper.current_action`;
- EEF pose/velocity;
- drawer joint coordinate/velocity;
- named contact pairs;
- reward and exact Open predicate.

If repetitions disagree, diagnose nondeterministic initialization or process-global
state before considering a cohort redesign.

## 5. First-divergence localization

Compare the local reference with the historical HDF5 trajectory at every action:

- total state L2 and component-wise qpos/qvel;
- audited drawer joint error;
- EEF-to-middle-drawer/handle geometry;
- gripper command/opening/current action;
- controller targets and robot buffers where historical evidence exists;
- contact-pair symmetric differences;
- action/state indexing and final action semantics.

Save the first thresholds crossed at state L2 `1e-6`, `1e-4`, `1e-2`, and `1e-1`,
plus the first contact-set and drawer-motion divergence. Do not report only the final
error.

## 6. Source and model audit

Revalidate for demos 16–18:

- exact dataset file SHA-256 and HDF5 attributes;
- embedded `model_file` hashes before and after explicit path rewriting;
- BDDL task source and runtime object IDs;
- actuator, joint, geom, and body counts/names;
- timestep, solver, integrator, contact parameters, and option fields;
- robosuite/LIBERO/MuJoCo revisions and Windows compatibility patch;
- initial state restoration under legacy, full-physics, and integration specs.

Any discovered difference must be tested by a controlled A/B run and preserved as a
new immutable run rather than patched into prior evidence.

## 7. Repair decision tree

### Outcome A: an omitted deterministic state field is found

Add it to the explicit reference-construction schema, rerun demos 10–19, and require
30/30 success plus exact round trips. Only then restart EXP5 from descriptor
development with a new preregistration SHA.

### Outcome B: current-runtime replay is deterministic but historically divergent

Record exact demos 10–19 as unsuitable for the original confirmatory design. Define
a new experiment—not a continuation of the failed EXP5 result—with an outcome-blind
eligibility rule applied to all candidates before q outcomes.

A defensible replacement rule is:

```text
scan demos 10–49 in increasing index order;
require same-runtime success and exact snapshot round trips;
select the first 10 eligible demos per task;
report eligibility rates and every rejection;
freeze the resulting cohort before descriptors or q outcomes.
```

This changes the target population toward replayable demonstrations and must be
reported as selection conditioning. It must not be presented as the original EXP5
cohort.

### Outcome C: failure is nondeterministic

Stop the q-criticality mainline and reconcile simulator determinism. Neither branch
selection nor repeated-until-success reference construction is allowed.

### Outcome D: exact cohort cannot be repaired and selection bias is unacceptable

Re-collect new successful demonstrations in the current runtime with complete
corrected-D snapshots captured online. This is scientifically cleaner than silently
screening historical demonstrations, although more expensive.

## 8. Hard gates

EXP6 passes only if one of these prospectively declared endpoints is reached:

1. exact demo17 same-runtime success reproduced in all five isolated repeats, with a
   documented deterministic repair that also preserves all 29 previous successes;
2. the historical replay limitation is proven irreducible enough to justify a new
   selection-conditioned or newly collected cohort protocol.

Do not call “demo17 happened to pass once” a repair. Do not change the drawer success
predicate or use the historical final reward as a substitute for runtime success.

## 9. Required outputs

- repeated-reference metrics and complete per-step divergence tables;
- component/contact/controller first-divergence tables;
- model/XML/source audit and hashes;
- controlled repair A/B comparisons;
- failure examples and plots;
- cohort-bias analysis for any replacement rule;
- a clear restart/no-restart decision for state-conditioned criticality;
- `reports/exp6_report.md` and `reports/next_exp_from6.md`.

## 10. Claim boundary

EXP6 is a simulator/data reconciliation experiment. Even a successful repair does
not support state-conditioned criticality, sparse control, a sensitive q subspace,
or latent RL. It only restores eligibility to run the causal confirmation that EXP5
could not legally execute.
