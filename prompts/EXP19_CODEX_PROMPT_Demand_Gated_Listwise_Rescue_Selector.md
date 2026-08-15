# EXP19 Codex Prompt — Demand-Gated Listwise Rescue Selector

## Mission

Retain EXP17 candidates and fix EXP18's selector failure. Separate the decision into two predictions: whether the frozen default is likely to fail, and which alternative route is most likely to safely recover that specific branch. Use complete-demo cross-fitting and only pre-action inputs.

## Routes

1. branch-vector kNN that retrieves whole candidate outcome vectors from similar training branches;
2. k=1/3/5/7/11 and physical/object/contact/progress distance views;
3. exact branch-rank versus continuous state distance;
4. default-failure classifier plus rescue-route ranker;
5. pairwise `route rescues while default fails` prediction;
6. listwise softmax and top-two margin;
7. route complementarity/diversity regularization;
8. abstention to default under low demand probability or low rescue margin;
9. compact MLP and ridge interaction baselines;
10. fixed-k9, EXP18 context-kNN, random and oracle controls.

Freeze primary as branch-vector k=3, demand threshold 0.4, rescue-margin abstention 0.05. Report threshold sensitivity but do not select the primary after outcomes.

## Success rule

Unchanged: +10 safe-success points over default; >=60% demand recovery; >=75% oracle headroom capture; safety-stop no worse than default; at least two tasks non-inferior; no leakage.

If EXP19 passes, complete leakage/output/bootstrap audit and declare the action-consequence coordination structure successful in the development scope. If not, EXP20 adds a learned candidate policy value model or prospectively larger training cohort. Continue until success or EXP62.

