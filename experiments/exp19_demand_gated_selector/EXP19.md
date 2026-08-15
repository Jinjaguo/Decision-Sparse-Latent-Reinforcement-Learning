# EXP19 — Demand-Gated Listwise Rescue Selector

EXP19 retrieves whole eight-route outcome vectors from similar training branches.
It first predicts whether the frozen default needs rescue, then ranks alternatives
by joint safe-success and complementarity to default. The frozen primary uses
k=3, demand probability 0.4, and rescue margin 0.05.

