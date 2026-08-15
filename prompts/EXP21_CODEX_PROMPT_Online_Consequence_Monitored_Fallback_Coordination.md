# EXP21 Codex Prompt — Online Consequence-Monitored Fallback Coordination

## Mission

Replace one-shot full-policy selection with receding-horizon consequence coordination. Execute a candidate for a frozen dwell, observe success/contact/progress/safety consequences, and switch to a complementary candidate when expected progress is not realized. Target future actions remain hidden.

## Routes

Compare fixed default and fixed weighted-k9 controls; default→k9, k9→smooth-low, smooth-low→k9, median→k9, default→smooth-low, three-stage default→k9→smooth-low, and k9→medoid. Test dwell 50/70/90, two versus three stages, route order, progress-stall versus pure timeout switching, safety-triggered fallback, and maximum total horizon 160/200/240.

The frozen primary is default for at most 70 steps followed by weighted-k9 for at most 130 additional steps if exact success has not occurred. The default baseline remains capped at the EXP17 140 steps. All routes use EXP16 expert-relative safety.

## Success rule

On formal same-state executions, primary coordination must improve safe success by >=10 points over the 140-step default, recover >=60% default-demand groups, capture >=75% oracle headroom relative to the eight EXP17 candidates, not worsen safety-stop rate, and be non-inferior in at least two tasks. No target future leakage.

If passed, perform final audit and declare the structure successful in development scope. If not, EXP22 uses learned progress-stall models and richer initial-chunk descriptors. Continue until success or EXP62.

