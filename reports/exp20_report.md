# EXP20 实验报告：Independent Recovery Training Cohort

EXP20 构建 30 demos、120 branches 的独立 selector training cohort。八路线严格 leave-one-demo-out：目标 demo 同时从 neighbor pool 和 feature scale 排除。运行 960 rollouts、60,168 steps，训练 cohort 可用率 90.83%、需求恢复 76.09%，数据模块通过。

冻结主 `task_rank_consequence` 只用这 120 个训练分支，随后一次性测试 EXP17 formal outcomes。safe success 73.81%，默认 58.33%，提高 15.48 点；safety stop 降至 3.57%。但需求恢复仅 40%，oracle headroom capture 44.83%，因此失败。task-route/ridge 同为 73.81%，context-kNN 为 70.24%–71.43%，没有路线接近 60% demand recovery。

该结果说明失败不是 formal crossfit 样本复用，而是一次性从初始 context 预测完整长程 recovery policy 成败的信息不足。EXP21 改成 receding-horizon consequence monitoring：短段执行后根据已观察后果切换互补候选。

协议提交 `c28aa8c`，selector 提交 `ae23b58`；runs 为 `exp20_s1_training_recovery_20260815`、`exp20_s2_training_recovery_analysis_20260815`、`exp20_s3_independent_selector_test_20260815`。

