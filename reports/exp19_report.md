# EXP19 实验报告：Demand-Gated Listwise Rescue Selector

EXP19 把八路线 outcome 作为 branch vector，先预测默认失败，再预测互补 rescue。冻结主 k=3、demand threshold 0.4、margin 0.05。

主结果 safe success 70.24%，比默认高 11.90 点，safety stop 从 10.71% 降到 7.14%；但需求恢复 37.14%、oracle headroom 34.48%，失败。Drawer/Stove 需求恢复分别 21.43%/47.62%。65 个预冻结 k/threshold/margin/listwise 变体中，最高需求恢复也只有 45.71%。

因此问题不是主阈值选错，而是 selector outcome 训练样本太少、demo-level route complementarity 不稳定。EXP20 在独立 EXP8 demos 上实际运行 frozen candidates，严格 leave-one-demo-out，形成更大的 selector training cohort；formal outcomes 只作最终测试。

协议提交 `7492f62`，run `exp19_s1_demand_gated_selector_20260815`，leakage audit 通过。

