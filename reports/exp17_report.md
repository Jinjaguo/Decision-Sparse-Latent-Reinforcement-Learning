# EXP17 实验报告：Long-Horizon Drawer Recovery Ensemble

## 1. 问题与变化

EXP16 的唯一主要缺口是早期 Drawer：候选统一 80 步，但成功 expert 尚需 89–125 步。EXP17 在不读取目标 future 的前提下冻结 140-step horizon，并比较 k=3/9 weighted、median/medoid、两档平滑、progress persistence 和短 chunk tracking。

## 2. 协议

安全 envelope、1000 N 应急上限、EXP8 训练库、冻结默认和所有 recovery gates 均继承 EXP16。目标 future 仅作隔离 expert 上界。九条路线先 calibration，最多八条进入 formal。

## 3. 校准

`exp17_s1_calibration_recovery_20260815` 执行 216 rollouts、13,046 步。可用率 91.67%，需求率 37.5%，需求恢复 77.78%，有效率 100%；Drawer/Bowl/Stove 分别 75%/100%/100%。校准 gate 通过。

## 4. 正式执行

`exp17_s3_formal_recovery_20260815` 在 84 分支执行 672 rollouts、37,841 步，八条授权路线，expert success 100%，target-future access false。

## 5. 正式主结果

`exp17_s4_formal_analysis_20260815`：安全候选可用率 92.86%，需求率 41.67%，需求恢复率 82.86%，候选有效率 100%。全部 recovery gates 通过。

## 6. 按任务

Drawer 可用率从 EXP16 的 46.43% 提高到 85.71%，14 个需求组恢复 10 个；Bowl 100%；Stove 92.86%，21 个需求组恢复 19 个。至少两个任务的 60% 门显著通过。

## 7. 路线

整体 safe success：weighted-k9 75%，smooth-low 73.81%，weighted-k3/smooth-high 69.05%，medoid 67.86%，median 64.29%，默认 58.33%。固定 weighted-k9 本身已比默认高 16.67 个百分点，为 coordinator 提供可学习的 route signal。

## 8. 安全与 fidelity

所有候选无裁剪且状态有限。route safety-stop 为 5.95%–11.90%；安全可用性按组仍为 92.86%。没有使用 target future；长 horizon 为所有任务统一的固定 140，而非按目标剩余长度设置。

## 9. 科学解释

长程 feedback 解决了早期 Drawer 的人为截断，同时保留 Bowl/Stove。候选可用性、默认需求和 oracle headroom 现在同时存在，满足训练 action-consequence selector 的必要条件。

## 10. Claim 边界

EXP17 支持 reference-free recovery candidates 在 formal development cohort 上跨任务可用；不支持 selector 已成功、独立确认、部署安全或超出三任务的泛化。

## 11. 模块状态

```yaml
recovery_candidate_availability: formal_passed
drawer_recovery: formal_passed
bowl_recovery: formal_passed
stove_recovery: formal_passed
decision_demand: formal_passed
oracle_recoverability: formal_passed
consequence_selector: not_yet_tested
project_complete: false
```

## 12. 下一实验

EXP18 在 complete-demo cross-fitting 下预测 Contact/Motion/Outcome/Uncertainty，并在执行前选择一个 recovery candidate。与 default、random、固定最佳、heuristic、pairwise/listwise、uncertainty/abstention 和 richer predictor 比较。

## 13. 追溯

- 协议提交 `b1c2cee`。
- calibration `exp17_s1_calibration_recovery_20260815` / analysis `exp17_s2_calibration_analysis_20260815`。
- formal `exp17_s3_formal_recovery_20260815` / analysis `exp17_s4_formal_analysis_20260815`。
- 5/5 helper tests 通过。

