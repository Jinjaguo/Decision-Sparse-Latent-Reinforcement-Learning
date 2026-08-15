# EXP16 实验报告：Safety-Calibrated Task-Specific Recovery

## 1. 实验目标

校准 contact-rich expert 的力安全范围，修复 EXP15 的 Stove 误停止，并通过任务特异检索、持续轨迹、ensemble 和平滑提高 reference-free recovery。

## 2. Expert safety audit

`exp16_s0_expert_safety_audit_20260815` 在候选 outcome 前重放 24 个成功 expert suffix。Drawer expert P99.5 为 457.85 N、最大 565.31 N，主阈值冻结为 503.64 N；Bowl 为 200 N；Stove P99.5 为 298.95 N，主阈值 328.85 N。连续三步超限才停止，1000 N 为绝对应急上限。

## 3. 路线

比较冻结默认、task-weighted、两档 progress-monotone、persistent chunk、conservative median、medoid、smooth weighted 和 contact-smooth 九条路线。目标 future 对候选不可见，训练库仍为独立 EXP8 demos。

## 4. 校准结果

216 个 rollout、9,229 步。安全可用率 83.33%，需求率 41.67%，需求恢复 60%，候选有效率 100%，完整通过校准 recovery gate。Bowl/Stove 可用率均 100%，Drawer 50%。八条路线获 formal 授权。

## 5. Formal 工程失败

首次 formal run `exp16_s3_formal_recovery_20260815` 只执行 84 个默认 rollout。authorization 过滤错误地重新遍历 EXP15 旧路线表，导致七条 EXP16 路线消失。run 保留；修复一行后预检查明确列出八条路线，并以 `_r1` 重跑。

## 6. 正式数据

`exp16_s3_formal_recovery_r1_20260815` 在 84 个分支执行 672 个候选 rollout、26,709 步，route count 8，expert success 100%，target-future access false。分析 run 为 `exp16_s4_formal_analysis_20260815`。

## 7. 正式主要结果

安全候选可用率 78.57% 通过 70%；需求率 47.62% 通过 30%；候选有效率 100% 通过 90%。需求恢复率为 55%，低于 60%，因此整体 formal gate 失败。

## 8. 按任务结果

Bowl 可用率 100%，默认也成功所以无需求。Stove 可用率 89.29%，22 个需求组恢复 19 个（86.36%）。Drawer 可用率 46.43%，18 个需求组仅恢复 3 个（16.67%），是唯一主要缺口。

## 9. 路线表现

整体 safe success 最高为 smooth weighted 67.86%，task weighted 64.29%，median 61.90%，medoid 55.95%。所有路线无裁剪且状态有限；formal safety stop 为 2.38%–9.52%，说明 expert-relative envelope 消除了 EXP15 的大规模误停止。

## 10. 安全规则影响

原 200 N 对成功 Drawer/Stove expert 本身过严。改用 expert-relative envelope 后 Stove 从 EXP15 校准的 3/8 可用提高到 EXP16 formal 25/28，支持安全测量修正；这不等于真实部署安全证明，因为 envelope 仍来自 development expert trajectories。

## 11. Drawer horizon 诊断

Drawer 成功主要出现在后两处分支。失败早期分支的 remaining expert length 为 89–125，而政策统一在 80 步终止；成功分支通常只剩 40–74 步。Drawer formal 没有 safety stop，说明下一瓶颈是恢复 horizon 与策略进度，而不是 force gate。

## 12. 协议/实现失败

expert audit 首次导入遗漏 `src`，在 run 创建前失败并修复。首次 formal authorization route-set bug 留下完整失败 run，修复后 `_r1` 才是科学结果。所有旧 run 保留，没有删除内容。

## 13. 科学解释

reference-free feedback recovery 已在 Bowl/Stove 上形成强正式证据，并把总体可用率推过门槛。剩余 5 个百分点不是全局模型失败，而是 Drawer 早期长程恢复不足。直接进入 selector 仍不合格，因为需求组 oracle recovery 未达到预设 60%。

## 14. Claim 边界

允许：expert-relative safety 显著减少 Stove 误停止；Bowl/Stove recovery 在 formal development 上通过任务门；总体 candidate availability 和 demand 已建立。禁止：整个 recovery 模块通过、真实安全已证明、coordinator 已改善选择或项目完成。

## 15. 模块状态

```yaml
expert_relative_safety_audit: supported_in_development
bowl_recovery: formal_supported
stove_recovery: formal_supported
drawer_recovery: formal_unsupported
overall_availability: passed
decision_demand: passed
demand_recovery: failed_0.55_vs_0.60
project_complete: false
```

## 16. 下一实验

EXP17 使用固定、目标无关的 140-step recovery horizon，并比较 k=3/5/9、median/medoid、不同平滑、progress advance、source persistence 和短周期 chunk tracking；同时报告 80–160 horizon sensitivity。

## 17. 追溯

- 协议提交 `4bb7f4f`；audit import 修复 `e550cf2`；formal route 修复 `1d98f5b`。
- calibration analysis `exp16_s2_calibration_analysis_20260815`（通过）。
- formal primary `exp16_s3_formal_recovery_r1_20260815`；analysis `exp16_s4_formal_analysis_20260815`（失败）。
- EXP15/16 helper tests 通过，依赖检查通过。

