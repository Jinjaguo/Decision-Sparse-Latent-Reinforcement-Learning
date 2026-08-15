# EXP15 实验报告：Reference-Free Closed-Loop Recovery Candidates

## 1. 实验问题

完全隐藏目标 demo future 后，来自独立成功示范的反馈候选是否能恢复任务，并相对一个冻结的非 oracle 默认策略形成真实选择需求？

## 2. 提案修正

EXP12–14 都把成功 expert continuation 当成可选 nominal，并要求其他候选高出 0.05。EXP15 将 expert future 改为隔离的只读评价上界。候选只访问当前观察、最近反馈和 EXP8 独立示范库，每次输出短 chunk 并重新观察。

## 3. 九条路线

实现了冻结默认 physical-kNN/10-step、object feedback、contact/full-state weighted feedback、单调轨迹每步/每十步重规划、weighted-k、两档物体帧 retargeting 和 conservative median ensemble。所有路线使用统一 action bounds 和精确夹爪符号。

## 4. 数据隔离

候选库来自 `exp8_s2_independent_refs_20260814`，不含 calibration episodes 41–42。目标 actions 只在 `expert_upper_bound.parquet` 生成路径中使用，没有传入 retrieval 或 `choose_chunk`。pre-outcome protocol 和 branch manifests 已哈希。

## 5. 校准执行

`exp15_s1_calibration_recovery_20260815` 在 24 个 corrected-D 分支执行 216 个候选 rollout，共 6,946 步。每个候选最多 80 步，成功或安全停止即结束。expert upper bound 成功率为 100%，target-future candidate access 标记为 false。

## 6. 主要结果

安全候选可用率为 15/24（62.5%），低于 70%；冻结默认失败/安全停止形成的 decision demand 为 14/24（58.33%）；需求组中仅 5/14 被其他候选恢复（35.71%），低于 60%。有效候选比例 93.98% 通过 90% 门槛，但整体 gate 失败。

## 7. 按任务结果

Bowl 的九条路线在 8/8 分支全部成功，安全可用率 100%，默认也全部成功，因此需求率为 0。Drawer 可用率 4/8、需求率 6/8、需求恢复 2/6；Stove 可用率 3/8、需求率 8/8、需求恢复 3/8。修正后的评价不再被 expert ceiling 阻塞，并清楚定位 Drawer/Stove policy source。

## 8. 路线结果

总体 safe success 最高的是 conservative 54.17%，其次 weighted feedback 50%、contact feedback 和 high-retarget 各 45.83%。Drawer conservative 达到 4/8，是最佳单路线。Bowl 所有路线均 8/8。Stove 最好只有 weighted 和 high-retarget 各 2/8。

## 9. 安全停止问题

绝对 200 N 阈值导致 default/object/contact 等路线总体 33%–42% safety stop。按任务看，Bowl 无停止、Drawer 很少，而 Stove 多数路线 5/8–8/8 提前停止。该阈值在协议前没有用成功 expert trajectory 校准，因此无法区分危险碰撞和任务本身所需的按钮接触。

## 10. 执行 fidelity

除 retarget 外路线均无动作裁剪；low-retarget 平均裁剪 7.48%，high-retarget 15.32%，后者超过 primary 10% 门。所有记录状态有限。整体有效候选 93.98%，说明 feedback 执行器本身可用，但高增益几何修正需要 guard 和幅度限制。

## 11. 为什么没有进入正式阶段

冻结授权要求路线 safe success 至少 25%、平均裁剪不高于 10%、safety stop 不高于 10%。所有非默认路线都因未校准的 stop rate 超门而未获授权；默认仅因协议要求被保留。没有足够的多候选正式集合，继续跑 formal 不能检验 recovery selection，因此 Stage formal 未执行，EXP15 判为校准失败并自动进入下一实验。

## 12. 工程验证

新增 retrieval distance、weighted chunk、monotone window 和 recovery metrics 共 5 个单测，全部通过；`pip check` 通过。运行未发生异常或失败 run。训练库、protocol 和 branches 均在 outcome 前锁定。

## 13. 科学解释

EXP15 首次证明 revised setup 形成了真实决策需求，并且无需目标 future 的候选可以在 Bowl 全面成功、在部分 Drawer/Stove 分支恢复。失败不再是“无法超过专家”，而是明确的 policy/safety 问题：Drawer 的检索还不稳定，Stove 的接触被未经校准的安全规则截断。

## 14. Claim 边界

允许声称：reference-free feedback candidate 在 calibration 上提供 62.5% 安全可用率，Bowl 8/8，且形成 58.3% 默认需求。不能声称：跨任务恢复已通过、力安全已验证、formal 泛化成立、consequence coordinator 已训练或闭环项目成功。

## 15. 模块状态

```yaml
target_future_leakage: not_detected
closed_loop_candidate_execution: supported
bowl_reference_free_recovery: supported_in_calibration
drawer_reference_free_recovery: partial
stove_reference_free_recovery: partial_and_safety_confounded
decision_demand: established_in_calibration
formal_recovery_gate: not_reached
project_complete: false
```

## 16. 下一实验

EXP16 先用成功 expert force/contact 轨迹冻结 task×phase safety envelope，再并行比较任务特异 progress features、单调检索、source persistence、conservative aggregation、phase-guarded retarget、动作平滑和 recovery specialists。EXP15 的默认与最佳路线保留为 controls。

## 17. 提交与运行

- 协议提交：`3962336`。
- calibration run：`exp15_s1_calibration_recovery_20260815`。
- analysis run：`exp15_s2_calibration_analysis_20260815`。
- 216 rollouts，6,946 steps，24 branches，9 routes。
- helper tests 5/5；依赖检查通过。

