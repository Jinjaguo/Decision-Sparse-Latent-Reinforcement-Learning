# EXP13 实验报告：Task-Aware Multi-Source Candidate Generation

## 1. 实验问题

在不训练新协调器的前提下，八类互补候选生成机制能否让同一状态下经常出现比成功 nominal continuation 更好的动作块，并把这种机会从 Stove 扩展到 Drawer 和 Bowl？

## 2. 为什么需要 EXP13

EXP12 已找到后果预测和 pairwise 排序信号，但 84 个候选组中只有 10 个存在超过 0.05 的改进机会，而且全部属于 Stove。选择器无法选择不存在的好动作，因此 EXP13 先解决候选覆盖，而不是继续扩大预测网络。

## 3. 相比 EXP12 的结构变化

EXP13 同时实现了多通道解析模式、时间扭曲、训练集残差、跨示范动作库、任务进度方向、EXP12 引导、精确夹爪时序和受限组合八条路线。所有学习或检索均排除目标 demo；候选先冻结，再进入模拟器执行。

## 4. 冻结协议

episodes 41–42 仅用于校准授权，episodes 43–49 用于 held-demo development-formal 复现。每个正式 demo 使用四个既有 corrected-D 分支。授权规则为：裁剪块比例不高于 10%、成功率至少 80%、后果非退化并至少产生一个校准改进机会；每个任务最多四个 family。正式结果前已冻结候选计划。

## 5. 校准数据和结果

`exp13_s1_calibration_candidates_r1_20260815` 执行 24 个分支、589 个候选和 40,507 条逐步记录。修正分母后的分析 `exp13_s2_calibration_analysis_r2_20260815` 显示 98.64% 候选 fidelity-valid，终点失败率 7.47%，灾难性接触代理 1.53%。机会率为 4/24（16.7%）：Drawer 0/8、Bowl 0/8、Stove 4/8。只有 Stove 的 G2 temporal、G3 residual、G4 library 和 G1 multichannel 获得正式授权。

## 6. 正式执行

`exp13_s4_formal_candidates_20260815` 在 84 个正式分支的冻结清单上，对 28 个 Stove 分支执行 308 个候选，共产生 18,381 条逐步记录。没有授权候选的 Drawer/Bowl 分支仍保留在总体分母。same-state zero replay 全部通过，wrench 有效率为 1.0。

## 7. 主要结果

正式机会率为 12/84（14.29%），低于 30% 门槛；Drawer 为 0/28，Bowl 为 0/28，Stove 为 12/28（42.86%）。中位 oracle gap 为 0，P90 为 0.08266，最大观察 gap 为 0.3628。100% 正式候选 fidelity-valid，终点失败率 2.92%，灾难性接触代理 0.97%。因此失败不是数值执行问题，而是跨任务候选缺失。

## 8. Family 贡献

在 12 个正式改进组中，oracle-best 来源为 G1 multichannel 5 个、G3 residual 3 个、G4 library 3 个、G2 temporal 1 个。单 family 的 Stove 机会率分别为 G1 42.86%、G3 25.0%、G4 17.86%、G2 14.29%。这说明多源设计确实比单一 intervention 更有价值，但价值仅在 Stove 上复现。

## 9. 安全和尾部

正式候选无裁剪、无非有限状态，且 zero gate 通过。G4 library 的成功率最低，为 87.5%；G1 为 100%，G2/G3 各为 98.81%。总体终点失败率和灾难性接触代理较低，但不能据此宣称协调器安全，因为本实验没有运行选择器或闭环策略。

## 10. 消融解释

校准阶段 G6 guided 在 Stove 仅有 12.5% 机会，未进入前四；G8 composed 和 G7 gripper 均为 0，说明简单组合或只改夹爪时序不足。Drawer/Bowl 的时间编辑和动作库虽产生较大后果多样性，却没有正向机会，且部分路线降低终点成功。任务进度残差保持成功但方向增益太弱。现有数据支持“需要不同提案抽象”，不支持只增大幅度或放宽门槛。

## 11. 未完成的次要消融

冻结 prompt 中的 shared-vs-task-specific mode、三种 retrieval context、diversity on/off 和 4/8/12 budget 没有全部形成独立正式矩阵。核心八 family、guided/unguided、single/composed 和 clipped intent 诊断已有覆盖，但这些缺失限制了对 family 内机制的归因。由于主结论由跨任务 0 机会决定，这些缺失不会把 EXP13 的失败改判为成功，后续实验应显式补齐对应结构比较。

## 12. 工程失败与修复

第一次校准执行因 robosuite 固定写入 `C:\tmp\robosuite.log` 被沙箱拒绝，失败 run 被保留，随后在获批权限下以 `_r1` 重跑。分析脚本第一次在导入阶段因缺少仓库根路径失败，未创建 run；修复只影响导入。随后发现正式分母若只按已执行候选分组会漏掉无授权任务，于是改为从冻结计划读取全部 84 个分支，并将有限且未裁剪定义为 primary fidelity-valid。第一次修正版因执行 run 不含分支清单而留下失败的 `_r1` 分析目录，最终通过显式 `--plan-run` 在 `_r2` 完成。所有失败和旧结果均未删除。

## 13. 科学解释

EXP13 支持一个任务特异结论：多源局部动作提案能稳定扩大 Stove 的有效改进空间。它同时反驳这些局部机制已经解决跨任务候选生成。Drawer 需要获得并维持把手接触后沿抽屉轴拉动；Bowl 需要抓取、抬升、运输、放置和释放的几何序列。围绕 nominal 十步动作做局部编辑没有显式表达这些意图。

## 14. Claim 影响

允许的表述是：在 development-formal Stove 分支上，多源动作候选使 42.9% 的组出现有效 nominal 改进；执行 fidelity 良好。禁止的表述是：候选生成已解决、改进可跨任务泛化、后果协调器已改善选择、闭环控制已成功、或当前三维后果表示已经充分。

## 15. 模块状态

```yaml
multi_source_execution_fidelity: supported
stove_candidate_opportunity: supported
drawer_candidate_opportunity: unsupported
bowl_candidate_opportunity: unsupported
cross_task_candidate_generation: unsupported
offline_consequence_coordination: blocked_by_candidate_coverage
closed_loop_utility: not_tested
project_complete: false
```

## 16. 下一实验

EXP14 转向 Object-Centric Task-Space Candidate Planning。它不靠单一阈值变化，而是并行测试大跨度 phase skip、EEF 航点、物体坐标方向、跨 demo 几何重定向、局部逆响应、任务语义有限状态技能、预测器引导低维搜索和受限混合。成功门槛保持 30% overall、至少两任务各 20%、候选有效率至少 90%。

## 17. 提交、hash 和审计

- 协议冻结提交：`37915d9`；生成器级分析提交：`204b711`。
- 校准 candidate plan SHA256：`48cd12ac3c2e68853e28419e29628e08da9e18e5862601f75eead4128f12c337`。
- 校准 per-step SHA256：`2a78aabca8a2044ea2be342fbb42373dffad5e9613f937a837770a64f6ea3f60`。
- 正式 candidate plan SHA256：`8991d81165c692e2c4bb46449d6768608094b612a6c5623220986a743b2e4bf9`。
- 正式 per-step SHA256：`565bfe672eda5a822fa84737c7dc37c2b1167a404210025e73b7786bce3d9300`。
- 正式分析 metrics SHA256：`c8967f0a4447be86ff4d9017e0a858ca433675117c71bf66b6fa51c9c8554ce0`。
- EXP13 单元测试 5/5 通过，`pip check` 通过。
