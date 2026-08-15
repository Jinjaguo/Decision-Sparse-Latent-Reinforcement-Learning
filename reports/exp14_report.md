# EXP14 实验报告：Object-Centric Task-Space Candidate Planning

## 1. 实验问题

把候选从局部动作扰动改成物体中心和任务空间意图后，是否能在 Drawer、Bowl、Stove 中至少两类任务上经常生成比成功 expert nominal continuation 更好的动作？

## 2. 为什么需要 EXP14

EXP13 已证明多源局部动作提案能改善 Stove，却在 Drawer/Bowl 上完全没有机会。EXP14 因此没有调幅度或放宽阈值，而是改变候选表示，加入 EEF 航点、物体坐标方向、跨示范几何重定向、局部逆响应和任务语义技能。

## 3. 实现路线

T1 使用 5/10/20 步大跨度 lookahead；T2 使用 object-relative EEF waypoint；T3 使用目标、任务轴和垂直方向基；T4 重定向其他示范的 EEF 位移；T5 拟合动作到 EEF 位移的正则逆映射；T6 实现 Drawer/Bowl/Stove 语义技能模板；T7 用冻结的几何/动作代价做多样化引导；T8 受限组合两个来源。所有 proposal 学习使用独立 EXP8 demos，目标 future outcome 不进入生成。

## 4. 工程和协议门

OSC translation 采用已审计的 0.05 m 输入缩放，动作限制为 `[-1,1]`，夹爪保持精确符号。632 个校准候选在预执行范围审计中均无预期裁剪。corrected-D same-state zero replay 和 wrist wrench 有效性均通过。

## 5. 校准数据

冻结计划 `exp14_s0_calibration_plan_r1_20260815` 包含 24 个分支和 632 个候选，T1–T8 覆盖三任务。早期 `s0` 计划因 leave-one-demo-out 只有一个训练示范而无法合法支持 T5，被保留但未执行；`r1` 显式使用 EXP8 独立训练参考库后重新冻结。

## 6. 校准结果

`exp14_s1_calibration_candidates_20260815` 生成 43,859 条逐步记录。候选 fidelity-valid 为 100%，但终点失败率 46.36%，灾难性接触代理 14.72%。总体机会 5/24：Drawer 0/8、Bowl 0/8、Stove 5/8。只有 Stove 的 T3 object-frame、T7 guided、T8 composed、T6 skill 获授权。

## 7. 正式执行

正式计划在结果前冻结，包含全部 84 个分支作为分母，只在 28 个 Stove 分支执行 308 个授权候选。轮转预算保证四条路线均被执行，而不是被候选较多的第一路线挤占。`exp14_s4_formal_candidates_20260815` 产生 18,381 条逐步记录，zero gate 和 wrench gate 均通过。

## 8. 正式结果

`exp14_s5_formal_analysis_20260815` 的总体机会率为 14/84（16.67%），低于 30%；Drawer 0/28、Bowl 0/28、Stove 14/28（50%）。候选有效率 100%，终点失败率 8.12%，灾难性接触代理 0.32%，中位 oracle gap 为 0，P90 为 0.2182。

## 9. 路线贡献

Stove 单路线机会率为 T7 guided 42.86%、T3 object-frame 32.14%、T8 composed 28.57%、T6 skill 10.71%。T3 成功率 99.11%，T7 为 96.43%，T8 为 82.14%，T6 为 80.36%。任务空间和组合路线对 Stove 有增量，但没有跨任务迁移。

## 10. Drawer/Bowl 失败机制

校准中 Drawer 的 expert nominal composite 固定为 6.0，所有成功候选的最大 gap 只有 0.00188；Bowl nominal 为 5.9108–5.9614，最大成功 gap 为 0.03888。二者都不可能通过冻结的 0.05 improvement gate。另一方面，改变前十步后盲目恢复原示范 suffix 会让抓取/contact 状态错位，解释了 task-space 候选的高失败率。

## 11. 消融解释

大跨度 lookahead、直接 waypoint、重定向和 inverse routes 在 Drawer/Bowl 上通常产生较大轨迹变化，却降低终点成功；这表明“动作明显不同”不等于“任务更好”。T7 的保守混合比纯 skill 稳定，T8 介于两者之间，支持反馈与风险控制的必要性。位置+orientation 的 T4 没有优于位置主导路线，开放环 waypoint 不是充分结构。

## 12. 工程失败与修复

EXP14 单测首次收集因测试未加入仓库 `src` 路径而失败，未创建实验 run；按既有测试模式修复后 6/6 通过。初版校准计划缺少足够的 leave-demo-out T5 训练样本，保留后以独立 EXP8 库重建 `_r1`。正式选择器最初可能按列表顺序耗尽 16 候选预算；在读取校准结果前改为路线轮转分配并提交。没有删除任何失败或旧计划。

## 13. 科学解释

EXP14 表明单次开放环 task-space chunk 仍不足以支持 contact-rich recovery。更重要的是，连续三轮都把不可部署的成功 target future 当成 selectable nominal，并要求候选显著击败它，这把“候选可用性”误写成“超越专家”。Stove 可以通过按钮角度 overshoot 获得额外 motion 分；Drawer/Bowl 的成功质量已接近上界，没有类似余量。

## 14. 提案修改

从 EXP15 起，target-demo future 只作为隐藏的评价上界，不再作为候选或默认动作。候选必须由其他 demos、行为模型、反馈技能或规划器生成，并在每个短 chunk 后根据真实观察重规划。选择价值相对一个冻结的非 oracle 默认策略评估，同时保留 expert-relative gap 作为诚实诊断。

## 15. Claim 边界

允许声称：task-space routes 在 Stove 正式分支上产生 50% 的 expert-improvement 机会；执行 fidelity 良好；开放环 suffix mismatch 和 expert-ceiling benchmark 是 Drawer/Bowl 的结构障碍。不能声称：task-space candidate generation 已解决、闭环恢复有效、consequence selector 改善行为或项目成功。

## 16. 模块状态与下一步

```yaml
task_space_execution_fidelity: supported
stove_task_space_opportunity: supported
drawer_expert_improvement_opportunity: unsupported_and_ceiling_limited
bowl_expert_improvement_opportunity: unsupported_and_ceiling_limited
open_loop_cross_task_generation: unsupported
benchmark_revision_required: true
project_complete: false
```

EXP15 构建 Reference-Free Closed-Loop Recovery Candidates，比较 object-centric kNN、单调轨迹检索、局部行为克隆、反馈技能、恢复 specialists、后果引导搜索、policy ensemble 和受限层次组合。

## 17. 验证与可追溯性

- EXP14 协议提交：`909cfdc`；正式路线平衡修复：`f9d33e8`。
- 校准 run：`exp14_s1_calibration_candidates_20260815`；分析：`exp14_s2_calibration_analysis_20260815`。
- 正式 run：`exp14_s4_formal_candidates_20260815`；分析：`exp14_s5_formal_analysis_20260815`。
- 校准 replacement 632，per-step 43,859；正式 replacement 308，per-step 18,381。
- EXP14 单测 6/6 通过；联合 EXP13/14 helper 测试 11/11 通过。

