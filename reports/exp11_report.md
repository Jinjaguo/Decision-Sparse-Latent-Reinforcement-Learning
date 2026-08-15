# EXP11 实验报告：Structured Macro-Action Replacement Causal Response

日期：2026-08-15
最终状态：Stage 0、Stage 1、availability-limited Stage 2 全部执行完毕
最终分类：`predictable_but_not_sparse_macro_response`

## 1. 结论摘要

EXP11 首次把因果干预从单个边界的局部关节状态脉冲，改成未来 10 步动作块的结构化替换：

```text
A_rep[t:t+K] = A_ref[t:t+K] + alpha * B_k
```

这次改变解决了 EXP9/EXP10 的近零目标病理。独立 formal pilot 中：

- 1,232 个 replacement、84,740 条逐步响应；
- I-A analytic、I-B residual、I-C phase edit 的 H10 非平凡效应比例分别为 75.36%、92.86%、76.19%；
- 84/84 formal matched-zero branches 的 integration、EEF 和 predicate 差异均为精确 0；
- wrist force/torque 逐步字段 100% 有效；
- 最佳 P-E graph 路线相对最佳简单基线的 H10/H20 误差改善 37.51%/38.70%，3/3 tasks 为正，demo-bootstrap absolute-improvement CI 为 `[0.0292, 0.0481]`，90% simultaneous path coverage 为 0.9172；
- terminal success-flip AUROC 为 0.8251，demo-cluster 95% CI `[0.7243, 0.9073]`，通过 terminal 分类门；
- coarse-regime AUROC/AUPRC 为 0.9354/0.9158，AUROC CI `[0.8868, 0.9823]`，固定 0.5 阈值下 sensitivity/specificity 为 0.9395/0.8355，也通过选择性门；
- 但 terminal endpoint P90 相对基线恶化 315.94%，所以完整 trajectory-distribution route 仍失败；
- macro decision sparsity 失败：top-20% branches 仅解释 25.59%–37.94% 总 consequence，没有达到 50%。

因此最强结论不是“稀疏决策点已找到”，而是：

> 在本次 3-task、7-demo/task 的独立 pilot 中，结构化动作块替换产生了非平凡、可预测、可校准的宏观因果响应；这些响应并未集中在少数 branch，因此支持宏响应建模，但不支持 macro decision-sparsity claim。

Stage 2 每任务只有 7 条全新 demo，小于协议的 8 条 full-confirmation 下限。全文所有“支持”均必须带有 `availability-limited pilot` 限定。

## 2. 主要 immutable runs

| 阶段 | Run | 内容 |
|---|---|---|
| Stage 0 | `exp11_s0_s6_support_audit_20260814` | action semantics、K=5/10/20 支持、残差秩、basis stability、动作边界、未使用 demo 审计 |
| Calibration reference | `exp11_s1_calibration_refs_20260814` | episodes 41–42，共 6 条成功 reference |
| Calibration replacement | `exp11_s1_calibration_replacements_r1_20260814` | 24 branches、1,116 replacements、77,180 per-step rows |
| Calibration analysis | `exp11_s1_calibration_analysis_r1_20260814` | 幅值、正负号、phase、clipping 分析 |
| Taskwise authorization | `exp11_s1_taskwise_authorization_r1_20260814` | family × task 授权 |
| Formal reference | `exp11_s2_formal_refs_20260814` | episodes 43–49，共 21 条独立成功 reference |
| Formal replacement | `exp11_s2_formal_replacements_r1_20260814` | 84 branches、1,232 replacements、84,740 per-step rows |
| Fidelity re-audit | `exp11_s2_formal_fidelity_reaudit_r1_20260814` | 按“未饱和 mismatch + clipped-chunk rate”重审执行保真度 |
| Final model analysis | `exp11_s2_formal_analysis_r4_3seed_deterministic_20260814` | 3-seed deterministic GPU routes、conformal、terminal、regime、sparsity |
| Final output audit | `exp11_s3_final_output_audit_r2_3seed_20260815` | 31 个必需 artifact、24 图、raw hash、行数和 gate 全部通过 |

早期 `r1/r2` formal analysis 保留为审计轨迹。`r3` 在启用 deterministic CUDA 后因缺少 `CUBLAS_WORKSPACE_CONFIG` 明确失败；最终 `r4` 在运行前固定 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，不挑选 seed，作为最终路线结果。

## 3. Stage 0：动作语义和 reference-only 支持

### 3.1 控制器动作语义

通过当前安装的 robosuite 1.4.0 源码核实：

- 7 维动作，前 6 维属于 `OSC_POSE`；
- 0:3 为归一化 EEF 位置增量，`1.0 -> 0.05 m`；
- 3:6 为归一化 axis-angle 增量，`1.0 -> 0.5 rad`；
- arm action 在控制器缩放前裁剪到 `[-1, 1]`；
- 第 7 维是 Panda gripper sign command，内部每步按符号积分，不能连续插值。

I-D task/object-space targeted edit 没有进入实验。原因不是实现困难，而是当前没有经审计、跨三任务有效的 task-progress-to-OSC/IK 映射；协议禁止临时猜测通道语义。

### 3.2 Stage-0 family 支持

| Family | Drawer | Bowl | Stove | Stage 1 |
|---|---:|---:|---:|---:|
| I-A DCT/spline/pulse | 支持 | 支持 | 支持 | 是 |
| I-B task×phase residual SVD | 支持 | 支持 | 支持 | 是 |
| I-C phase index edit | 支持 | 支持 | 支持 | 是 |
| I-D task-space target | unavailable | unavailable | unavailable | 否 |
| I-E gripper timing | 否 | 支持 | 支持 | task-specific |

Stage 0 产出 63 条 phase-support、518 条 spectrum、63 条 basis-stability、810 条 action-bound audit。Residual modes 只从 EXP8 reference demos 训练，并做 leave-demo-out subspace stability；formal demos 从未参与 basis 构造。

## 4. Stage 1：校准执行

### 4.1 Cohort 和预算

episodes 41–42 的 6 条 reference 全部成功、finite，integration/controller round-trip 最大误差均为 0。每条 demo 冻结 4 个 reference-only landmark，共 24 branches。

1,116 个 replacements 的分配：

- I-A analytic：864；
- I-B residual：192；
- I-C phase edit：48；
- I-E exact gripper timing：12。

连续 family 使用 0.05/0.10 两档幅值和正负配对；analytic 同时覆盖 DCT、cubic Bernstein spline、smooth pulse 各 3 个 temporal modes。

### 4.2 Zero 和 raw schema

- 24/24 matched-zero twins 精确重放；
- maximum integration L2 = 0；
- maximum EEF-position L2 = 0；
- predicate divergence = 0；
- wrist force/torque validity = 1.0；
- 每步记录 requested/executed/reference action、clip flags、restore hash、EEF/object state、predicate、exact contact mode、signed gap、normal velocity、force 和 torque。

### 4.3 Calibration effect 和幅值

| Family | n | Mean H10 | Median H10 | amplitude-effect corr | paired ± asymmetry | clipped chunks |
|---|---:|---:|---:|---:|---:|---:|
| I-A analytic | 864 | 0.0483 | 0.0400 | 0.4482 | 0.0299 | 1.62% |
| I-B residual | 192 | 0.1398 | 0.1217 | 0.6352 | 0.0393 | 5.21% |
| I-C phase edit | 48 | 0.1247 | 0.0981 | N/A | N/A | 0% |
| I-E gripper timing | 12 | 0.0159 | 0.0160 | N/A | N/A | 0% |

正负效应近似反对称但不是完全线性；两档幅值与效应呈明确正相关。I-E 在 Bowl/Stove 的效应均未达到门，因此拒绝。

I-B 在 Bowl calibration 的 clipped chunks 为 10.94%，刚超过 10% 门，因此 formal 只在 Drawer/Stove 使用。I-A 和 I-C 进入三任务 formal。

## 5. Stage 2：独立 formal pilot

### 5.1 Cohort、隔离和设计

formal 只使用 episodes 43–49：

- 21 条新 reference，7 条/任务；
- 21/21 全部成功、finite；
- snapshot/controller round-trip 最大误差 0；
- 与 calibration episodes 41–42、EXP8 episodes 30/31–40、以及更早 cohort 完全隔离。

每条 demo 4 branches，共 84 branches。formal replacement 数量：

| Task | I-A | I-B | I-C | Total |
|---|---:|---:|---:|---:|
| Drawer | 280 | 112 | 56 | 448 |
| Stove | 280 | 112 | 56 | 448 |
| Bowl | 280 | 0 | 56 | 336 |
| 合计 | 840 | 224 | 168 | 1,232 |

I-A formal 设计保留 DCT/spline 前两个 modes，并把 pulse 第一 mode 作为 held-out temporal family；幅值固定为 calibration 后的 0.10。I-B 使用 training-only residual modes。I-C 使用 reference index ±1 phase edit。

### 5.2 Formal zero、wrench 和 fidelity

84/84 zero branches 精确；84,740 条逐步 force/torque 全部有效。

原始 formal executor 的 family summary 把“饱和通道上预期发生的 requested-executed 差”也放进 mismatch P95，因而对 I-A/I-B 给出了过度保守的全局 `execution_valid=false`。没有修改原始 run；独立 re-audit 严格按协议使用：

```text
unsaturated requested/executed L∞ P95 < 1e-7
AND clipped-chunk fraction <= 10%
```

taskwise 结果：

| Task × family | clipped chunks | unsaturated mismatch P95 | Valid |
|---|---:|---:|---:|
| Drawer I-A | 2.50% | 0 | 是 |
| Drawer I-B | 9.82% | 0 | 是 |
| Drawer I-C | 0% | 0 | 是 |
| Stove I-A | 2.86% | 0 | 是 |
| Stove I-B | 2.68% | 0 | 是 |
| Stove I-C | 0% | 0 | 是 |
| Bowl I-A | 17.50% | 0 | **否** |
| Bowl I-C | 0% | 0 | 是 |

Bowl I-A rows 保留为 intent-to-replace 饱和诊断，不用于“faithful I-A Bowl”主张。

### 5.3 Formal causal effect

| Task | I-A mean H10 | I-B mean H10 | I-C mean H10 |
|---|---:|---:|---:|
| Drawer | 0.0802 | 0.2066 | 0.0660 |
| Stove | 0.0804 | 0.1425 | 0.1072 |
| Bowl | 0.0975* | unavailable | 0.1887 |

`*` Bowl I-A 有 17.5% clipped chunks，只能作为 intent-to-replace 描述。

按 temporal basis 聚合：

| Basis | n | Mean H10 | Median H10 |
|---|---:|---:|---:|
| DCT | 336 | 0.0693 | 0.0679 |
| Spline | 336 | 0.0831 | 0.0819 |
| Held-out pulse | 168 | 0.1252 | 0.1242 |
| Residual SVD | 224 | 0.1745 | 0.1860 |
| Phase index shift | 168 | 0.1206 | 0.0892 |

I-C 比 I-A 在 Stove/Bowl 更强，在 Drawer 更弱；它是 task-specific stronger consequence，而不是统一最优 family。Residual modes 在其被授权的 Drawer/Stove 上产生最强平均效应。

各任务最敏感区域：

- Drawer：P1 mean 0.1355，P3 0.1261，P2 0.1220，P4 0.0564；
- Bowl：P3 mean 0.1597，P1 0.1034，P2 0.0888；
- Stove：P1 mean 0.1084，P2 0.1071，P3 0.0895。

这些数值说明 phase 有响应差异，但后面的 cross-demo sparsity gate 没有通过，不能把它们叫作稳定稀疏决策点。

## 6. 多路线实现和完整-demo cross-fitting

所有训练/验证/测试拆分以完整 `(task, episode)` 为单位。21 demos 按 identity 排序 round-robin 分成 5 个非空 folds；每个 outer-train fold 再划分独立 calibration demos，用于 simultaneous path conformal，不在 intervention 行级随机拆分。

实现的方法：

1. global mean baseline；
2. action-norm ridge baseline；
3. P-A linear functional ridge；
4. P-B object-centric ridge；
5. P-C observed-phase switching experts；
6. P-D action-sequence GRU；
7. P-E object-graph MLP；
8. P-F residual generative audit；
9. P-G terminal regression/classification；
10. coarse-regime consequence classifier；
11. demo-level split-conformal path sets；
12. representation、switching、basis、phase、amplitude、sparsity ablations。

P-D/P-E 最终使用 3 个冻结 seeds `111012/111013/111014` 等权平均，无 seed 选择。RTX 4090、PyTorch 1.11.0+cu113、CUDA 11.3、cuDNN 8200；float32 训练、float64 scoring；deterministic algorithms、cuDNN deterministic 开启，benchmark 关闭。GRU/graph MLP 参数量分别为 11,444/9,620，最终 GPU 分析约 27.1 秒。

## 7. Prediction route 结果

| Route | H10 error | H20 error | H10 improve | H20 improve | Demo improvement CI | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Mean baseline | 0.05853 | 0.10288 | −0.04% | −0.03% | crosses 0 | 0.9269 |
| Action-norm baseline | 0.05851 | 0.10285 | 0% | 0% | `[0,0]` | 0.9286 |
| P-A linear | 0.04683 | 0.08217 | 19.96% | 20.11% | `[0.0140,0.0278]` | 0.9269 |
| P-B object | 0.04472 | 0.07535 | 23.56% | 26.74% | `[0.0161,0.0348]` | 0.9237 |
| P-C switching | 0.05698 | 0.09146 | 2.61% | 11.08% | `[-0.0067,0.0294]` | 0.9343 |
| P-D GRU, 3-seed | 0.03990 | 0.06757 | 31.80% | 34.30% | `[0.0253,0.0438]` | 0.9367 |
| P-E graph, 3-seed | **0.03656** | **0.06305** | **37.51%** | **38.70%** | **`[0.0292,0.0481]`** | **0.9172** |

P-B 相对 P-A 的 H20 改善只有 8.30%，未达到 object-centric route 的 15% 增量门，因此 `object_centric_prediction=unsupported`。P-E 的优势不能单独归因于 object-centric features，因为它同时改变了函数族。

P-C switching 比 P-A 更差，phase expert 没有帮助。P-D 比 P-B 改善约 10.3%，说明动作序列适配有方向性价值，但本实验没有隔离“状态 history”与“action sequence”，因此不能声称 history adaptation 已独立成立。

## 8. Distribution、terminal 和 coarse-regime gates

### 8.1 完整 trajectory-distribution route

P-E 满足：

- H10/H20 改善均超过 15%；
- demo-bootstrap improvement CI lower > 0；
- 3/3 tasks point estimate positive；
- simultaneous path coverage 0.9172，位于 `[0.85,0.95]`。

但 P-G terminal endpoint P90 相对 baseline 恶化 315.94%，mean improvement demo CI `[-0.0101,-0.0042]`。由于 trajectory route 要求 terminal P90 至少改善 15%，最终 `trajectory_distribution=unsupported`。

### 8.2 P-F generative

残差 kurtosis 很高，但 two-Gaussian audit 的 component separation 只有 0.0528，低于预冻结的 1.0；高 BIC 改善来自重尾/方差差异，不足以证明可分离多模态。因此 P-F 没有被科学授权，也没有用“高 kurtosis”包装成多模态成功。

### 8.3 P-G terminal consequence

- success-flip AUROC 0.8251；
- demo-cluster 95% CI `[0.7243,0.9073]`；
- 达到 AUROC ≥0.80 且 CI lower ≥0.70。

所以分类分支独立通过，`terminal_consequence=supported`。Endpoint regression 分支失败，二者必须分开报告。

### 8.4 Coarse regime

- AUROC 0.9354，demo-cluster CI `[0.8868,0.9823]`；
- AUPRC 0.9158；
- sensitivity 0.9395；
- specificity 0.8355；
- false-safe 0.1645。

预测门和选择性门均通过。这是 outcome-focused risk signal，不等于 macro decision sparsity。

## 9. Macro decision sparsity

| Task | Cross-demo rank | Top-20% consequence mass | Held-out pulse rank rho | Overall |
|---|---:|---:|---:|---:|
| Drawer | 0.7460 | 0.2809 | 0.9643 | 失败 |
| Bowl | 0.3339 | 0.3794 | 1.0000 | 失败 |
| Stove | 0.3021 | 0.2559 | 0.9643 | 失败 |

Held-out pulse 的 phase ranking 在 3/3 tasks 中保持，但：

- 只有 Drawer cross-demo rank ≥0.60；
- 0/3 tasks 的 top-20% mass ≥0.50；
- 没有 2/3 task replication。

所以 `macro_decision_sparsity=unsupported`。最重要的负面结果是宏观 consequence 广泛分布，而不是只出现在少数 branch。

## 10. Multi-axis classification

| Axis | Classification | 依据 |
|---|---|---|
| causal_macro_effect | supported | 三个 family 均产生大量非平凡效应；taskwise fidelity 有明确边界 |
| temporal_basis_transfer | supported | held-out pulse rank rho 0.964–1.0，DCT/spline/residual/pulse 均有非零响应 |
| object_centric_prediction | unsupported | P-B 相对 P-A 增量 <15% |
| switching_dynamics | unsupported | P-C 不及 P-A，CI 跨 0 |
| trajectory_distribution | unsupported | terminal endpoint P90 硬门失败 |
| terminal_consequence | supported | success-flip AUROC 和 CI 通过 |
| macro_decision_sparsity | unsupported | top-20% mass 与 2/3 replication 失败 |

Summary：`predictable_but_not_sparse_macro_response`。

## 11. 对协议最终问题的逐项回答

1. **哪些 family 有支持？** Stage 0 支持 I-A/I-B/I-C；I-E 仅 Bowl/Stove 有事件支持；I-D unavailable。Formal taskwise fidelity 支持 Drawer/Stove 的 I-A/I-B/I-C，以及 Bowl 的 I-C；Bowl I-A 因 clipping 失败。
2. **Stage-1 zero 是否精确？** 是，24/24 精确 0；formal 84/84 也精确 0。
3. **动作块能否忠实执行？** 未饱和通道 mismatch P95 全为 0。除 Bowl I-A 外，所有正式 task×family clipped-chunk rate ≤10%。
4. **clipping 有多少？** Formal 总体 I-A 7.62%、I-B 6.25%、I-C 0%；但 Bowl I-A 为 17.5%，单独失败。
5. **哪些 family 产生非平凡 effect？** I-A/I-B/I-C；I-E calibration effect 太弱。
6. **effect 是否随幅值增大？** 是，calibration correlation 为 I-A 0.448、I-B 0.635。
7. **正负号是否不对称？** 有轻微不对称，平均约 0.030/0.039，不是 EXP6 那种强不对称。
8. **哪些 phase 最敏感？** Drawer P1/P3，Bowl P3，Stove P1/P2；但不形成稳定稀疏 branch 集合。
9. **DCT/spline 是否 transfer？** 是，formal mean H10 0.069/0.083；held-out pulse ranking 也保持。
10. **training-only residual 是否 transfer？** 是，在 Drawer/Stove mean H10 0.207/0.142，且执行 gate 通过。
11. **phase edit 是否更强？** 在 Stove/Bowl 比 I-A 强，在 Drawer 更弱，属于 task-specific。
12. **gripper timing 是否重要？** 否，calibration mean H10 0.0159，未授权 formal。
13. **是否解决近零目标病理？** 是，三类 formal family 的非平凡比例 75%–93%。
14. **哪些 prediction routes transfer？** P-A、P-B、P-D、P-E 的 H10/H20 均超过 15% 改善；P-C 不通过。
15. **object-centric 是否胜过 full state？** 有 8.3% 改善，但未达到 15% 硬门，故不支持该轴。
16. **switching dynamics 是否帮助？** 否。
17. **history adaptation 是否帮助？** GRU action-sequence 有方向性提升，但没有隔离 state-history contribution，不能作独立结论。
18. **generative route 是否通过 path coverage？** 未授权；残差是重尾而非可分离多模态。
19. **terminal predicate consequence 是否可预测？** 是，AUROC/CI 通过。
20. **macro effect 是否稀疏？** 否。
21. **top-20% fraction？** Drawer 28.09%、Bowl 37.94%、Stove 25.59%。
22. **held-out temporal modes 是否保排名？** 是，rho 0.964/1.0/0.964。
23. **是否 task/family-specific？** 是，尤其 Bowl I-A clipping、I-B 可用性、I-C 强度均不同。
24. **multi-axis classification？** 见第 10 节；summary 为 `predictable_but_not_sparse_macro_response`。
25. **最强允许主张？** 结构化动作块替换暴露了可预测、可校准、但广泛分布的宏观因果响应。
26. **offline decision-allocation utility 是否 eligible？** 稀疏敏感度分配不 eligible；基于已通过 terminal/coarse-risk 的 outcome-focused 离线分配可作为独立后续实验，但不能叫 sparse scheduler。
27. **专门 latent representation 是否 eligible？** 仅 development-level eligible，因为 causal macro 和 terminal consequence 已支持；必须在新独立 cohort 确认压缩后才谈 utility。
28. **latent RL 是否 eligible？** 否；尚无 utility/control 证据，且 sparsity 失败。
29. **EXP12 应测试什么？** 见 `reports/next_exp_from11.md`：校准的因果 macro-action predictive latent，重点验证压缩、terminal/regime 保真与 OOD abstention，不训练 policy/RL。

## 12. 完整性、测试和限制

- 最终 tests：129/129 passed；
- EXP11 known-answer tests：25/25 passed；
- `pip check` 在最终交付前再次运行；
- 31 个必需 artifact、24 张规定图、所有 raw hashes 和行数审计通过；
- MuJoCo 使用 CPU，神经路线使用 RTX 4090；
- raw runs 位于 `runs/`，按项目规则不纳入 Git，但最终报告记录 canonical run IDs 和 hash audit。

限制：

1. formal 每任务只有 7 demos，不能升级为 full confirmation；
2. Bowl I-A 不满足 clipping 门；
3. trajectory target 主要是 normalized macro-effect path，不能替代完整高维世界状态生成；
4. P-E 胜出不等于 object-centric 增量轴通过；
5. endpoint regression 和 full trajectory-distribution gate 均失败；
6. 当前三任务的 public demonstrations 0–49 已被历史实验占用，下一次独立确认必须扩展任务或生成真正新的成功 trajectories；
7. 本实验没有运行 scheduler、online control、policy learning、MPC、VLA 或 latent RL。
