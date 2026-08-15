# EXP9 实验报告：Action-Conditioned Hybrid Contact Response Distribution

日期：2026-08-14
状态：Stage A 完整执行；Stage B 按预注册停止
最终 Stage-A 分类：`action_conditioning_insufficient`
目标构造：`runs/exp9_a0_hybrid_targets_r2_20260814`
模型比较：`runs/exp9_a1_a6_retrospective_models_r4_20260814`
独立审计：`runs/exp9_a7_output_audit_20260814`

## 1. 结论摘要

EXP9 严格执行了协议规定的先回顾、后确认结构。Stage A 在不修改 EXP8
锁定数据的前提下，构造了 5 步参考历史、未来 5 个冻结策略动作、离散接触事件、
H1/H3/H5/H10/remaining 连续响应目标，并用整条 demonstration 隔离的五折
cross-fitting 比较了：

1. EXP8/EXP5 风格 Baseline B；
2. factorized event/regression baseline；
3. lifted hybrid linear baseline；
4. compact graph Gaussian-mixture model；
5. compact temporal-sequence Gaussian-mixture model。

连续分布预测出现了明确的开发信号。相对 Baseline B：

| 模型 | H1 energy 改善 | H3 energy 改善 | H5 energy 改善 |
|---|---:|---:|---:|
| Graph mixture | +37.52% | +43.42% | +44.52% |
| Temporal sequence mixture | **+42.70%** | **+47.91%** | **+49.33%** |

两种 mixture 模型的 90% 边际区间覆盖约 0.925–0.945，且 mixture entropy、
预测方差和跨样本方差均非退化。对应的 demo-level energy-score improvement
bootstrap CI 在 H1、H3 都完全为正。

但是 EXP9 的最低可行性门是联合门，而不是只看连续分数。没有任何非 oracle
模型能在 held-out EXP8 demonstrations 上同时达到：

```text
specificity > 0.50
sensitivity >= 0.85
H1 和 H3 energy score 均优于 Baseline B
连续不确定性非退化
```

最佳 mixture 模型在 H1/H3 的 specificity 只有约 0.30–0.35，false-safe rate
约 0.65–0.70；其 sensitivity 也未在 H1/H3 同时达到 0.85。根据预注册规则，
`new_cohort_authorized=false`，所以没有审计/生成 EXP9 新 cohort，没有运行新的
matched-zero，没有产生新的 q interventions，也没有执行 Stage B。

这不是“EXP9 什么都没做”。它完成了协议要求的最便宜、最关键的可行性实验，
并避免了在一个尚不能选择性预测接触安全性的模型族上再花费 17,280 次仿真。

## 2. 研究问题与可允许结论

EXP9 问的是：跨 demonstration 的可复用结构是否存在于

```text
p(contact event, physical response | history, q perturbation, action chunk)
```

而不是单一局部 Jacobian。

Stage A 支持的开发性结论是：小型 action/history-conditioned mixture models
能显著改善 held-out demo 的连续 proper score，并产生非退化、接近目标覆盖率的
不确定性。

Stage A 不支持：

- 一个联合 hybrid model 已经可用于接触风险选择；
- history 或 5-action chunk 单独具有显著增量价值；
- contact graph 比纯 temporal encoding 更重要；
- mixture 改善已经证明真实多模态动力学；
- EXP9 已完成独立确认；
- offline scheduler、online control 或 latent RL 已经合格。

## 3. 前置审计

### 3.1 必读资料与路径差异

执行前完整读取 PROJECT、EXP1–EXP8 报告、EXP8 next-experiment 文档、证据表和
最新研究日志。协议所写的 EXP3 路径在仓库中实际包含 `_q_`；协议所写的 EXP4
实验摘要文件不存在，因此使用完整 EXP4 report、控制协议、manifests 与 run
证据作为审计替代，没有虚构缺失文件。

### 3.2 环境基线

- conda：`libero-exp1`；
- Python：3.8.20；
- PyTorch：1.11.0+cu113；
- GPU：NVIDIA GeForce RTX 4090；
- baseline tests：76/76 passed；
- `pip check`：无 broken requirements；
- MuJoCo：Stage A 未重新运行；EXP8 锁定 CPU 仿真仍是数据真值。

### 3.3 Git 与用户文件

执行开始时用户已有 `README.md`、`reports/README.md` 和若干 prompt/pytest-temp
变更。它们没有被覆盖、清理或纳入 EXP9 科学数据。

## 4. EXP9-A0：目标构造

### 4.1 锁定来源

核心 EXP8 原始文件在构造前后 SHA-256 保持不变：

| 文件 | SHA-256 |
|---|---|
| `interventions.parquet` | `31cd1ff86a9196c83d0c7312e4c1013fa80671db28ebf793e091a376d6540c9a` |
| `per_step_effects.parquet` | `58c293bd7cf6f389f2e221b1ff28efdd6c269a13148c0f66645877b1d58dd8c3` |
| reference contact frames | `34cbdb1410cac408dcda6d1da288942e51a8f4a4adcdbebf0bbc9bf547b1c26b` |

独立 output audit 再次计算所有来源 hash，全部通过。

### 4.2 规模与对齐

| 项目 | 数量/维度 |
|---|---:|
| Demonstrations | 30（10/task） |
| Interventions | 17,280 |
| H1/H3/H5/H10/remaining targets | 86,400 |
| History | 5 × 61 |
| Frozen action chunk | 5 × 7 |
| Baseline hybrid input | 82 |
| Graph input | 340 |
| Flattened temporal input | 375 |
| Drawer/Stove response | 26 |
| Bowl response | 32 |

响应维度差异来自 EXP8 已冻结的 `3*n` task-object channel 规则：Bowl 包含 bowl 与
plate 两个 task objects，不能后验截断为 26。模型按任务独立拟合和评分，只使用该
任务合法的输出通道。

历史以 branch 前的当前 reference boundary 为末端，轨迹开头左填充最早状态并
提供显式 mask。action chunk 从 `branch_time` 的下一冻结策略动作开始，末端右零
填充并提供 mask。每个 action 与 `trajectory_states.npz` 中的 7 维 actions 对齐。

### 4.3 事件支持

| Horizon | Drawer preservation | Bowl preservation | Stove preservation |
|---|---:|---:|---:|
| H1 | 0.6330 | 0.3198 | 0.4783 |
| H3 | 0.5347 | 0.1344 | 0.2691 |
| H5 | 0.5003 | 0.0856 | 0.1790 |
| H10 | 0.4090 | 0.0495 | 0.0738 |
| Remaining | 0.1741 | 0.0201 | 0.0278 |

named-pair add/drop 和 physical-group add/drop 在每个任务/主要 horizon 都有大量
正例，因此 Stage A 没有因事件稀少而停止。

### 4.4 目标边界

锁定 raw 数据足以构造：exact-mode preservation、named-pair add/drop、group
add/drop、first change time、26/32 维 signed physical response、signed-gap delta
和 normal-velocity delta。

EXP8 per-step raw 没有保存 perturbation 后的完整 wrench/force trajectory；只有
reference contact-frame feature 中的当前 force。EXP9 因此没有伪造 wrench rollout
target。该缺失是 Stage-A scope limitation，也是未来若重新采集数据必须修正的 schema
要求。

## 5. 模型与 leakage 控制

### 5.1 Cross-fitting

- 每任务 5 outer folds；每 fold 恰好 2 条 test demos；
- 同一 demo 的任何 state、branch、direction 或 horizon 都不进入训练；
- held-out random q direction 从训练中排除，但保留在测试 intent population；
- normalization、response scale、residual uncertainty 和 event threshold 只用训练 demos；
- test event labels 从未输入 continuous head；
- 259,200 条 test predictions 先写入并 hash，再计算汇总 gate。

Prediction SHA-256：

```text
b2ce4d6d486fcdadbbdd235c9abe08e9d185ea09c2402321001730054a941d8f
```

### 5.2 模型

| 模型 | 结构 | 参数/说明 |
|---|---|---:|
| Baseline B | logistic + ridge diagonal Gaussian | linear |
| Factorized | predicted event probabilities + ridge response | linear |
| Lifted hybrid | state squares + state×action interactions | compact linear |
| Graph mixture | 128/96 set-feature encoder + 3-component MDN | 102,592 |
| Temporal sequence mixture | history GRU48 + action GRU24 + 96/80 head + 3-component MDN | 74,624 |

两个神经模型均低于预注册 0.5M small cap。训练种子 `190901/190902/190903` 等权
ensemble，未选择最好种子。CUDA 使用 float32，proper scores/bootstrap 使用
float64。启动前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 并启用 deterministic
algorithms。

## 6. 连续分布结果

Demo-averaged multivariate energy score（越低越好）：

| 模型 | H1 | H3 | H5 |
|---|---:|---:|---:|
| Baseline B | 4.6787 | 4.9942 | 5.0121 |
| Factorized | 4.7763 | 5.0968 | 5.1470 |
| Lifted hybrid | 9.1099 | 9.1875 | 9.1128 |
| Graph mixture | 2.9234 | 2.8255 | 2.7805 |
| Temporal sequence mixture | **2.6807** | **2.6015** | **2.5396** |

Graph 的 Baseline-B minus model demo difference：

- H1：+1.7553，95% bootstrap CI `[1.1304, 2.4461]`；
- H3：+2.1687，CI `[1.5055, 2.9217]`。

Temporal sequence：

- H1：+1.9980，CI `[1.3334, 2.7296]`；
- H3：+2.3927，CI `[1.6960, 3.1770]`。

这些是开发性 held-out-demo 结果，不是独立 EXP9 confirmation。

### 6.1 不确定性

| 模型 | H1 coverage90 | H3 coverage90 | H5 coverage90 |
|---|---:|---:|---:|
| Baseline B | 0.7700 | 0.7621 | 0.7630 |
| Graph mixture | 0.9254 | 0.9280 | 0.9278 |
| Temporal sequence | 0.9362 | 0.9410 | 0.9447 |

Mixture predicted-standard-deviation median 为 0.56–0.60，P95 为 1.59–1.75，
mean mixture entropy 为 0.74–0.82，满足 Stage-A “非退化”要求。覆盖率是逐通道
边际覆盖，不等同于 26/32 维联合 ellipsoid coverage。

## 7. 接触事件结果与停止原因

整体 mode-preservation risk：

| 模型 | Horizon | AUROC | AUPRC | ECE | Sensitivity | Specificity | False-safe |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline B | H1 | 0.5991 | 0.5976 | 0.2070 | 0.6847 | 0.4353 | 0.5647 |
| Baseline B | H3 | 0.6117 | 0.4359 | 0.2611 | 0.6408 | 0.5047 | 0.4953 |
| Graph mixture | H1 | 0.6232 | 0.6254 | 0.0478 | 0.8414 | 0.3047 | 0.6953 |
| Graph mixture | H3 | 0.6072 | 0.4984 | 0.1788 | 0.7946 | 0.3458 | 0.6542 |
| Temporal sequence | H1 | 0.6238 | 0.5868 | 0.0248 | 0.8385 | 0.3024 | 0.6976 |
| Temporal sequence | H3 | 0.6100 | 0.4059 | 0.1826 | 0.8262 | 0.3159 | 0.6841 |

阈值由各 fold 训练数据中“先满足 sensitivity≥0.85，再最大化 specificity”选出；
上表是 held-out demos 的实际泛化结果。训练侧约束没有在 test demos 保持，且
specificity 远低于 Stage-A 0.50，更远低于 confirmatory H2 的 0.70。

最难的任务是 Bowl：H3 preservation 仅 0.134，Graph H3 sensitivity 0.707，
Temporal H3 sensitivity 0.796；其连续 energy 也高于 Drawer/Stove。Stove 的 H1
specificity 最低，Graph 0.238、Temporal 0.268。失败不是由单一任务解释。

## 8. 四类模型的可行性判断

| 模型 | H1/H3 response | 风险门 | 不确定性 | Stage A |
|---|---|---|---|---|
| Factorized | 失败，较 B 差约 2% | 失败 | 通过 | 失败 |
| Lifted hybrid | 严重失败 | 失败 | 通过 | 失败 |
| Graph mixture | 通过 | 失败 | 通过 | 失败 |
| Temporal sequence mixture | 通过 | 失败 | 通过 | 失败 |

没有一个模型同时通过三列。因此 Stage A 不支持选择 1 primary + 2 secondary，
`architecture_selection.json` 中 selection 为空。

## 9. 为什么没有执行 Stage B

精确协议规定：若 Stage A 没有一个 non-oracle model 同时通过最低 gate，必须：

```text
STOP before new cohort
write feasibility-failure report
```

所以以下数量全部为零，并非遗漏：

| Stage-B 项目 | 数量 |
|---|---:|
| 新 qualified demos | 0 |
| 新 branches | 0 |
| 新 matched-zero continuations | 0 |
| 新 q interventions | 0 |
| 新 per-step simulator rows | 0 |

因此 EXP9 不能回答“新 cohort 的 corrected-D 是否仍为零”；它只确认 EXP8 来源 hash
与既有 corrected-D 数据不变。

## 10. 未执行的正式 H1–H6 与 ablations

Stage B 的正式 H1（对所有 baselines 至少 10%）、H2 完整 event macro metrics、
H3 vector error/coverage、H4 H10/remaining rollout、H5 history/action ablations、H6
multimodality test 均未执行。Stage A 已构造 H10/remaining targets，但按照停止规则
不继续扩大模型分析。

因此不能声称：

- history 显著有效；
- 5-action chunk 显著有效；
- graph contact identity 有增量价值；
- mixture 优于 unimodal 的原因是多模态而非更灵活的 heteroscedastic fit。

Temporal model 比 graph model 的 energy 更好，是架构比较信号，不是 isolation
ablation。

## 11. 保留的失败与修正

1. `exp9_a0_hybrid_targets_20260814`：3 个 trajectory-start 的 gripper current action
   是 1 维初始化，其余为 2 维。固定为 2 维、缺失初始化分量零填充并保留 history
   mask；新 run id 重建。
2. `exp9_a1_a6_retrospective_models_20260814`：发现 task-specific response 是
   26/32 维，停止后按 EXP8 冻结 schema 处理，未截断 Bowl。
3. `...models_r1...`：deterministic CUDA 因缺少 cuBLAS workspace 环境变量停止；
   新进程前设置官方要求变量，无 CPU fallback。
4. `...models_r2...`：linear Monte Carlo sampler 仍硬编码 26 维；改为冻结的
   per-task dimension。
5. `...models_r3...`：predictions 已锁定，但空 plots 目录被重复创建导致最终失败；
   不修改失败 run，修正目录创建后完整重跑。
6. `...models_r4...`：唯一完整模型 run；prediction lock、metrics 与 plots 完成。

所有修正都是 schema/device/output-path 问题；没有根据测试结果改模型、fold、seed、
epoch、threshold rule 或 gate。

## 12. 产物与审计

Stage A 必需产物全部生成：

- `retrospective_target_audit.json`；
- `retrospective_event_prevalence.parquet`；
- `retrospective_model_comparison.parquet`；
- `retrospective_risk_metrics.parquet`；
- `retrospective_distribution_metrics.parquet`；
- `retrospective_training_cost.json`；
- `retrospective_power_estimate.json`；
- `architecture_selection.json`；
- write-once `retrospective_predictions.parquet` 与 prediction lock；
- task-level independent audit metrics；
- 6 个非 placeholder Stage-A plots。

独立审计通过全部条件：source/prediction hashes、259,200 rows、key uniqueness、
5 model set、5 folds、H1/H3/H5、非空 plots、Stage B 未授权且不存在 Stage-B runs。

最终 repository-wide tests：86/86 passed；`pip check` clean；`compileall` 对
`scripts/exp9` 与 `src/decision_sparse_rl/metrics/exp9.py` 通过。新增测试覆盖 history/
action padding masks、contact add/drop、energy score、training-only threshold、lifted
dictionary determinism、mixture parameter validity/moments、interval coverage 和
demo-fold isolation。Stage-B-only 的 simulator/raw-lock tests 没有伪装为已执行。

训练成本：Baseline B 5.53 s，factorized 0.60 s，lifted 5.28 s，graph mixture
102.41 s，temporal sequence 213.03 s；完整 wall time 427.98 s。

## 13. 最终问题回答

| 问题 | 回答 |
|---|---|
| Stage A 是否授权新数据？ | 否。联合最低门失败。 |
| 哪些模型有连续预测可行性？ | Graph/temporal mixture；temporal 最好。 |
| 是否有独立 confirmatory cohort？ | 未审计、未收集；协议禁止继续。 |
| 新 demos/branches/interventions？ | 0/0/0。 |
| Hybrid distribution 是否击败 baselines？ | 在 development energy score 上击败 B；不是 Stage-B formal confirmation。 |
| Event selectivity/calibration 是否通过？ | 否；specificity 与泛化 sensitivity 均失败。 |
| History 是否显著？ | 未经 isolation ablation，不能声称。 |
| 5-action chunk 是否显著？ | 未经 isolation ablation，不能声称。 |
| Mixture 是否证明多模态？ | 否；proper score 改善但 H6 未执行。 |
| Lifted hybrid 是否优于显式基线？ | 否；energy 约 9.1，显著更差。 |
| False-safe rate？ | Graph 0.695/0.654，Temporal 0.698/0.684（H1/H3）。 |
| 90% interval coverage？ | Graph 0.925–0.928；Temporal 0.936–0.945。 |
| 最难任务？ | Bowl 的 H3 event prediction；Stove specificity 也很差。 |
| 最终分类？ | Stage-A `action_conditioning_insufficient`。 |
| 最强允许结论？ | 连续 probabilistic response 有开发信号，但当前模型不能选择性预测接触安全性。 |
| Offline scheduler eligible？ | 否。 |
| Latent representation experiment eligible？ | 仅可作为新的 macro-action/trajectory estimand 开发，不是已验证表示。 |
| Latent RL eligible？ | 否。 |

## 14. 最终 disposition

EXP9 完成了协议允许的全部工作，并在正确位置停止。单步/短时连续响应可以由小型
概率模型更好地拟合，但 contact event 泛化仍是联合模型的决定性瓶颈。下一实验应
改变时间尺度与因果抽象，而不是在同一 5-step、per-boundary event target 上扩大网络。

具体 EXP10 建议见 `reports/next_exp_from9.md`。
