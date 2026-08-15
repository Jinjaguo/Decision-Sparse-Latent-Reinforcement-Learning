# Next experiment from EXP11

日期：2026-08-15
建议实验：**EXP12 — Calibrated Causal Macro-Action Predictive Latent**

## 1. 为什么不是直接做 sparse scheduler 或 latent RL

EXP11 给出了一组混合但非常有用的结果：

- structured action replacement 的 causal macro effect 明确非零；
- P-A/P-B/GRU/graph 都能跨完整 demo 预测 H10/H20 response；
- terminal success-flip 和 coarse-regime consequence 通过各自严格门；
- 但 top-20% branches 只解释 25.6%–37.9% consequence，macro decision sparsity 明确失败；
- full trajectory-distribution route 又被 terminal endpoint P90 失败否决。

因此不应把下一步写成“稀疏决策点调度器”，更不能直接进入 latent RL。协议第 69 条对应的正确动作是：保留 world-action / macro-response modeling，撤销 sparse-decision claim。

同时，EXP11 已满足专门 predictive-latent 开发的最低科学资格：causal macro effect 与 terminal consequence 至少有一条支持。下一步应该回答一个更窄的问题：

> 能否把 pre-state + candidate action chunk 压缩成小型 `z_macro`，同时保留 EXP11 已验证的 trajectory magnitude、terminal flip、coarse regime 和 calibrated uncertainty？

这仍然是 latent engineering，不是 latent RL。

## 2. 与近期工作的关系

以下近期工作只用于设计启发，不替代本项目验证：

- [PACE: Phase-Aware Chunk Execution (2026)](https://arxiv.org/abs/2606.00537) 显示固定 chunk execution horizon 往往 task-dependent 且非单调，并利用 phase/速度结构选择重规划边界。EXP11 的 phase sensitivity 和非稀疏结果与它的“不要用统一 horizon”观察一致；EXP12 可把 phase 作为预测条件，但暂不部署在线 horizon selector。
- [DREAM-Chunk: Reactive Action Chunking with Latent World Model (2026)](https://arxiv.org/abs/2606.18589) 用轻量 latent world model 对多个 candidate chunks 的未来进行想象和选择。它直接支持“先验证 action-chunk predictive latent，再考虑选择”的顺序；本项目应先做离线预测和 calibration，不能跳到 test-time control。
- [Temporal Action Selection for Action Chunking (2025)](https://arxiv.org/abs/2511.04421) 通过缓存多个时间点的 chunks 改善 reactivity/consistency。它提示未来 utility 实验应比较固定、phase-adaptive、risk-adaptive query，但 EXP12 本身不运行控制。
- [STAR: Learning Diverse Robot Skill Abstractions (ICML 2025)](https://openreview.net/forum?id=n1cqQK4hhC) 使用离散 skill abstraction 和 causal skill composition。它支持把 VQ latent 作为候选，但 EXP11 residual audit 没有证明可分离多模态，因此 VQ 只能是与连续 bottleneck 对照的方法，不能预设它会成功。

## 3. 核心 estimand

定义：

```text
z_macro = E(reference physical history, candidate action chunk, task/object graph)
```

预测：

```text
H10/H20 macro-response path
terminal predicate consequence
coarse-regime consequence
execution/clipping risk
pathwise uncertainty / abstention
```

EXP12 不是重建图像，也不是生成 robot action。Encoder 只服务于 causal consequence prediction。

## 4. 必须先解决的数据问题

当前三个任务的 public demo indices 0–49 已被 EXP1–EXP11 的 calibration/formal/历史 cohort 使用。不能再次复用 EXP11 formal demos 来作独立 confirmation。

建议两阶段：

### Stage A：锁定 EXP11 raw 的 development-only latent comparison

- 只读 `exp11_s2_formal_replacements_r1_20260814`；
- 完整 demo nested cross-fitting；
- 不产生 paper confirmation claim；
- 用于筛掉 collapse、过大 latent、欠覆盖模型。

### Stage B：新任务/新 trajectory confirmation

在 LIBERO 中 prospectively 选择至少 3 个与当前操作原语匹配、但未被历史实验使用的任务变体，例如：

- drawer opening 的不同层/方向；
- object-on-support 的不同 object/support；
- articulated switch/button 的不同实例。

具体任务名必须由 BDDL/dataset source audit 决定，不在协议中猜测。每任务先按 EXP5 修正规则收集至少 10 条成功 reference，若某任务不足则继续按升序 episode 补跑；formal confirmation 至少 8 条/任务，development/calibration demos 不得补入 formal。

若现有 demonstrations 无法提供真正未使用且成功的 cohort，则停止 Stage B，报告 `independent_latent_confirmation_unavailable`，不要用旧 demo 充数。

## 5. 建议一次实现的方法

保持“多方法但统一数据和 gate”：

1. **No-latent baselines**：EXP11 P-A、P-B、3-seed GRU、3-seed graph；
2. **PCA/PLS bottleneck**：4/8/16/32 维；
3. **deterministic predictive autoencoder**：仅以 consequence heads 训练，不以重建全状态为主；
4. **variational bottleneck**：β-VAE/IB，检查 KL collapse；
5. **VQ latent**：8/16/32 codes，报告 code utilization/perplexity/dead codes；
6. **graph-latent distillation**：把 EXP11 P-E graph 的 action/object interaction 压到 8/16 维；
7. **temporal latent**：GRU/TCN encoder，对 action chunk 顺序敏感；
8. **multi-head latent**：共同预测 trajectory、terminal、regime、clipping risk；
9. **contrastive causal latent**：同一 state 下把 ±/不同 basis 的 outcome 排序作为辅助目标；
10. **OOD/abstention heads**：task-held-out、basis-held-out、amplitude-held-out；
11. **pathwise split-conformal**：仍以完整 demo 为 calibration unit；
12. **capacity controls**：parameter-matched no-latent MLP、shuffled chunk、shuffled object、latent-0 collapse baseline。

生成模型只有在训练 residual 通过明确的 multimodality test 时启用。EXP11 的两高斯 separation 仅 0.053，因此默认不启用 flow/CVAE mixture；不能把重尾误称为多模态。

## 6. 冻结的数据隔离

建议 outer evaluation 同时包含：

```text
whole-demo 5-fold
leave-one-task-out transfer
held-out temporal family: pulse
held-out amplitude
held-out object/task variant
```

每个 outer fold 内：

- basis/normalization/encoder 只在 fit demos 训练；
- calibration demos 只确定 conformal quantile 和 abstention threshold；
- test demos 只写一次 prediction；
- 3 个冻结 seeds 等权，不挑 seed；
- latent dimension/model family 在 Stage-A development 中选择后锁定，再进入 Stage B。

## 7. 主要 gates

### 7.1 Compact predictive utility

一个 latent 只有同时满足才算成功：

- `dim <= 16`；
- H10/H20 error 不比最终 P-E graph 差超过 5%；
- 相对简单 baseline 的 H10/H20 改善仍 ≥15%；
- demo improvement CI lower >0；
- 90% pathwise coverage 在 `[0.85,0.95]`；
- parameter count 或 inference cost 至少降低 25%；
- 3 seeds 无 collapse，seed-relative SD <5%。

### 7.2 Terminal/regime preservation

- terminal AUROC ≥0.80，demo-cluster CI lower ≥0.70；
- coarse-regime AUROC CI lower ≥0.75，AUPRC ≥0.70；
- sensitivity ≥0.85、specificity ≥0.60、false-safe ≤0.40；
- latent 相对 no-latent head 的 AUROC 降幅 ≤0.02。

### 7.3 Representation quality

- VQ code utilization ≥70%，无 >20% dead codes；
- continuous latent effective rank ≥50% nominal dimension；
- latent-0、shuffled chunk、shuffled object 必须明显变差；
- held-out pulse/amplitude/task 的 ranking rho ≥0.60；
- OOD abstention AUROC ≥0.80，且 coverage 在 accepted subset 中保持。

### 7.4 禁止把 latent 当成 sparsity 证据

EXP12 仍单独复核 top-20% mass 和 rank reliability。除非新独立 cohort 在至少 2/3 tasks 同时达到 EXP11 sparsity gates，否则不得恢复 sparse-decision claim。

## 8. Stop rules

1. 若 Stage A 没有任何 `dim<=16` latent 在性能/coverage/compactness 三者同时通过，停止，不生成 Stage B；
2. 若 latent-0 或 shuffled controls 与候选一样好，标记 `latent_target_collapse`；
3. 若新任务 reference 不足 8 条/任务，只允许 availability-limited pilot；
4. 若 task-held-out 性能崩溃，保留 task-specific predictor，不声称 shared latent；
5. 若 terminal/regime heads 不能复现 EXP11，停止 utility 路线；
6. 若 compact latent 成功但 sparsity 仍失败，结论为 `compact_predictive_but_not_sparse`；
7. EXP12 不运行 policy learning、online selection、MPC 或 RL。

## 9. 后续资格

- 若 compact latent 在独立 cohort 通过，可让 EXP13 做 **outcome-focused offline counterfactual allocation**，比较固定 query、phase-aware query 与 calibrated-risk query；
- 若同时恢复 macro sparsity，才可单独讨论 sparse allocation；
- 只有离线 utility 明确通过后，才有资格设计受限的 online simulator scheduler；
- latent RL 仍至少晚于 utility 和安全门两个阶段。

## 10. 推荐 summary label

EXP12 应在下列标签中选一个，而不是泛化 PASS/FAIL：

```text
compact_causal_macro_latent_confirmed
compact_predictive_but_not_sparse
task_specific_latent_only
terminal_regime_latent_only
latent_target_collapse
latent_no_better_than_explicit_predictor
independent_latent_confirmation_unavailable
```

推荐主线：先证明“宏动作后果可以被紧凑 latent 保真预测”，再讨论“何时使用这个 latent 做决策”。不要把 representation success 提前改写成 RL success。
