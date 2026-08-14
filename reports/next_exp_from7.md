# Next experiment from EXP7：连续接触几何条件化的局部响应场

建议编号：EXP8

建议名称：**Continuous Contact-Frame-Conditioned Local Response Field**

## 1. 为什么这是最小且必要的下一步

EXP7 已经把问题缩小得很明确：

- 同一个 reference state、双符号保持 exact mode 时，H1 operator 高度收敛；
- held-out direction 的 demo-median 预测通过；
- next-step mode preservation 可预测；
- 但相同 exact mode + 1 mm margin class 并不能跨 demo 复用 operator，H4 改善为 `-0.03736`，CI 完全在 0 以下。

因此，继续只换 matcher、增加 demo 或重复相同 mode label 不会直击失败原因。下一步应测试：**离散 mode 内部的连续接触 manifold 坐标，能否解释并复用局部 operator。**

这仍然是 estimand validation，不是控制实验，不偏离 decision-sparse latent RL 主线。

## 2. 最新相关方法给出的启发

只借鉴表示与验证思想，不直接复制控制系统：

1. [TACTIC (2026)](https://arxiv.org/abs/2607.09218) 使用 contact-centric hybrid predictive model，并通过 contact Jacobian 将学习的动态与解析运动学结合。这支持把接触法向、proximity 和 Jacobian-relative directions 纳入 EXP8 descriptor。
2. [Action-conditioned Face Interaction Graph Networks (2025)](https://arxiv.org/abs/2509.12151) 将几何 interaction graph 与 action 条件显式结合，并同时预测运动与力/力矩。这提示 EXP8 不应只用 mode ID，而应把 exact geom pair、nearest points、相对速度和候选 q direction 形成 action-conditioned contact graph。
3. [Efficient Differentiable Contact Model with Long-range Influence (2025)](https://arxiv.org/abs/2509.20917) 指出接触模型设计会导致梯度突变或消失，并提出更良态的 contact derivative 条件。这提示 EXP8 要把 MuJoCo 原始硬接触 response 与一个仅用于分析的 smooth contact-coordinate baseline 对照，但不能后验平滑正式 outcome。
4. [Koopman global linearization of contact dynamics (2025)](https://arxiv.org/abs/2511.06515) 尝试在 lifted space 中统一跨 contact changes 的分段动态。它适合作为 EXP8 失败后的明确备选方向；在连续 contact-frame 条件化尚未检验前，不应立即跳到全局 Koopman/MPC。

## 3. 核心研究问题

> 在 exact mode 固定且 H1 mode preserved 的条件下，连续 contact-frame descriptor 是否能跨 demonstration 预测局部 q-response operator，并显著优于 exact mode+margin、normalized time、scalar progress 和 EXP5 physical-state descriptor？

## 4. 新独立 cohort

继续使用同一 corrected-D runtime，但不得复用 EXP7 formal demos。

建议按每任务严格升序扫描剩余未使用 episode：

- Drawer：从 demo 31 开始；
- Bowl：从 demo 30 开始；
- Stove：从 demo 30 开始。

目标仍为每任务 10 条成功、finite、exact corrected-D reference。若不足 10 条，遵循当前项目规则继续升序补跑；失败轨迹保留，不打回已经合格的轨迹。

## 5. Reference-side descriptor

每个 branch 在任何 q outcome 产生前记录并冻结：

- exact named geom-pair set 与四类 pair group；
- 每个 active/nearest pair 的两个最近表面点；
- contact normal 与两个 tangent axes；
- signed gap / penetration；
- normal 与 tangential relative velocity；
- contact age（连续保持该 pair 的 reference step 数）；
- MuJoCo reference contact normal/tangent impulse 或 force；
- EEF 相对 target 的 SE(3) pose/twist；
- target 相对 support/task object 的 SE(3) pose/twist；
- arm q/qvel、gripper state/current action、task progress；
- candidate q direction 投影到 contact Jacobian normal/tangent/null components；
- radius。

对没有 active contact 的 pair 使用经过 EXP7 审计的 `mj_geomDistance` nearest-point segment；不能用 body-center proxy。

## 6. 预注册模型

建议同时冻结三个层级，防止只报告最好的一个：

1. **Baseline A：exact mode + margin**，完全复现 EXP7 H4。
2. **Baseline B：continuous state matcher**，复现 EXP5 的 reference-only state distance，并加入 EXP7 signed gap/normal velocity。
3. **Primary：contact-frame local model**：
   - 对 variable-size contacts 使用 permutation-invariant contact graph encoder；
   - edge features 是 pair identity、nearest-point relative coordinates、normal/tangent velocities、impulse 和 action projection；
   - 输出 operator 的低秩 subspace/Gram matrix以及 held-out signed vector；
   - demo-level cross-fitting，训练 demo 与测试 demo 完全隔离。

为了控制复杂度，primary 首先可用 deterministic kernel/attention pooling + ridge regression，不必直接训练大模型。

## 7. 建议假设与门

### H1：跨 demo subspace reuse

在 H1、双符号 preserved-mode、最小 radius pair：

- contact-frame matcher/model 相对 `max(EXP7 mode+margin, time, progress, EXP5-state)` 的 top1 similarity 平均改善 ≥0.15；
- demo-cluster 95% CI 下界 >0；
- 三任务方向一致，至少 2/3 任务单独 CI 不跨 0。

### H2：held-out signed-vector prediction

- demo-median rank rho ≥0.70；
- demo-median vector relative error ≤0.35；
- p90 vector error ≤0.60，专门约束 EXP7 的重尾问题。

### H3：continuous geometry incremental value

对 Baseline B 加入 contact-frame features 后：

- nested demo-cluster permutation test BH q<0.05；
- improvement CI lower >0；
- 删除 force/impulse、nearest points、Jacobian projection 的预注册 ablation 都要报告。

### H4：horizon locality

同时报告 H1/H3/H5/remaining 的 intent 与 conditional：

- 允许 H1/H3 通过、remaining 失败；
- 不允许只报告 survivor；
- 至少报告 coverage-adjusted score，例如 `preservation_rate × conditional_top1`。

### H5：选择性 mode-risk gate

在 EXP7 predictor 上增加 constrained threshold：

- demo-cross-fit AUROC CI lower ≥0.75；
- ECE≤0.05；
- specificity≥0.70；
- sensitivity≥0.85；
- 可增加 demo-cluster conformal risk bound；
- threshold 只从 training folds 冻结。

## 8. 规模与半径

保留 EXP7 已验证的三个 radius：0.0003125、0.000625、0.00125。

建议先做 outcome-blind support audit：

- 30 demos；
- 每 demo 12 branches；
- 360 branches；
- 若计算预算允许，仍为 7+1 directions、双符号、17,280 interventions；
- 若预算受限，只能在 formal freeze 前用 reference-only power/runtime audit 决定减少 branch 数，不能看 q outcome 后删减。

## 9. 必须改进的统计设计

- H1/H2/H4 明确限定 preserved-mode conditional，同时保留 intent-to-perturb；
- demo 无 preserved support 时，主门保守计失败；
- cross-demo matching 禁止同一 demo 泄漏；
- predictor bootstrap 必须按 demo cluster；
- 除均值/中位数外报告 p90/p95，避免 EXP7 H3 重尾被中位数掩盖；
- margin 不再只用 1 mm 二元 bin，改用连续 signed gap spline/standardized distance；
- task-specific near-boundary support 在 formal outcome 前审计，若不足则停止 H2 的 task-general claim。

## 10. Stop / go 决策

### 若 EXP8 H1+H2+H3 通过

才进入“operator/risk-aware scheduler 的离线反事实评估”，仍先不做 online control。验证 scheduler 是否能在保持 task success 的同时减少 intervention/decision count。

### 若连续 contact-frame model 仍未通过跨 demo reuse

停止把局部 operator 当作可共享显式对象。下一条主线改为：

- lifted hybrid dynamics / Koopman representation；或
- action-conditioned contact graph forward model；
- 目标改为预测 response distribution/risk，而不是复用单一 Jacobian-like operator。

### latent RL 条件

EXP8 之前仍不满足。至少需要：

- 可跨 demo 复用的连续 contact-conditioned representation；
- 选择性合格的 risk gate；
- 离线 scheduler utility 改善；
- 新 cohort replication。

## 11. 推荐的 EXP8 一句话目标

> 在新的独立 LIBERO cohort 上，检验连续 contact-frame geometry 与 action projection 能否把 EXP7 的“同状态 H1 局部收敛”提升为“跨 demonstration 可复用、重尾受控的局部响应场”，并同时验证具有高 specificity 的 next-step mode-risk gate。
