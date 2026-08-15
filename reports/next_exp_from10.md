# Next experiment from EXP10

日期：2026-08-14

建议实验：**EXP11 — Structured Macro-Action Replacement Causal Response**

## 1. 为什么必须改变 intervention，而不是继续换 encoder

EXP10 的 A–F 六条路线全部失败。最关键的共同限制不是模型数量不足，而是所有路线仍在回答同一个问题：

```text
在某一个 boundary 注入极小 local-q impulse 后，能否从当前状态与 reference action chunk
预测后续 response？
```

该干预的绝大多数连续目标非常接近零。结果是 0 维常数模型和欠分散 CVAE 能得到很低 energy score，
但 coverage 分别只有约 0.128 和 0.62。与此同时，延长 history/chunk、加 phase、加 MoE、加 latent、加
mixture 都没有稳定提高跨 demo 泛化。

因此 EXP11 应遵循 EXP10 协议第 68 条：**重新定义 intervention 本身**。推荐把原子 q impulse 改为
结构化 action-chunk replacement，直接测试“一个阶段性的动作选择是否改变宏观后果”。研究主线仍然是
decision sparsity：我们不训练 controller，而是测量哪些 phase/boundary 上的宏动作替换具有可重复的因果敏感性。

## 2. 核心研究问题

> 在 reference trajectory 的同一物理状态上，用结构化、有效且成对的替代 action chunk 替换未来
> 5–20 步控制序列，是否产生跨 demonstration 可复用、可校准的 trajectory/terminal consequence？

新的因果对象是：

```text
ΔY = Y(s_t, A_ref + α B_k) - Y(s_t, A_ref)
```

其中 `B_k` 是冻结的时间基，而不是单步 q 方向。候选基：

1. DCT/spline temporal modes：低频平移、旋转、夹爪开合；
2. phase-aligned residual chunks：只从训练 demonstrations 的同 phase action residual 中构造；
3. amplitude-matched ± pairs：每个 mode 同时运行正负替换；
4. matched zero twin：完全相同 state restore 与 continuation，仅 replacement amplitude=0。

这仍是 simulator causal measurement，不是 learned policy、MPC 或 online RL。

## 3. 为什么近期工作支持这个方向，但不替代本项目验证

以下只用于方法设计，不作为 LIBERO 结论：

- [TacForeSight (2026)](https://arxiv.org/abs/2606.11184) 在 contact-rich manipulation 中预测由 wrist
  force/torque 条件化的短时 tactile latent dynamics。它支持 EXP11 在采集时真正记录 perturbation-side
  wrench/contact evolution，而不是像 EXP9/10 那样因 raw 缺失而猜测。
- [ContactWorld (2026)](https://arxiv.org/abs/2606.13877) 的系统比较强调空间结构、时间连续性与跨模态兼容性
  对长时 contact prediction 的重要性。EXP11 可借用“结构化、连续表示”的原则，但不引入昂贵视觉模型。
- [ParticleFormer (CoRL 2025)](https://openreview.net/forum?id=7wGYX11BJB) 使用 action-conditioned 3D
  particle dynamics 处理多物体/多材料交互。它提示对象级状态比扁平全机器人 state 更适合宏动作 response。
- [DyWA (2025)](https://openreview.net/forum?id=flY9V3CfyJ&noteId=flY9V3CfyJ) 联合预测未来状态并从历史适应
  dynamics variation。EXP11 只借鉴其 training-only dynamics context，不直接进入 policy learning。
- [Conformalized Adaptive Forecasting of Heterogeneous Trajectories (ICML 2024)](https://proceedings.mlr.press/v235/zhou24l.html)
  提供整条 heterogeneous trajectory 的 simultaneous coverage 思路。EXP11 应把 pathwise coverage 设为主门，
  避免再次出现低 energy 但欠覆盖的假阳性。
- [World Models That Know When They Don't Know (2025)](https://arxiv.org/abs/2512.05927) 在 robot datasets
  上研究 latent-space calibrated uncertainty/OOD detection。它支持将 calibration 与 predictive fit 分开考核。

这些论文没有证明 action replacement 会在本项目三项任务上成功；它们只说明该设计有可比的技术先例。

## 4. EXP11 分阶段设计

### Stage 0：只读 feasibility/power audit

先不运行新 simulator：

1. 从 reference actions 构造 phase-aligned 5/10/20 步 chunk；
2. 审计每个 phase 的 action residual covariance 和有效秩；
3. 生成 DCT/spline basis，并验证 action bounds、夹爪离散语义和 temporal smoothness；
4. 对训练 demos 做 leave-demo-out basis construction，防止测试 demo residual 泄漏；
5. 估算 effect size、零方差和所需 branches；
6. 若至少两个 task 没有 ≥3 个稳定非夹爪 temporal modes，则停止，不收集新数据。

### Stage 1：小型 calibration cohort

仅在 Stage 0 通过后，使用现有 unused demo 中每 task 2 条进行 calibration：

- 4 reference-only branch landmarks/demo；
- chunk length 5 和 10；
- 3 temporal modes + 1 held-out random smooth mode；
- 2 amplitudes × ± signs；
- 每个 branch 一个 matched zero twin；
- 总替换量约 `3 tasks × 2 demos × 4 branches × 2 lengths × 4 modes × 2 amplitudes × 2 signs = 768`。

Calibration 只允许确定：安全 amplitude、恢复可重复性、wrench/contact logging schema、runtime 和 power；
不能用于 formal claim。

### Stage 2：独立 confirmatory cohort

冻结 Stage 1 后，从完全未使用 demos 中按 `n=min(10,min qualified)`，每 task 至少 8 条。若任一任务不足 8，
只能报告 pilot。预算在看 outcome 前从 Full/Medium/Trajectory-focused 三档选定。

## 5. 必须新增的 raw schema

每个 zero/replacement rollout 每步保存：

```text
reference/restored simulator state hash
executed action and requested replacement action
action saturation/clipping flag
EEF pose/twist
task-object pose/twist
task-specific progress and exact predicate
exact contact identities and coarse regime
contact normal/tangential impulse or force
wrist wrench if simulator/runtime exposes it
gripper state/opening
all-state finite flag
```

若 runtime 不能可靠提供 wrench，允许明确删除 wrench estimand；禁止用 reference force 或 contact count 代替。

## 6. 模型路线：先简单，再有条件启用生成模型

统一比较：

1. zero/phase/task mean；
2. linear functional response on temporal-basis coefficients；
3. object-centric state-space model；
4. switching linear dynamical system with observed coarse regimes；
5. small action-conditioned particle/contact graph；
6. conditional flow/CVAE only if calibration support ≥1,000 effective training replacements and residual multimodality test passes；
7. every probabilistic model receives demo-level split-conformal path calibration。

不再把低 marginal energy 单独当成功。生成模型必须同时通过 pathwise coverage、sharpness、tail error 和
selective-risk gate。

## 7. 主要 estimands

1. H5/H10/H20 trajectory functional response；
2. terminal task-object displacement/orientation and exact predicate flip；
3. coarse regime edit path；
4. integrated wrench impulse（仅在真实可用时）；
5. branch-level macro sensitivity `M(s_t,B_k,α)`；
6. phase × temporal-mode interaction；
7. amplitude convergence与 ± asymmetry。

## 8. 硬门

### Data/causal validity

- zero-twin replay primary physical channels p95 within已冻结 tolerance；
- requested/executed chunk mismatch p95 <1e-7（未饱和通道）；
- saturation rows intent-to-replace 保留并单独标记；
- 每 task 至少 8 formal demos、每 phase/mode 至少 30 paired replacements；
- whole-demo cross-fitting，无 basis/fold leakage。

### Predictive gate

一个 route 只有同时满足才合格：

- H10 与 H20 energy score 相对最佳简单 baseline 均改善 ≥15%；
- demo-bootstrap improvement CI lower >0；
- 至少 2/3 tasks 正向；
- 90% **simultaneous path coverage** 在 [0.85,0.95]；
- prediction-set width 比 unconditional conformal baseline 至少缩小 10%；
- p90 terminal endpoint error 改善 ≥15%；
- 若做 risk selection：sensitivity≥0.85 时 specificity≥0.60，false-safe≤0.40。

### Decision-sparsity signal gate

只有当：

- phase/branch 间 macro sensitivity 的 cross-demo rank reliability ≥0.6；
- top-20% branches 捕获 ≥50% total macro consequence；
- held-out temporal mode 仍保持正向 effect ranking；

才允许下一实验研究 offline decision allocation utility。否则继续禁止 scheduler。

## 9. 必需消融

- q impulse vs action-chunk replacement；
- single-step action vs 5/10/20 chunk；
- DCT/spline basis vs training-only residual basis；
- correct phase vs shuffled phase；
- object-centric state vs full robot state；
- current state vs history-adaptive context；
- marginal Gaussian vs conformalized distribution；
- exact pair vs coarse regime vs terminal-only；
- force/wrench included vs removed（仅数据真实存在时）。

## 10. 预期结果与停止规则

可能结果：

- 若 action replacement 的宏观 effect 可预测且集中在少数 phases，主线重新获得“稀疏决定”的可验证接口；
- 若 effect 可预测但不稀疏，则保留 world-action model，放弃 decision-sparse claim；
- 若 effect 本身可测但跨 demo 不可预测，说明需要 task/object-specific model，不应进入 latent RL；
- 若 zero replay 或 action execution 不可靠，实验在 calibration 停止；
- 若所有 route 仍失败，应收窄论文到已支持的 state-local causal sensitivity，不再继续 latent/control 路线。

## 11. 明确禁止

EXP11 不训练 scheduler、policy、MPC、VLA 或 RL；不使用 EXP10 test outcomes 选择 formal basis；不把 calibration
当 confirmation；不删除 adverse rows；不把 clipped action 当未执行；不伪造 wrench；不因 generative model
energy 好看而忽略 coverage。

## 12. 推荐执行顺序

1. 写 EXP11 protocol 与 source-hash manifest；
2. 只读 action-residual rank/support audit；
3. 冻结 temporal bases、phase landmarks、folds、amplitudes和 gates；
4. 运行 2 demos/task calibration；
5. 先审计 zero/action execution/wrench schema；
6. 通过才审计并冻结 formal cohort；
7. 收集 formal raw 并 hash；
8. 写后锁定所有 model predictions；
9. 先 calibration/path coverage，再 proper score/decision sparsity；
10. 仅在全部硬门后讨论 offline utility，仍不直接进入 latent RL。

EXP11 的核心变化只有一个：从“微小 state impulse 的后果”转为“结构化宏动作替换的后果”。这保持了原研究
主线，同时直接回应 EXP10 已证伪的建模对象。
