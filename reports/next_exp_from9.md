# Next experiment from EXP9

## 推荐：EXP10 — Phase-Conditioned Macro-Action Trajectory Response

EXP9 的 Stage A 分类是 `action_conditioning_insufficient`。这个分类不意味着连续
概率预测完全失败：temporal mixture 相对 Baseline B 的 H1/H3/H5 energy score
改善 42.7%/47.9%/49.3%，且 90% 边际覆盖为 0.936–0.945。失败发生在离散接触
安全性：H1/H3 specificity 约 0.30–0.32、false-safe 约 0.68–0.70，训练侧
sensitivity 规则也没有在 held-out demos 泛化。

因此 EXP10 不应：

- 把 EXP9 的 temporal hidden state 称为已验证 control latent；
- 扩大同一个 per-boundary graph/GRU 模型；
- 直接做 scheduler、MPC、online control 或 latent RL；
- 在没有 retrospective gate 的情况下再采集 17,280 次 q interventions。

建议把因果单位从“单边界小 q 扰动后的 exact-pair event”提升为：

```text
p(
  phase transition,
  contact-regime trajectory,
  task-space trajectory,
  terminal predicate consequence
  |
  recent phase history,
  candidate macro-action chunk
)
```

核心变化是同时改变时间尺度与输出抽象：预测 phase/contact-regime trajectory，
不要求从连续特征精确识别每一个 geom-pair 的下一步保持状态。

## 1. 为什么是这个方向

EXP3–EXP4 已稳定复现“效应随时间不均匀”，但精确时间/方向不普适；EXP5–EXP8
依次否定了粗状态匹配、无条件局部算子、离散 mode+margin、连续 contact-frame
显式 operator 的跨 demo 复用。EXP9 又表明：

- 连续 response distribution 可预测；
- exact named-pair preservation 的选择性风险预测不可行；
- temporal sequence 比 graph set encoder 更好，但没有隔离出 history/action 的
  单独增量价值；
- lifted state×action dictionary 明显失败。

这组证据更像是“正确结构位于较长时间尺度的任务/接触阶段”，而不是“再加一层
contact graph 即可解决单步 geom-pair event”。

## 2. 与近期方法的关系

以下一手资料只提供设计动机，不是本项目成功证据：

- [DreamTrajectory](https://arxiv.org/abs/2608.01381) 用候选 action chunk 引起的
  trajectory 作为世界模型目标，提示 EXP10 应预测任务空间轨迹，而非只预测局部
  pair event。
- [LaWAM](https://arxiv.org/abs/2606.15768) 用 action-conditioned latent future
  feature 代替完整像素重建，提示可以测试紧凑的 trajectory latent；但 EXP10 仍只做
  forward-estimand validation，不连接 policy。
- [HarmoWAM](https://arxiv.org/abs/2605.10942) 区分 predictive 与 reactive stages，
  提示 phase-conditioned experts 可能优于一个统一 per-step head。
- [DyWA](https://openreview.net/forum?id=lUuExXyZmv) 通过历史轨迹适应接触丰富的
  动力学变化，支持把 history 用于 phase/dynamics context；它不能替代本项目的
  demo-level cross-fitting 与 causal q intervention。

这是从文献作出的设计推断，不是对 LIBERO 结果的外推。

## 3. Stage A：仍先使用锁定数据

EXP10 仍应先做 retrospective feasibility，不立即收集新 cohort。

### 3.1 输入 chunks

冻结多尺度 action chunks：

```text
length 5
length 10
length 20
phase-to-next-landmark
```

保留 q intervention，但把它视为 chunk 初始条件的一部分。所有 chunks 来自冻结
reference policy actions，不是 learned macro policy。

### 3.2 Phase targets

从 reference-side variables 预先定义、只用训练数据校准边界：

- approach/free-space；
- first sustained target contact；
- grasp/engagement；
- articulated motion 或 object transport；
- placement/support transition；
- predicate-completion approach。

不要用 task success 或 q response 后验选择 phase 边界。对不适用 task 的 phase 使用
mask，不强迫三任务具有完全相同语义。

### 3.3 输出

- coarse contact-regime sequence：free / transient / sustained target-gripper /
  target-environment / task-object-support；
- EEF/task-object task-space trajectory；
- predicate first-change time 与 terminal consequence；
- signed physical response distribution；
- gap/velocity；
- 只有新数据确实记录 perturbation-side wrench 时才加入 wrench trajectory。

Named geom-pair add/drop 保留为 diagnostic，不再作为唯一 primary safety label。

## 4. 最小模型比较

1. 当前 EXP9 temporal mixture，作为短 chunk baseline；
2. multi-scale TCN/GRU macro-action model；
3. phase-conditioned mixture-of-experts；
4. compact trajectory latent model（例如 16/32 维 latent）；
5. deterministic heteroscedastic unimodal baseline。

模型仍限制在 <0.5M 参数的 small ladder，3 个固定种子，五折整 demo 隔离。

## 5. EXP10 retrospective stop/go gate

至少一个非 oracle 模型必须同时满足：

```text
coarse phase-transition macro AUROC CI lower >= 0.75
phase-transition specificity > 0.55 at sensitivity >= 0.85
false-safe <= 0.45

H5/H10 trajectory energy score
relative improvement >= 15% over EXP9 temporal baseline
demo-cluster CI lower > 0

terminal predicate-consequence AUROC >= 0.75
or explicitly declare that endpoint unsupported

90% marginal coverage in [0.85, 0.95]
```

如果失败，停止新的 LIBERO q-sweep，并把主线转向任务级 trajectory outcome 或寻找
可生成新初始状态/成功 reference 的 validated policy substrate。

## 6. 必需 ablations

- exact pair identity vs coarse physical group；
- phase label vs no phase label；
- chunk 5/10/20；
- no history；
- shuffled action chunk；
- shuffled phase history；
- unimodal vs mixture；
- latent dimension 0/16/32；
- graph features vs temporal-only；
- terminal predicate head removed。

任何增量价值声明仍要求 demo-cluster CI 为正且 BH q<0.05。

## 7. 数据 schema 修复

若 EXP10 retrospective gate 通过并允许新 cohort，新的 per-step raw 必须显式保存：

- perturbation-side contact force/torque；
- matched-zero force/torque；
- deterministic contact-frame transform；
- force validity mask；
- phase label source与 first-crossing index；
- chunk start/end action IDs。

EXP9 已证明现有 EXP8 raw 无法重建完整 perturbation wrench trajectory；不得后验
伪造。

## 8. 确认阶段与主线边界

只有 retrospective gate 通过后，才审计 EXP8 之后剩余的所有 unused demos，并按
EXP9 的 balanced `n=min(10,min qualified)`、最低 8/task 规则建立新 cohort。

即使 EXP10 成功，也最多授权下一次独立的 offline counterfactual decision-allocation
utility experiment。它不直接授权 online scheduler、MPC、VLA 修改或 latent RL。

## 9. 最终建议

最高价值下一步不是“更大的 EXP9”，而是一个可证伪的多尺度 phase/macro-action
retrospective experiment。它保留 EXP9 中真正有价值的连续分布信号，同时绕开已经
反复失败的 exact next-step named-pair safety abstraction。
