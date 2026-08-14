# 基于 EXP1 的下一实验建议

## 1. 文档信息

| 项目 | 内容 |
|---|---|
| 来源实验 | EXP1 |
| 来源报告 | [`exp1_report.md`](exp1_report.md) |
| 建议实验代号 | **EXP1-R：完整积分状态的成对反事实分支校准** |
| 性质 | EXP1 的 reconciliation / validation 子实验，不进入后续 latent-RL 阶段 |
| 优先级 | 最高；当前主线的必要前置步骤 |
| 文献检索日期 | 2026-08-13（America/New_York） |
| 是否允许直接执行 q 扰动 | 否；必须先通过本文规定的 R1–R3 门禁 |

## 2. 一句话建议

下一次不要继续尝试让公共 LIBERO 的 79 维 `time + qpos + qvel` 状态精确复现原始
历史，而应在同一固定运行时中重新生成一条无扰动 reference rollout，逐步保存 MuJoCo
`mjSTATE_INTEGRATION` 和控制器历史，然后从同一完整快照产生“零扰动 twin”与“q 扰动
twin”；只有两个零扰动 twin 能近似逐位一致时，才重新开放决策关键性测量。

## 3. 为什么这是最接近研究主线的下一步

EXP1 的核心科学问题没有失效：不同时间点的局部构型干预是否产生高度不均匀的结果
敏感性。失效的是当前因果比较的基础设施——公共 HDF5 中保存的 flattened state 不足以
支持可靠的历史分支。

EXP1 已经证明：

- 记录的 `states[0]` 可以精确写回；
- 同一状态重复写回的误差为 0；
- 记录动作在 9/9 条轨迹上仍能完成最终任务；
- 但逐步状态误差 P95 为 0.4799，91.5% 的比较超过 0.01；
- 误差以 qvel 为主，最大值出现在接触丰富阶段；
- MuJoCo 2.3.2 与 3.2.3 的对照几乎相同。

因此，最小且科学上合理的修复不是更换 benchmark、训练一个大模型或放宽阈值，而是把
反事实比较改成同一运行时内的 matched-twin design。这个设计仍然回答原来的因果问题，
只是把比较基线从“另一平台生成的历史轨迹”换成“当前固定环境中同时生成的无扰动
孪生分支”。

## 4. 关键技术依据

### 4.1 当前 79 维状态不是完整动力学输入

robosuite 1.4.0 的 wrapper 只把 simulation time、qpos 和 qvel 拼成 flattened state。
它没有保存 solver warm-start、control、applied forces、mocap、userdata 或其他运行时量。

MuJoCo 官方文档明确把 `mjSTATE_INTEGRATION` 定义为 forward dynamics 的完整输入集合，
并指出从非初始状态恢复时，如果要求完美数值复现，`qacc_warmstart` 可能非常关键。
官方还说明：两个 `mjData` 如果 integration state 相同，其 dynamics pipeline 输出应当
一致。[MuJoCo Simulation / Integration state](https://mujoco.readthedocs.io/en/latest/programming/simulation.html#integration-state)

当前 MuJoCo API 中，`mjSTATE_INTEGRATION` 包括 full physics、user state 和 warm-start；
其中包含 time、qpos、qvel、act、ctrl、applied forces、equality activation、mocap、
userdata、plugin state 与 warm-start 等组成部分。
[MuJoCo `mjtState` API](https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html#mjtstate)

这给出了比“继续猜测 qpos/qvel 之外缺什么”更直接的实验路径：先使用官方定义的完整
integration state，再通过 ablation 判断 Python 控制器历史是否仍需额外保存。

### 4.2 历史信息可能是操作控制的有效状态组成

近期机器人学习研究也强调当前观测未必构成充分 Markov state。AEM 使用 action-effect
history 学习紧凑的时间表示，并报告 history-aware pretraining 优于单帧与直接堆帧；
这不直接证明本项目的 controller cache 是误差来源，但支持把历史状态作为需要实证排查
的变量。[Action-Effect Memory, 2026](https://arxiv.org/abs/2606.12499)

KEMO 则使用运动学和视觉变化检测任务事件，仅保留与任务状态变化相关的 keyframes，
并把关键帧用于长期操作记忆和训练加权。它与本项目“少量事件点承担主要决策作用”的
研究方向高度一致，但更适合作为现象确认后的模型设计参照，而不是替代当前的因果测量。
[KEMO, 2026](https://arxiv.org/abs/2606.23589)

### 4.3 干预必须保持因果一致性

RoCoDA 的核心原则是只修改因果无关部分，或按照几何等变关系同时变换状态和动作；其
LIBERO 实验说明反事实增强需要尊重任务中的因果依赖。对本项目而言，这强化了一个设计
约束：q 扰动分支必须保留所有非干预 integration/controller state，不能把恢复误差混入
处理效应。[RoCoDA, 2024](https://arxiv.org/abs/2411.16959)

### 4.4 不应过早改变动作抽象

2026 年关于机器人 action space 的大规模研究把 temporal abstraction 与 spatial
abstraction 分开分析，并强调动作参数化、action chunking 和控制接口会显著影响性能。
这提示我们在 reconciliation 阶段应固定现有 OSC delta-action 接口，不要同时更换为
joint action、absolute action 或 learned latent action，否则无法确定改进来自 state
恢复还是 action representation。[Demystifying Action Space Design, ICLR 2026](https://openreview.net/forum?id=nAOgwZ9Ymj)

## 5. 推荐实验：EXP1-R

### 5.1 核心研究问题

1. 保存并恢复 MuJoCo `mjSTATE_INTEGRATION` 是否足以实现同一运行时内的确定性分支？
2. 如果不足，额外保存 robosuite controller 与 robot buffers 是否能够消除差异？
3. 在零扰动 twin 可靠后，q 扰动相对于 matched zero twin 的处理效应是否仍随时间高度
   不均匀？

前两个问题是本次实验的硬门禁。第三个问题只允许进行极小 smoke test，不做完整 sweep。

### 5.2 假设

#### H-R1：完整积分状态假设

仅保存 `mjSTATE_INTEGRATION`，即可显著降低从中间时刻恢复后的单步和多步状态误差，
尤其是 qvel 误差。

#### H-R2：控制器历史假设

若 H-R1 仍有误差，则 integration state 加上显式 controller/robot buffer snapshot 会比
integration-only 恢复更可靠。

#### H-R3：成对差分可识别性假设

当两个分支从同一完整快照和同一 controller snapshot 出发时，零扰动 twin 的差异应
接近机器精度。此时 perturbation twin 与 zero twin 的差异才可解释为局部 q 干预效应。

#### H-R4：主线现象假设（仅在门禁通过后测试）

固定幅值 q 扰动的处理效应在时间上非均匀，并在接触、抓取、释放或任务状态转换附近
增大。

## 6. 实验设计概览

```text
public HDF5 state[0] + recorded actions
                  |
                  v
      local reference rollout（当前固定运行时）
                  |
      每个 policy step 保存完整 runtime snapshot
                  |
          +-------+----------------+
          |                        |
          v                        v
  zero twin A                zero twin B
  同动作 continuation        同动作 continuation
          |                        |
          +---- 必须近似一致 ------+
                  |
             门禁通过后
                  |
          +-------+----------------+
          |                        |
          v                        v
  zero matched control       q-perturbed twin
          |                        |
          +---- paired effect -----+
```

关键变化是：不再把公共 HDF5 的未来 `states[j+1:]` 当作每个分支必须复现的唯一轨迹。
公共数据仍提供任务、初始 scene、成功动作与原始示范语义；因果 effect 则在当前固定运行时
内由成对分支估计。

## 7. 实验条件与 ablation

对同一个本地 reference snapshot，比较以下恢复条件：

| 条件 | MuJoCo 状态 | Python 控制器/robot buffers | 目的 |
|---|---|---|---|
| A：Legacy | time + qpos + qvel | 不保存 | 复现 EXP1 失败基线 |
| B：Full physics | `mjSTATE_FULLPHYSICS` | 不保存 | 区分基础 physics state 的作用 |
| C：Integration | `mjSTATE_INTEGRATION` | 不保存 | 检验 warm-start、ctrl、forces 等完整输入 |
| D：Integration + controller | `mjSTATE_INTEGRATION` | 显式保存并恢复 | 检验 Python 控制历史 |
| E：Full `mjData` diagnostic | `mj_copyData` 或等价完整副本 | 显式保存并恢复 | 仅用于定位 C/D 仍失败的情况 |

条件必须按 A→B→C→D 顺序实现和测试。E 只作为诊断上界，不应直接成为默认数据格式，
因为它体积大且可能绑定具体 MuJoCo revision。

### 7.1 Controller snapshot 规则

不能通过猜字段名实现。必须在运行时和 robosuite 1.4.0 源码中枚举并记录：

- OSC controller 的 goal position/orientation；
- interpolation state（当前配置虽预期为 `None`，仍需断言）；
- robot recent action/qpos/torque/EEF buffers；
- gripper controller state；
- 任何会参与下一次 `env.step()` 的 mutable NumPy array 或 scalar。

每个字段必须保存：对象路径、类型、shape、dtype、恢复前后哈希或数值误差。不能使用
不透明的全对象 pickle 作为唯一证据。

## 8. 数据与采样计划

### 8.1 任务

沿用 EXP1 冻结的三个任务，不重新挑选：

1. `open_the_middle_drawer_of_the_cabinet`；
2. `turn_on_the_stove`；
3. `put_the_bowl_on_the_plate`。

保持同一任务集合可以把 EXP1-R 与 EXP1 的误差直接比较，避免 selection after results。

### 8.2 演示

第一阶段沿用 `demo_0`、`demo_1`、`demo_2`，共 9 条。若门禁通过，再扩展到每个任务
10 条演示；完整 q 扰动实验仍需另行预登记。

### 8.3 Branch times

每条轨迹先使用 12 个不依赖结果的时间点：

- 10 个等分位点：5%、15%、…、95%；
- 第一次 gripper command 符号变化后的一个点；
- 最大 contact-count 变化的第一个点。

如果事件点与等分位点重复，使用最近尚未选中的时间点补齐。事件检测只使用 reference
rollout，不读取扰动结果。所有最终 branch indices 在运行任何 perturbation 前写入冻结
manifest。

## 9. 分阶段执行方案

### R0：运行时状态 API 审计

实现建议：

```text
scripts/exp1/reconcile/audit_runtime_state.py
src/decision_sparse_rl/envs/mujoco_snapshot.py
```

必须记录：

- `mujoco.__version__`；
- `mujoco.mjtState` 中可用 bit flags；
- `mj_stateSize(model, mjSTATE_INTEGRATION)`；
- legacy flattened state 与 integration state 的字段和维数差异；
- `sim.model.nq/nv/na/nu/nbody/nmocap/nuserdata`；
- controller/robot mutable state manifest；
- 每个 snapshot 字段的序列化和恢复 round-trip error。

R0 不运行任务扰动。

### R1：本地 reference rollout 生成

实现建议：

```text
scripts/exp1/reconcile/generate_local_reference.py
```

对每条选定演示：

1. 使用公共 episode XML 和 `states[0]` 初始化；
2. 执行记录动作；
3. 在每个明确的 pre-action boundary 保存：
   - legacy flattened state；
   - full physics state；
   - integration state；
   - controller/robot snapshot；
   - action；
   - task success；
   - contact pairs/count；
   - q、gripper、EEF pose；
4. 保存 reference rollout 的完整本地 future states；
5. 验证最终任务仍成功。

边界必须定义为“调用该 policy action 的 `controller.set_goal` 之前”。如果实际 robosuite
调用顺序不同，应以源码和 instrumentation 结果为准，并在 manifest 中改写精确定义。

### R2：零扰动 twin 门禁

实现建议：

```text
scripts/exp1/reconcile/validate_twin_restore.py
```

对 A–D 每个恢复条件和每个 branch time：

1. 恢复同一 snapshot；
2. 执行相同 continuation actions，得到 zero twin A；
3. 再次恢复同一 snapshot；
4. 执行完全相同 continuation actions，得到 zero twin B；
5. 比较每一步 integration state、legacy state、q、EEF、gripper、contact 和 success；
6. 重复至少 3 次，以检测运行顺序或未清理缓存造成的差异。

这里最重要的比较是 twin A 对 twin B，而不是先比较公共历史 future state。

### R3：恢复条件选择

选择满足门禁且保存状态最少的条件。预定优先级：C > D > E。

- C 通过：使用 integration-only；
- C 失败而 D 通过：使用 integration + controller；
- D 仍失败：运行 E 诊断并停止，不进入扰动；
- A/B 的结果只用于解释误差来源，不决定主方法。

### R4：最小 q 扰动 smoke test

只有 R2/R3 通过后才能运行。

建议先使用：

- 每个任务 1 条演示；
- 每条演示 4 个冻结 branch times（早、中、晚、事件点）；
- 每点 2 个随机单位方向；
- 每方向 `+epsilon` 与 `-epsilon`；
- 一个 `epsilon`，建议从 Panda 关节范围的 0.5% 开始；
- 每个 perturbation 配一个从相同完整快照恢复的 zero matched control；
- 仅修改运行时验证的 Panda arm qpos indices；
- 非干预 integration state 在干预瞬间必须逐元素保持一致。

R4 只回答“paired effect 是否可测且大于 zero-twin noise”，不回答完整决策稀疏性假设。

## 10. 指标

### 10.1 状态恢复指标

- full integration-state L2 / L∞ error；
- qpos、qvel、warm-start、ctrl 分量误差；
- controller snapshot field-wise error；
- 单步 error；
- 5、10、25 和 full-horizon cumulative error；
- bitwise equality fraction；
- 每个 branch 的最大首次分歧 timestep。

### 10.2 任务和几何指标

- final success agreement；
- reward agreement；
- Panda joint-position difference；
- EEF position/orientation difference；
- gripper difference；
- contact pair symmetric difference；
- task predicate agreement。

### 10.3 成对干预指标

令 zero matched branch 的 outcome 为 `Y_t^0`，q 扰动 branch 为
`Y_t^{delta}`：

\[
\Delta_t = d\left(Y_t^{\delta}, Y_t^0\right).
\]

同时记录：

- success flip；
- terminal object-pose error；
- maximum future q/EEF divergence；
- recovery time；
- zero-twin noise floor；
- standardized effect：`Delta_t / (zero_noise + epsilon_num)`。

只有 effect 明显高于 zero-twin noise floor 时，才能称为 perturbation sensitivity。

## 11. 硬门禁

### 11.1 R1 reference gate

- 9/9 本地 reference rollouts 最终成功；
- 无 NaN/Inf；
- 所有 snapshot shapes 在同一 task/model 内一致；
- 每个 snapshot 能完成即时 save→restore round trip；
- controller manifest 不存在未分类的 mutable control field。

### 11.2 R2 zero-twin gate

对最终选用的恢复条件：

- 9 条演示全部覆盖；
- 每条 12 个 branch times；
- 每点 3 次重复；
- 总计至少 `9 × 12 × 3 = 324` 个 zero-twin comparisons；
- final success agreement = 100%；
- integration state 所有数值有限；
- integration-state L2 median ≤ `1e-10`；
- integration-state L2 P95 ≤ `1e-8`；
- integration-state L2 max ≤ `1e-6`；
- terminal object-pose L2 P95 ≤ `1e-6`；
- 不存在与 branch time 或 contact event 系统相关的 zero-twin error spike。

这些阈值应在首次正式 R2 运行前写入 config 并提交。若初始 instrumentation 表明官方
API 在当前精度下有稳定的非零舍入下界，只能通过一个独立 calibration run 修改一次，
且必须保留原阈值和修改理由，不能根据 perturbation 结果调整。

### 11.3 R4 扰动可测性门禁

- 所有 `delta=0` matched controls 通过 R2 阈值；
- 干预瞬间非 arm-q integration components 的 L∞ error ≤ `1e-12`；
- 扰动后 state 有限且 joint limits 有效；
- 至少一个非零 perturbation effect 超过对应 zero-noise P99 的 10 倍；
- effect 与 zero-twin restoration error 的相关性不显著或效应量很小；
- 不根据 outcome 手工删除 branch。

若 R4 失败，只能说明当前 epsilon 或 effect metric 不可测，不能说明不存在决策稀疏性。

## 12. 统计分析计划

### 12.1 Reconciliation

对 A–D 条件报告：

- 每个任务和全局的 median/P95/max；
- bootstrap 95% confidence interval；
- qpos/qvel/warm-start/ctrl 分解；
- 接触与非接触 timestep 分层；
- 误差相对 EXP1 legacy baseline 的倍率下降。

关键比较不是只看均值，而是要求最坏情况也满足门禁，因为单个异常 branch 就可能生成
虚假 critical timestep。

### 12.2 最小干预

R4 仅绘制每条轨迹的 paired effect curve，并同时叠加 zero-noise band。不得在 smoke
test 阶段报告全局“稀疏性指数”或论文级显著性。

## 13. 预期结果与决策树

### 情形 A：Integration-only 通过

这是最佳结果。说明 EXP1 失败主要来自 legacy flattened state 不完整。后续使用官方
`mjSTATE_INTEGRATION` 重新开放 branch restoration，并保留 public demonstrations 作为
动作与任务来源。

### 情形 B：Integration + controller 才通过

说明 MuJoCo 完整积分状态仍不是 robosuite policy-step 的完整状态。后续 snapshot schema
必须版本化保存 controller fields，并增加字段缺失断言。该方法仍保持主线不变。

### 情形 C：只有 full `mjData` 同进程副本通过

说明可移植序列化困难，但 paired in-memory branching 仍可能可行。此时 EXP1 可改成同一
进程内即时 twin branching，不宣称跨进程或跨机器复现。

### 情形 D：完整 `mjData` 与 controller 仍不通过

停止直接 simulator restore 路线。转向下面的备选方法 1，不运行 q perturbation。

## 14. 当前方法仍不可行时的备选方法

### 备选方法 1：Prefix replay + matched branching（首选替代）

不从中间状态直接恢复。对每个 branch time `t`：

1. 从 episode 初始模型和状态重新开始；
2. 完整执行 actions `0...t-1`；
3. 创建两个当场内存副本或并行环境；
4. 一个不扰动，一个施加 q 扰动；
5. 执行相同 future actions并比较。

优点：重建了 controller 和 solver 历史；仍是精确的 paired causal intervention。
缺点：计算量约为 O(T²)，且需要验证两个环境在 fork 前完全一致。对于当前 3×3 pilot
规模仍可接受，可通过 checkpoint cache 降低成本。

### 备选方法 2：本地重新采集 + closed-loop recovery policy

如果 recorded open-loop actions 在更大样本上无法维持成功，则使用相同 LIBERO tasks
训练或采用一个固定 closed-loop imitation policy，并在本地 rollouts 中做 matched twin
干预。policy 必须对 zero 和 perturbation branches 使用完全相同的冻结权重与随机种子。

这一变化会把研究对象从“某条人类演示的 open-loop continuation”改成“固定闭环策略的
决策敏感性”，仍与决策稀疏主线相容，但论文 claim 必须相应改写。AEM/CDP 一类
history-aware policy 可以作为后续模型候选，不能在 reconciliation 阶段与状态修复同时
引入。

### 备选方法 3：模型化反事实（最后选择）

训练带不确定性的局部 dynamics/world model，估计 q intervention 的短期 effect，并只在
模型置信度高的区域使用。这接近 model-based counterfactual augmentation，但因模型误差
可能与所谓 criticality 混淆，必须用 simulator paired branches 校准，不能作为第一选择。

### 不建议的替代

- 直接把 0.01 门槛放宽到 0.5；
- 只比较 final success 而忽略状态误差；
- 跳过 E3/E4 直接训练 latent RL；
- 根据当前最大误差位置挑选“关键时间点”；
- 同时更换 simulator、控制器、动作表示和任务集；
- 把 KEMO 等事件关键帧方法的成功当作本项目因果假设的直接证据。

## 15. 与长期 latent-RL 主线的连接

EXP1-R 不是旁支工程。它会决定后续 latent action 应该建立在哪一种状态和时间抽象上：

- 如果完整 integration state 足够，q-grounded latent transition 可以明确绑定物理快照；
- 如果 controller/history 必需，latent state 必须包含 action-effect memory，而不能只编码
  当前视觉或 q；
- 如果 critical effects 与接触事件对齐，可把 KEMO 式事件检测作为 later-stage learned
  boundary proposal；
- 如果 sensitivity 并不稀疏，则项目应放弃“少量决策点”强假设，转向连续但自适应的
  temporal abstraction。

因此，该 reconciliation 实验既修复当前测量，也为后续 latent state/action 设计提供
可证伪的结构依据。

## 16. 建议产物

下一实验完成时至少应生成：

```text
reports/
├── exp1r_report.md
└── next_exp_from1r.md

experiments/exp1_decision_sparsity/manifests/
├── runtime_state_schema.json
├── controller_state_schema.json
└── branch_times_reconciliation.json

runs/<run_id>/artifacts/
├── reference_snapshots_manifest.json
├── zero_twin_comparisons.parquet
├── component_errors.parquet
├── failure_examples.json
└── plots/
    ├── restore_condition_comparison.png
    ├── zero_twin_error_by_time.png
    └── contact_stratified_error.png
```

原始 snapshot arrays 可以使用 HDF5/NPZ 存在 ignored data/run 目录，但 schema、hash、
统计和失败样例必须进入可版本控制的报告或 manifest。

## 17. 最终建议

优先执行 EXP1-R，而不是开始新的算法实验。第一目标是比较 legacy、full physics、
integration 和 integration+controller 四种恢复条件；第二目标是建立零扰动 matched-twin
noise floor；只有两者通过后，才运行极小 q perturbation smoke test。

如果 integration-state matched twins 能复现，则项目可以基本不偏离原始 EXP1，只需把
状态恢复层升级为 MuJoCo 官方完整积分状态。如果仍不能复现，则采用 prefix replay +
matched branching。这两个方案都保留“在成功操作轨迹的不同时刻进行局部物理干预”的
原研究问题，比立即转向纯相关性的关键帧检测或 latent-policy 训练更有科学连续性。
