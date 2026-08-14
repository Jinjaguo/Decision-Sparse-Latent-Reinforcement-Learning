# EXP1 实验报告：LIBERO 确定性回放与决策稀疏实验前置门禁

## 1. 报告信息

| 项目 | 内容 |
|---|---|
| 实验编号 | EXP1 |
| 实验名称 | Decision Sparsity in LIBERO Demonstrations |
| 执行时间 | 2026-08-13 至 2026-08-14 |
| 执行状态 | 已执行至 Stage E3，并按硬门禁规则停止 |
| 最终门禁状态 | **未通过** |
| 停止原因 | 无扰动回放无法充分复现记录的 MuJoCo 状态轨迹 |
| 最终结果提交前代码版本 | `11854ffd31a8669f8b1e9e9062bbcf6f8515dfde` |
| 报告前最新提交 | `7c61f71` |
| 原始执行规范 | [`experiments/exp1_decision_sparsity/EXP1.md`](experiments/exp1_decision_sparsity/EXP1.md) |
| 项目总纲 | [`PROJECT.md`](PROJECT.md) |

## 2. 执行摘要

EXP1 的目标是验证：在 LIBERO 成功演示中，从精确记录的 MuJoCo 状态出发，
对 Panda 机械臂关节构型施加受控小扰动并继续执行原演示动作，是否能够可靠测量
不同时间点对最终任务结果的敏感性。

该目标存在一个不可绕过的前提：在不施加任何扰动时，记录动作必须能够从记录状态
稳定复现原始状态轨迹。否则，反事实分支中的变化无法区分是人为扰动造成的，还是基础
回放本身的数值偏差造成的。

本次执行完成了：

1. 本机 Conda、CUDA、LIBERO、robosuite 和 MuJoCo 环境审计；
2. LIBERO 任务枚举和三类 pilot 任务的预先冻结；
3. 官方 `libero_goal` 数据下载和完整 HDF5 schema 审计；
4. 依据检出源码实现确定性回放；
5. 在 3 个任务、每个任务 3 条演示上完成最终 E3 门禁运行；
6. 使用隔离的 MuJoCo 2.3.2 环境完成版本兼容性对照。

最终结果呈现出一个重要但不满足门禁的组合：

- 9/9 条无扰动回放最终都完成任务；
- 9/9 条轨迹的初始状态恢复和重复恢复误差均为 0；
- 但 953 个逐步状态比较中，91.50% 的误差超过 0.01；
- 全局状态误差中位数为 0.02679，P95 为 0.47990，最大值为 5.96284；
- 最大偏差主要来自 qvel，尤其发生在接触丰富的 bowl-on-plate 轨迹。

因此，EXP1 的确定性回放硬门禁失败。根据预先规定的停止规则，没有执行 Stage E4
任意时刻分支恢复、Stage E5 关节索引干预或任何 q 扰动 sweep。当前不能提出任何
“决策稀疏”科学结论。

## 3. 研究问题与可证伪条件

### 3.1 最终研究问题

成功操作演示对局部关节构型扰动的结果敏感性，是否集中在少量时间点，而不是沿整条
轨迹均匀分布？

### 3.2 本次实际检验的问题

本次运行没有直接检验决策稀疏性，而是检验其必要前提：

> 从记录的完整 MuJoCo flattened state 恢复后，继续执行记录动作，是否能以足够小的
> 状态误差复现原轨迹？

### 3.3 预登记的 E3 门禁

| 子条件 | 阈值或要求 | 结果 | 状态 |
|---|---:|---:|---|
| 每个任务的演示数 | 至少 3 | 3 | 通过 |
| 覆盖任务数 | 全部 3 个 pilot 任务 | 3 | 通过 |
| 误差有限性 | 全部为有限数值 | 全部有限 | 通过 |
| 最终成功率 | ≥ 90% | 100% | 通过 |
| 重复恢复最大 L2 误差 | ≤ `1e-10` | `0.0` | 通过 |
| 状态回放 P95 L2 误差 | ≤ `0.01` | `0.47990397595761136` | **失败** |
| 动作/状态索引 | 必须由源码确认 | 已确认 | 通过 |

门禁采用多指标判断，同时保留完整原始误差曲线，而不是只输出单一布尔值。

## 4. 软件、硬件与源码溯源

### 4.1 主运行环境

| 项目 | 版本或状态 |
|---|---|
| 操作系统 | Windows 10/11 兼容运行时，`Windows-10-10.0.26200-SP0` |
| Conda 环境 | `libero-exp1` |
| Python | 3.8.20 |
| NumPy | 1.22.4 |
| h5py | 3.11.0 |
| robosuite | 1.4.0 |
| MuJoCo | 3.2.3 |
| PyTorch | 1.11.0+cu113 |
| torchvision | 0.12.0+cu113 |
| torchaudio | 0.11.0+cu113 |
| GPU | NVIDIA GeForce RTX 4090，24,564 MiB |
| NVIDIA 驱动 | 596.49 |
| CUDA toolkit | 12.8 |
| PyTorch CUDA runtime | 11.3 |
| GPU 可见性 | 可见，`torch.cuda.is_available() == True` |
| `pip check` | 通过 |

### 4.2 源码版本

| 代码库 | Revision | 状态 |
|---|---|---|
| 本项目 E3 最终运行 | `11854ffd31a8669f8b1e9e9062bbcf6f8515dfde` | clean |
| LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` | clean |
| robosuite v1.4.0 参考源码 | `fbee5844ff5632f5b5698e204ec5357ca50be0df` | clean |

LIBERO 直接从检出的源码目录导入，没有使用来源不明的已安装 LIBERO 包。任务、BDDL
路径和回放 API 均由该 revision 的运行时代码发现并验证。

### 4.3 Windows 兼容处理

robosuite 1.4.0 的渲染后端选择逻辑在 Windows 上需要 WGL 兼容修补。修补只应用于
Conda 环境中的已安装包，未修改 `third_party/robosuite-src`。补丁保存在：

[`patches/robosuite-1.4.0-windows-wgl.patch`](patches/robosuite-1.4.0-windows-wgl.patch)

同时完成了 robosuite 所需 MuJoCo DLL 的校验复制，并建立 `C:\tmp`。最终 E3 使用
`ControlEnv` 且关闭 renderer 和 camera observations，避免把图像渲染误差引入状态回放。

## 5. 分阶段执行记录

### 5.1 Stage E0：环境与源码审计

最终有效运行：`exp1_e0_libero_exp1_20260813T2250`

完成事项：

- 盘点本机已有 Conda 环境；
- 创建独立环境 `libero-exp1`；
- 安装并核对 LIBERO 指定依赖；
- 验证 CUDA、GPU、PyTorch 和 h5py；
- 固定 LIBERO 与 robosuite 源码 revision；
- 验证 robosuite 版本满足 LIBERO 的 `robosuite==1.4.0` 要求；
- 建立不可覆盖的 run directory 记录格式；
- 记录 Windows WGL、DLL 和 `egl_probe` 兼容处理。

E0 结论：环境与源码审计通过，可以进入任务和数据审计。E0 当时尚未下载数据，
因此 `dataset_root_selected` 为 `null`，属于预期状态。

### 5.2 Stage E1：任务枚举与 pilot 任务冻结

最终有效运行：`exp1_e1_tasks_20260813T2300`

从实际运行的 LIBERO benchmark registry 中枚举到：

| Suite | 任务数 |
|---|---:|
| `libero_10` | 10 |
| `libero_90` | 90 |
| `libero_goal` | 10 |
| `libero_object` | 10 |
| `libero_spatial` | 10 |
| 总计 | 130 |

所有 130 个已枚举任务的 BDDL 路径均存在。registry 中还存在一个无法实例化的
`libero_100` 注册项，运行时产生 `KeyError`；该异常被完整保留，没有把它计入五个
可用 suite。

在查看任何扰动结果之前，冻结以下三个 `libero_goal` pilot 任务：

| Task ID | 任务名 | 结构类别 | 选择理由 |
|---:|---|---|---|
| 0 | `open_the_middle_drawer_of_the_cabinet` | articulated object | 抽屉关节和持续接触 |
| 7 | `turn_on_the_stove` | contact/switch | 开关接触和状态转换 |
| 8 | `put_the_bowl_on_the_plate` | pick-and-place | 抓取、搬运和放置接触 |

三项来自同一 suite，可减少初始下载量，同时覆盖不同物理交互结构。冻结清单：

[`experiments/exp1_decision_sparsity/manifests/selected_tasks_pilot.json`](experiments/exp1_decision_sparsity/manifests/selected_tasks_pilot.json)

源任务清单 SHA-256：
`B19F2BA127D0BE11FC6EC8A35BBEF1C22959583DA950B7C252685BE3E2913352`。

### 5.3 Stage E2：数据下载与 HDF5 schema 审计

下载运行：`exp1_e2_download_libero_goal_20260813T2310`

数据来源由检出 LIBERO 代码声明为 `yifengzhu-hf/LIBERO-datasets`。仅下载
`libero_goal` suite：

- HDF5 文件数：10；
- HDF5 总字节数：6,373,112,875；
- 三个选定任务文件全部存在。

选定文件的大小和哈希：

| 任务 | 文件大小 | SHA-256 |
|---|---:|---|
| drawer | 702,223,367 | `20252C7CF98CD7437061F7F200AE7B6CB6219FABBD53B4536DFAA8ABDA6AB737` |
| stove | 447,509,922 | `387FC10747696B80DEA6ED8D7F2BEAA162BF92AE11750B241073CBD33AAC73D5` |
| bowl-on-plate | 468,246,288 | `E69528B0CF10DFC59B20698E12EC2AFFC03F3887309034D3EB74CAC3EC929406` |

Schema 审计运行：`exp1_e2_schema_pilot_20260813T2320`

审计脚本不预设固定 HDF5 层级，而是遍历全部 group、dataset 和 attributes，然后根据
运行时观察到的 `states` 与 `actions` 识别 episode。结果如下：

| 任务 | Episode 数 | Dataset 数 | 动作长度 min / median / max |
|---|---:|---:|---:|
| drawer | 50 | 600 | 116 / 136.5 / 196 |
| stove | 50 | 600 | 75 / 89.5 / 119 |
| bowl-on-plate | 50 | 600 | 79 / 92 / 126 |

每条 episode 均确认存在：

- `states`；
- `actions`；
- `obs/joint_states`；
- `obs/gripper_states`；
- `obs/ee_states`；
- episode 级 `model_file` 和 `init_state` attributes。

所有 episode 中，`states` 和 `actions` 长度相等。根据 LIBERO 数据生成源码，只有
`action[j]` 到 `state[j+1]`、且 `j < T-1` 的转移具有对应的下一个记录状态。因此最后
一个 action 会执行以评估最终成功，但不会被计入逐步状态误差。

完整 schema manifests 位于：

[`experiments/exp1_decision_sparsity/manifests`](experiments/exp1_decision_sparsity/manifests)

### 5.4 Stage E3：确定性回放硬门禁

最终运行：`exp1_e3_final_3x3_20260814T0020`

#### 5.4.1 回放实现

对每条演示执行：

1. 读取 episode 保存的 `model_file`；
2. 对 XML 中旧的 robosuite/chiliocosm 绝对资产路径进行显式映射；
3. 每个映射后的资产路径必须在本机实际存在，否则拒绝运行；
4. 使用相应任务的已验证 BDDL 构造 `ControlEnv`；
5. 载入 episode XML；
6. 恢复 `states[0]`；
7. 再次恢复同一状态，测量重复恢复误差；
8. 顺序执行全部记录动作；
9. 对每个 `j < T-1` 计算：

\[
e_j = \left\|x^{recorded}_{j+1} - x^{replay}_{j+1}\right\|_2
\]

10. 分别记录 time、qpos 和 qvel 误差；
11. 使用运行时验证的 `env.check_success()` 记录最终任务成功；
12. 保存完整逐步 CSV，而不是只保存汇总值。

动作维数为 7。运行时模型确认 `nq=41`、`nv=37`，flattened state 维数为：

\[
1\text{ (time)} + 41\text{ (qpos)} + 37\text{ (qvel)} = 79.
\]

#### 5.4.2 每条演示结果

| 任务 | Episode | 轨迹长度 | 状态比较数 | 初始恢复误差 | 重复恢复误差 | 最终成功 |
|---|---|---:|---:|---:|---:|---|
| drawer | `demo_0` | 138 | 137 | 0 | 0 | 是 |
| drawer | `demo_1` | 138 | 137 | 0 | 0 | 是 |
| drawer | `demo_2` | 151 | 150 | 0 | 0 | 是 |
| stove | `demo_0` | 80 | 79 | 0 | 0 | 是 |
| stove | `demo_1` | 96 | 95 | 0 | 0 | 是 |
| stove | `demo_2` | 89 | 88 | 0 | 0 | 是 |
| bowl-on-plate | `demo_0` | 90 | 89 | 0 | 0 | 是 |
| bowl-on-plate | `demo_1` | 88 | 87 | 0 | 0 | 是 |
| bowl-on-plate | `demo_2` | 92 | 91 | 0 | 0 | 是 |

#### 5.4.3 全局误差结果

| 指标 | 数值 |
|---|---:|
| Episode 数 | 9 |
| 有效状态比较数 | 953 |
| 最终成功率 | 1.0 |
| 状态 L2 中位数 | 0.02679060442137793 |
| 状态 L2 P95 | 0.47990397595761136 |
| 状态 L2 最大值 | 5.962835299441354 |
| 超过 0.01 的比例 | 0.9150052465897167 |
| 第一时间四分位中位误差 | 0.015005099855476031 |
| 最后一时间四分位中位误差 | 0.062498676849251755 |
| 归一化时间与误差 Pearson r | 0.20614631029124275 |

#### 5.4.4 状态分量误差

| 分量 | 中位数 | P95 | 最大值 |
|---|---:|---:|---:|
| time absolute error | 0 | 0 | 0 |
| qpos L2 error | 0.01537471165633364 | 0.23426339102716812 | 0.32332294049562965 |
| qvel L2 error | 0.017073423126994967 | 0.4756146012201405 | 5.962319702658334 |

时间分量完全一致，说明 action/state 的时钟推进关系没有错一帧。最大误差主要由 qvel
贡献。全局最后四分位的中位误差大于第一四分位，但时间相关系数只有 0.206，说明误差
总体上后期更大，却不是简单的单调累积。

最大误差出现在 `put_the_bowl_on_the_plate/demo_0` 的 action index 67，归一化时间约
0.761。该任务中的多个大误差集中在抓取/放置接触阶段，与接触动力学放大速度差异的
现象一致；这里仅报告相关位置，不把相关性解释为已证明的因果机制。

#### 5.4.5 MuJoCo 版本对照

为了检验主环境的 MuJoCo 3.2.3 是否是主要原因，克隆出隔离环境
`libero-exp1-mj232`，只把 MuJoCo 改为 2.3.2，并重放同一 drawer `demo_0`。

| 环境 | 中位数 | P95 | 最大值 |
|---|---:|---:|---:|
| MuJoCo 3.2.3 | 0.0155143804 | 0.0558489375 | 0.1860132645 |
| MuJoCo 2.3.2 | 0.0155143749 | 0.0558489106 | 0.1860131354 |

差异接近数值舍入量级。因此，2.3.2 与 3.2.3 之间的版本变化不能解释本次观察到的
主要回放偏差。诊断环境被保留，未覆盖主实验环境。

## 6. 门禁失败分析

### 6.1 已排除或显著削弱的解释

1. **初始 state 没有正确写入**：初始恢复误差和重复恢复误差均为 0，排除。
2. **动作与状态差一帧**：时间误差始终为 0，且 indexing 与 LIBERO 源码一致，排除。
3. **个别失败演示污染结果**：9/9 最终成功，但状态误差仍大，排除。
4. **仅由 MuJoCo 3.2.3 升级导致**：2.3.2 对照几乎完全相同，显著削弱。
5. **NaN/Inf 或运行崩溃**：953 个误差全部有限，排除。

### 6.2 尚未解决的可能原因

robosuite 1.4.0 的 `MjSimState.flatten()` 只包含：

- simulation time；
- qpos；
- qvel。

它不包含 MuJoCo solver warm-start 数量、当前 controls 或 Python 控制器内部缓存。
因此，“flattened state 完全相等”不必然代表下一步动力学所需的全部隐含状态相等。

此外，检出的 LIBERO 数据转换代码设置 `cap_index = 5`，在保存公共训练轨迹前删除最初
五个样本，以避开初始 force sensor 不稳定区间。这意味着公共文件的首个动作并不是
原始采集 episode 的第一个控制器动作。公共文件也没有保存被裁剪五步对应的完整控制器
历史。

这些事实给出了合理的 reconciliation 方向，但目前尚未通过对照实验确定真正原因。
因此不能把任何一个可能性写成已证实结论。

### 6.3 为什么 100% 成功仍不能通过

最终任务成功是稀疏结果指标。两条状态轨迹即使在 qpos/qvel 上明显不同，也可能都达到
“抽屉打开”“炉灶开启”或“碗在盘子上”的最终条件。

EXP1 的后续步骤需要比较微小关节扰动对结果的影响。如果无扰动基线自身已经有 P95
0.48、最大 5.96 的状态偏差，那么观察到的 branch difference 可能来自基础回放误差，
而不来自施加的关节扰动。因此，“任务仍成功”不足以满足因果分支实验的内部效度要求。

## 7. 科学结论边界

### 7.1 当前允许陈述

可以陈述：

> 在当前已记录的软件和 Windows 运行环境中，三个 LIBERO pilot 任务的九条演示都能
> 从记录初态通过开环动作完成最终任务，但逐步 MuJoCo flattened-state 轨迹未达到
> 预登记的确定性回放门槛。

### 7.2 当前不允许陈述

不能陈述：

- LIBERO 操作轨迹存在决策稀疏性；
- 某些时间点比其他时间点更关键；
- q 扰动敏感性具有事件对齐结构；
- 潜在动作应该只在少量事件点更新；
- 当前结果支持任何论文级因果结论。

`paper/evidence_table.md` 已把 E3 记录为负的前置条件证据，而不是决策稀疏性的反证。

## 8. 停止决定

EXP1 执行规范明确规定：如果确定性回放不可靠，则在 E3 停止，下一项工作转为
simulator/data reconciliation。

因此本次没有：

- 实现或运行 Stage E4 任意时间分支恢复；
- 执行 Stage E5 Panda arm q-index intervention；
- 运行任何 q perturbation smoke test；
- 启动完整扰动 sweep；
- 根据失败后的结果修改 pilot 任务选择或门槛。

该停止是实验协议的正确执行，不是工程任务遗漏。

## 9. 原始证据与产物

### 9.1 最终 E3 运行

目录：`runs/exp1_e3_final_3x3_20260814T0020/`

核心文件：

- `metrics.json`：门禁、全局结果、分量结果和 episode 结果；
- `artifacts/replay_curves.csv`：953 行完整逐步误差；
- `artifacts/episode_results.json`：逐 episode 结果；
- `artifacts/failure_examples.json`：最大误差和失败样例；
- `artifacts/model_path_rewrites.json`：每个 XML 资产路径的验证映射；
- `config_resolved.yaml`：最终解析配置；
- `environment.txt`：Python 和关键包版本；
- `git_state.txt`：项目与 LIBERO revision；
- `command.txt`：实际执行命令；
- `stdout.log`、`stderr.log`：完整标准输出与警告。

运行目录位于 `.gitignore` 管理的原始结果区，避免把大规模运行产物提交到源码历史；
关键定量结果已经进入本报告、研究日志和 evidence table。

### 9.2 代码与清单

- 回放脚本：[`scripts/exp1/replay_demo.py`](scripts/exp1/replay_demo.py)
- 回放工具：[`src/decision_sparse_rl/envs/libero_replay.py`](src/decision_sparse_rl/envs/libero_replay.py)
- 环境审计：[`scripts/exp1/audit_environment.py`](scripts/exp1/audit_environment.py)
- 任务枚举：[`scripts/exp1/enumerate_tasks.py`](scripts/exp1/enumerate_tasks.py)
- 数据下载：[`scripts/exp1/download_selected_dataset.py`](scripts/exp1/download_selected_dataset.py)
- HDF5 审计：[`scripts/exp1/audit_dataset.py`](scripts/exp1/audit_dataset.py)
- 任务清单：[`experiments/exp1_decision_sparsity/manifests/tasks.json`](experiments/exp1_decision_sparsity/manifests/tasks.json)
- 选定任务：[`experiments/exp1_decision_sparsity/manifests/selected_tasks_pilot.json`](experiments/exp1_decision_sparsity/manifests/selected_tasks_pilot.json)
- 研究日志：[`research_log/2026-08-13.md`](research_log/2026-08-13.md)
- 证据表：[`paper/evidence_table.md`](paper/evidence_table.md)

## 10. 复现命令

以下命令假定工作目录为项目根目录，且 `libero-exp1` 环境已按 E0 记录配置完成。

### 10.1 环境审计

```powershell
C:\Users\Guoji\anaconda3\envs\libero-exp1\python.exe scripts\exp1\audit_environment.py --run-id <new-e0-run-id>
```

### 10.2 任务枚举

任务 manifest 是冻结产物，枚举脚本默认拒绝覆盖。若要复验，应指定新的 manifest 路径：

```powershell
C:\Users\Guoji\anaconda3\envs\libero-exp1\python.exe scripts\exp1\enumerate_tasks.py --run-id <new-e1-run-id> --manifest <new-manifest-path>
```

### 10.3 数据 schema 审计

现有 schema manifests 同样不可覆盖。复验时应指定新目录：

```powershell
C:\Users\Guoji\anaconda3\envs\libero-exp1\python.exe scripts\exp1\audit_dataset.py --run-id <new-e2-run-id> --manifest-dir <new-manifest-directory>
```

### 10.4 确定性回放

```powershell
C:\Users\Guoji\anaconda3\envs\libero-exp1\python.exe scripts\exp1\replay_demo.py --run-id <new-e3-run-id> --episodes 0 1 2
```

run ID 必须是新的单一路径组件。运行目录创建函数拒绝覆盖已有 ID。

## 11. 代码质量与版本历史

最终验证：

- 测试结果：17 passed；
- Git 工作树：clean；
- 所有最终运行使用已提交代码；
- 每个正式阶段有独立、不可复用的 run ID；
- 每个 run 保留配置、命令、环境、Git 状态、stdout、stderr 和 metrics。

本次实验的关键原子提交：

| Commit | 内容 |
|---|---|
| `544ef62` | 初始化 EXP1 研究骨架 |
| `c6bce00` | 环境审计与不可变运行记录 |
| `92b060e` | Windows robosuite 兼容补丁记录 |
| `d2996c2` | LIBERO 源码与 Torch/CUDA 溯源 |
| `55303a4` | E0 通过记录 |
| `50dbcfd` | 源码 API 任务枚举 |
| `82e11d5` | 冻结任务 manifest 与 pilot selection |
| `22c6d92` | 官方数据下载器 |
| `5bb5afa` | 数据下载记录 |
| `6221978` | 通用 HDF5 schema 审计 |
| `1b26ff2` | 固化三个数据 schema manifests |
| `a051950` | E3 确定性回放硬门禁 |
| `5b00d1c` | time/qpos/qvel 误差分解 |
| `11854ff` | 回放运行记录 MuJoCo 版本 |
| `7c61f71` | E3 门禁失败、证据表和停止决定 |

## 12. 建议的下一步：Simulator/Data Reconciliation

下一阶段不是调整门槛，也不是直接执行扰动，而是定位基线误差来源。建议按以下顺序：

1. **单步恢复矩阵**：对多个 `state[t]` 分别直接恢复并只执行 `action[t]`，测量 qpos、
   qvel 和观测误差，以区分累积误差与单步转换误差。
2. **非 flattened 状态盘点**：在运行时记录 `qacc_warmstart`、`ctrl`、actuator state、
   mocap state、userdata 和 controller buffers，确认哪些量会改变下一步转换。
3. **裁剪历史对照**：在可获得原始未裁剪演示时，重放被删除的前五步，再比较公开
   HDF5 的首个保留状态；若原始数据不可得，则明确记录这一数据缺口。
4. **同环境重新采集**：使用当前固定软件环境生成小规模新演示，同时保存足够的
   simulator/controller state，然后立即执行 round-trip replay。
5. **平台对照**：如果新采集数据可确定重放，再考虑 Linux 与 Windows 的同版本对照。
6. **重新开放门禁**：只有新的 E3 运行满足预登记条件，才能进入 E4 和 q 扰动。

## 13. 最终结论

EXP1 已按照执行规范完成到其合法停止点。环境、任务、数据与回放代码均已建立并留下
可复查证据。确定性恢复本身精确，开环动作也维持了 100% 最终成功，但状态级回放误差
显著超过门槛，尤其是接触阶段的 qvel 偏差。

因此本次实验没有提供决策稀疏性的支持或反对证据；它发现的是一个必须先解决的测量
有效性问题。在该问题解决之前，任何 q-space counterfactual sensitivity 结果都可能被
基础回放误差混淆。
