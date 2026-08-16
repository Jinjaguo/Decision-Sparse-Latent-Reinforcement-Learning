# EXP_R1 实验报告：Exact Expert Reconstruction and Same-State Counterfactual Benchmark

## 1. 实验状态与科学问题

EXP_R1 是 EXP12–EXP27 之后新命名空间中的第一个实验。本次实验只完成 Phase A 的基础验证，不训练 selector、不学习 router，也不新增大规模 simulator rollout。

预先规定的问题是：

> 能否从已审计的 EXP27 正式运行中，重建专家机制的精确定义，并建立一个保留失败候选、显式处理 ties、没有把 post-action outcome 当作 selector 输入的 same-state counterfactual benchmark？

本实验的目标是验证基准和审计 substrate，而不是宣称新的闭环性能。

## 2. 计划、实际执行与变更

计划内容：

- 读取 EXP12–EXP27 报告、审计、配置和实现；
- 从 `scripts/exp15/run_recovery_stage.py` 提取 EXP27 专家的实际输入、检索、评分、归一化、阈值、动作处理和 fallback；
- 将 EXP27 formal 的每个 restored branch state 与七条实际路线整理为一个 counterfactual candidate set；
- 保留 failed、unsafe 和 tied candidates；
- 进行 branch denominator、route matrix、future-action isolation、target-demo exclusion、finite-state 和 corrected-D restoration substrate 审计；
- 不训练模型，不在本实验中用后验 outcome 选择候选。

实际执行中发现，Bowl 有 20 个 branch state 在恢复后已经满足 success predicate。因此这 20×7 个 candidate summary 的 `steps=0`，没有对应的 per-step 行。该情况是 EXP27 的 success-immediately-stop 行为，不是数据缺失。修正后的 benchmark 保留这些候选，并将 action/post-action 字段置为 null，绝不伪造初始动作或后果。

第一次构建因错误假设每个 summary 都有 per-step 行而失败，失败目录 `runs/exp_r1_s1_same_state_benchmark_20260816` 保留；修正版先在 `_r1` 完成，随后移除一个不必要的派生 restore token 字段并在 `runs/exp_r1_s1_same_state_benchmark_20260816_r2` 重跑。两次修正只处理 schema/审计表达边界，没有改变候选、标签或结果。

## 3. 精确实现审计

机器可读审计位于：

`runs/exp_r1_s1_same_state_benchmark_20260816_r2/artifacts/expert_implementation_audit.json`

实际实现文件是 `scripts/exp15/run_recovery_stage.py`，执行入口为 `run_recovery_stage.main`，候选 chunk 选择入口为 `choose_chunk(spec, state, library, memory, exclude_episode)`。

核心机制的真实定义如下：

| 论文概念 | 实际实现 | 关键规则 |
|---|---|---|
| default | `C7_physical_default` | physical view；`k=1`；`replan=10`；nearest；distance selection |
| goal | `C0_goal_consequence` | full view；`k=7`；weighted；`smooth=0.15`；`search_k=48`；`consequence_weight=0.75`；按 successor-to-goal distance 评分 |
| progress | `C1_progress_consequence` | full view；`k=5`；monotone；`advance=2`；`search_k=36`；`consequence_weight=0.80`；按 retrieved progress 评分 |
| smooth | `C4_smooth_low` | full view；`k=5`；weighted；`smooth=0.25`；对 action[:6] 与 previous chunk 做平滑 |
| response | `C2_response_alignment` | full view；`k=5`；medoid；`search_k=48`；`consequence_weight=0.65`；按 successor-current 与 current-goal 的对齐评分 |
| soft-force | `V0_default70_soft_goal` 的 Stove task control | force guard 为 `scale`；`guard_fraction=0.55`；`guard_scale=0.25`；超过任务安全阈值时将 action[:6] 缩放到 25% |

共同执行规则包括：动作前六维 clip 到 `[-1,1]`，夹爪维使用 sign，success 后立即停止，连续力阈值超限或绝对 1000 N 触发 safety stop。EXP27 formal protocol 还明确设置了 `target_future_candidate_access=false`、`expert_path_isolated=true` 和 `exclude_target_demo_from_neighbors_and_scale=true`。

## 4. 数据、候选集和隔离

来源均为已消费的 EXP27 formal artifacts：

- candidate run：`runs/exp27_s3_formal_new_time_cascade_20260815`；
- analysis run：`runs/exp27_s4_formal_analysis_20260815`；
- final audit：`runs/exp27_s5_final_audit_20260815`；
- formal config：`experiments/exp27_success_preserving_cascade/configs/exp27.json`。

构建结果：

- 60 个唯一 branch state；
- Drawer / Bowl / Stove 各 20 个；
- 每个 state 7 个候选，共 420 行；
- route matrix 完整；
- failed、unsafe 和 tied candidates 未删除；
- 20 个 default-demand branch 全部具有至少一个 safe candidate；
- candidate oracle safe success 为 100%，default safe success 为 66.67%；
- 20/60 个 branch 存在 default-demand 的 oracle headroom。

该 benchmark 复用了 EXP27 formal 数据，因此不是新的 untouched confirmation set，也不构成独立泛化验证。

## 5. 标签和 selector 信息边界

标签采用保守的 lexicographic key：

```text
(safe_success, not_safety_stop, success)
```

相同 key 保持 tie；`steps` 只作为描述性变量，不用于强行打破标签 tie。这样不会把多个同样安全成功的候选人为排序成“更好”和“更差”。

候选文件：

`runs/exp_r1_s1_same_state_benchmark_20260816_r2/artifacts/candidate_consequences.parquet`

分组文件：

`runs/exp_r1_s1_same_state_benchmark_20260816_r2/artifacts/counterfactual_groups.parquet`

决策前允许使用候选 action/specification 和当前 state/retrieval representation。success、safety_stop、steps、terminal object state、post-action physical progress、post-action force 和 oracle route 都只作为执行后标签或分析变量，不能作为 selector 输入。

## 6. Same-state 与 restoration 审计

以下检查全部通过：

- 60 个 branch ID 唯一；
- 420 行 candidate matrix 完整；
- formal protocol 与 route set 一致；
- EXP27 final audit 已通过；
- target future action isolation 保持为 false access；
- expert path isolation 保持有效；
- target demo neighbor/scale exclusion 保持有效；
- summary finite-state 检查通过；
- 每个 branch 的 corrected-D reference、trajectory state、branch boundary 和 controller snapshot 均存在且形状/有限性有效。

本 EXP 的 restoration 检查是结构性 corrected-D audit，没有重新执行 simulator zero replay。该限制已写入 `artifacts/restoration_audit.json`，因此不能把本实验表述为重新验证了 EXP27 的动力学确定性。

## 7. 主要观察

`strict_pair_fraction_mean = 0.1603`，`tie_pair_fraction = 0.8397`。这说明该 benchmark 的主要困难不是候选缺失，而是大量候选在 safe-success 层面等价。后续 ranking 不能只报告 forced top-1 accuracy；至少需要同时报告：

- tie-aware pairwise accuracy；
- top-1 set selection with tie credit；
- safe-success regret；
- catastrophic selection；
- default-demand recovery；
- candidate-oracle headroom capture；
- demonstration-cluster bootstrap；
- abstention / fallback rate。

EXP27 的 formal route matrix 只能证明每条路线的整段 continuation 结果。它还不能单独证明某个 factorized consequence representation 在决策前已经解释了这些结果。因此 EXP_R1 不做 selector performance claim。

## 8. 结果与解释

EXP_R1 支持：

1. EXP27 的“default / goal / smooth / response / soft-force”可以映射到可审计的具体函数、参数和执行规则；
2. 现有 EXP27 formal artifacts 可以重构为 60×7 的 same-state candidate-set benchmark；
3. 候选集包含足够的 safe-success variation 和 default-demand headroom，适合进行后续内部 matched baseline；
4. ties 是真实的 benchmark 属性，不应被结果分析脚本隐藏。

EXP_R1 尚未支持：

- factorized consequences 优于 retrieval-only；
- factorized consequences 优于 scalar verifier；
- experts 已经被证明具有 measurable specialization；
- state-dependent routing 优于 task-specific cascade；
- 新的闭环泛化或独立 confirmation。

## 9. 产物和校验

- 实现：`scripts/exp_r1/build_same_state_benchmark.py`；
- 配置：`experiments/exp_r1_same_state_consequence_benchmark/configs/exp_r1.json`；
- 实验说明：`experiments/exp_r1_same_state_consequence_benchmark/README.md`；
- benchmark protocol：`runs/exp_r1_s1_same_state_benchmark_20260816_r2/manifests/benchmark_protocol.json`；
- expert audit：`runs/exp_r1_s1_same_state_benchmark_20260816_r2/artifacts/expert_implementation_audit.json`；
- restoration audit：`runs/exp_r1_s1_same_state_benchmark_20260816_r2/artifacts/restoration_audit.json`；
- metrics：`runs/exp_r1_s1_same_state_benchmark_20260816_r2/metrics.json`。

源文件 SHA256 已写入 `metrics.json`，包括 EXP27 candidate summaries、per-step artifact、protocol、branch manifest、analysis metrics、final audit metrics 和 EXP27 config。

## 10. Claim boundary and disposition

EXP_R1 通过了 benchmark validity module，但只完成了 Phase A 的基础设施。它授权 EXP_R2 进行严格匹配的 retrieval-only、scalar verifier/value 和 factorized consequence baselines；不授权直接进入 learned router，也不消费新的 untouched confirmation set。

## 11. 复现命令

```powershell
& 'C:\Users\Guoji\anaconda3\envs\libero-exp1\python.exe' scripts\exp_r1\build_same_state_benchmark.py --run-id exp_r1_s1_same_state_benchmark_20260816_r2
```
