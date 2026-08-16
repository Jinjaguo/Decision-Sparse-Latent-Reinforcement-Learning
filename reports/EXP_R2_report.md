# EXP_R2 实验报告：Admissible Pre-Action Feature and Baseline Feasibility Audit

## 1. 实验问题

EXP_R2 检查 EXP_R1 的 route-level benchmark 是否已经包含足够的决策时信息，从而可以公平比较：

1. retrieval-only；
2. scalar verifier/value；
3. factorized consequence selector。

本实验预先禁止使用 post-action physical progress、force、success、safety stop、steps 和 oracle route 作为 selector 输入。如果这些字段缺失或只能在动作后获得，则记录为未来训练 target / outcome label，而不把它们伪装成 online evidence。

## 2. 数据和协议

来源：`runs/exp_r1_s1_same_state_benchmark_20260816_r2`。

- 60 个 branch groups；
- 420 个候选行；
- 7 条 route candidates；
- 30 个 demonstration clusters；
- 没有打开 train/calibration/test split；
- 没有拟合任何模型；
- 没有使用 post-action 信息进行选择；
- 没有新增 simulator rollout。

这是一个前置可识别性审计，不是 selector 的负结果，也不是 independent confirmation。

## 3. 实际可用字段

EXP_R1 当前保存的决策前候选字段只有：

- `initial_estimated_progress`；
- `initial_retrieval_progress`；
- `initial_requested_action` / `initial_executed_action`（仅在确实执行了动作的候选上存在）；
- action norm、clip 和 gripper sign；
- route specification。

可见的 post-action 字段包括 physical progress、force、success、safety stop、steps 和 oracle route。这些字段在 EXP_R2 中全部被视为 forbidden selector inputs。

## 4. 结果

| 检查 | 结果 |
|---|---:|
| candidate groups | 60 |
| state-only 可区分候选的 groups | 40 / 60 = 66.67% |
| candidate action 行可用 | 280 / 420 = 66.67% |
| candidate action 行缺失 | 140 / 420 = 33.33% |
| zero-step candidate rows | 140 |
| route candidates per group | 7 |
| models fitted | 0 |

缺失的 140 行全部来自 zero-step success：恢复后的 Bowl 状态已经满足 predicate，EXP27 按 success-immediately-stop 正确结束，因此没有动作执行记录。这些行不能被删除，也不能补写一个不存在的 action。

## 5. 三类 baseline 的可行性判定

### Retrieval-only

当前不能进行忠实比较。EXP_R1 是 route-level continuation matrix，没有保存产生每个 route candidate 的历史 action-library row identity；因此无法重现“当前状态检索到哪个历史动作”的 retrieval-only 过程。对于 zero-step Bowl groups，当前状态字段在候选之间相同，state-only retrieval 也没有可用的区分信号。

判定：`blocked_as_faithful_comparison`。

### Scalar verifier/value

技术上可以用 route identity 拟合一个 route prior，但那会把 route-level 平均成功率当成 scalar action verification，并混入 zero-step outcome censoring。它不能回答“给定相同 state 和 candidate action，scalar consequence score 是否足够”。

判定：`not_authorized_yet`。

### Factorized consequence

当前的 physical progress 和 force 都是动作执行后的观测。它们可以在未来作为监督 target，但不能在当前 decision time 直接作为输入。若直接用它们训练并在同一行选择，就会产生后果泄漏。

判定：`not_authorized_yet`。

## 6. 解释与失败层级

EXP_R2 没有证明任何模型家族失败。它定位的是：

```text
benchmark rows available
        -> insufficient pre-action candidate context
        -> fair baseline comparison not identifiable
```

当前主要缺口是 instrumentation / schema，而不是 hidden dimension、loss 或阈值。继续训练模型会把 route identity、候选动作是否存在和 zero-step censoring 误解释成 consequence representation 的效果，因此不应进行。

## 7. 需要保留的审计产物

- implementation：`scripts/exp_r2/audit_admissible_features.py`；
- config：`experiments/exp_r2_admissible_feature_audit/configs/exp_r2.json`；
- feature protocol：`runs/exp_r2_s1_admissible_feature_audit_20260816/manifests/feature_protocol.json`；
- group audit：`runs/exp_r2_s1_admissible_feature_audit_20260816/artifacts/group_feature_audit.parquet`；
- field audit：`runs/exp_r2_s1_admissible_feature_audit_20260816/artifacts/field_audit.parquet`；
- feasibility：`runs/exp_r2_s1_admissible_feature_audit_20260816/artifacts/baseline_feasibility.json`；
- metrics：`runs/exp_r2_s1_admissible_feature_audit_20260816/metrics.json`。

## 8. Claim boundary

EXP_R2 支持：当前 EXP_R1 artifact 的 pre-action information 不足以支持公平的三类 baseline 对比；缺失是可定位的 instrumentation 问题。

EXP_R2 不支持：retrieval-only、scalar 或 factorized selector 的优劣；factorization 无效；router 无效；或 closed-loop generalization。

## 9. Disposition

EXP_R2 完成并转入 EXP_R3。下一实验必须补齐：

- corrected-D restore 后、第一候选动作执行前的 current-state snapshot；
- 每个 candidate 的明确 action/chunk specification 和 retrieval row identity；
- zero-step success 的 pre-action state record；
- post-action consequence 作为 label 的独立字段；
- branch/demo-level split and write-once hashes。
