# EXP12 实验报告：Offline Counterfactual Action-Consequence Ranking

## 1. Experiment question

给定同一 corrected-D 状态下的多个候选动作块，轻量的 Contact、Motion、Outcome 与 Uncertainty 预测是否能支持真正的候选动作排序，而不需要预测完整未来世界？

## 2. Why this EXP exists

EXP11 证明了结构化 action replacement 的宏观后果可预测，但没有证明这些预测能用于选动作。EXP12 是从“预测分析”到“动作选择效用”的第一座桥，只做离线 counterfactual ranking，不运行在线控制。

## 3. What changed from EXP11

实验单位从单个 replacement 的预测误差改成 same-state candidate set。每个 branch 显式加入成功 reference continuation 作为 nominal，并分别定义 terminal outcome、reference-contact preservation、task-native motion progress 与 success-first composite preference。比较四种输入编码、共享/独立 specialist、五类 loss、十三种 coordinator/ablation，以及一个输出 620 维未来轨迹的 richer-future baseline。

## 4. Frozen protocol

控制文件位于 `experiments/exp12_action_consequence_ranking/`。主分析仅使用通过 EXP11 独立 fidelity re-audit 的 task×family；Bowl I-A 因 17.5% clipped chunks 被排除，仅保留 intent-to-replace 诊断。完整 demo 是 train/calibration/test 隔离单元，21 demos 按 identity 排序 round-robin 分成 5 folds。Pair tie tolerance 和 nominal opportunity gap 均冻结为 0.05。

R2 是相对成功 reference 的 coarse-contact preservation proxy，不被解释成“所有 contact change 都有害”。R4 composite 只生成监督与评估标签，从不作为 test-time 输入。

## 5. Data/cohort

数据锁：`exp12_a0_candidate_dataset_r1_20260815`。

- 原始 rows：1,316，包括每组 nominal；
- fidelity 主集：1,036 candidates、84 groups、21 demos；
- 56 groups 有 17 个候选；Bowl 的 28 groups 各有 3 个候选；
- 所有主组至少有 3 个候选；
- source raw hashes 全部复核通过。

Oracle opportunity 是本实验最重要的数据结果：只有 10/84 groups 的 oracle-best 比 nominal 高超过 0.05，且 10 个全部属于 Stove。Drawer 与 Bowl 均为 0。当前候选集因此几乎没有跨任务改进空间。

## 6. Models/encodings/losses

编码比较包括 raw compact physical、object-centric、contact-aware 与五步 history。E5 对 contact/motion/outcome 分别拟合，E6 用 13,251 参数的共享三头 MLP，三 seeds 等权。Loss 包括 independent regression、Bradley–Terry 风格 pairwise、set-rank/listwise、catastrophe-weighted tail-aware 和 calibration/abstention。

Richer-future baseline 使用相同 E3 输入、相同 folds 和 candidates，预测 20 步 EEF、task objects、contact flags 与 predicate，共 620 个输出，再从预测未来映射到 consequence card。它不是外部大 WAM，但比三维 consequence card 预测明显更丰富。

## 7. Baselines

比较 nominal、deterministic random、contact-only、motion-only、outcome-only、三种双 specialist、完整三 specialist、hard filter、固定加权、learned-small、pairwise comparator、listwise、tail-aware、lexicographic、uncertainty penalty、abstention，以及 richer-future。

## 8. Metrics

主要指标是 pairwise preference accuracy、setwise top-1、median/P90/P95 regret、catastrophic-selection rate、nominal opportunity capture 和 abstention fallback。统计单位保持为 demo/task，而不是把 candidate rows 当独立实验。

## 9. Results

Pairwise 层面存在可复用信号：listwise route 的平均 pairwise accuracy 为 0.716，contact+outcome 为 0.694，richer-future 为 0.696，contact-only 为 0.688。

Setwise 层面未形成增量效用：

| Route | Top-1 | P90 regret | P95 regret | Catastrophic | Opportunity capture |
|---|---:|---:|---:|---:|---:|
| Nominal | 0.881 | 0.0556 | 0.1682 | 0 | 0 |
| Contact only | 0.881 | 0.0556 | 0.1682 | 0 | 0 |
| Listwise | 0.881 | 0.0556 | 0.1682 | 0 | 0 |
| Richer future | 0.881 | 0.0556 | 0.1682 | 0 | 0 |
| Full three consequences | 0.821 | 0.1311 | 0.2981 | 0 | 0 |
| Random | 0.464 | 0.6233 | 0.9206 | 0.202 | 0.20 |

Contact-only、listwise、richer-future 实际都复现了 nominal 行为。相对 nominal 的 demo-bootstrap regret-improvement CI 为 `[0, 0]`，因此不能称 setwise action selection 改善。

## 10. Tail/failure analysis

最保守路线没有选择 catastrophic candidate，但 P95 regret 仍为 0.1682，来自未捕获的少数 Stove 改进机会。Motion-only、Outcome-only、pairwise comparator 与 learned-small 会主动离开 nominal，并分别产生约 6%–13% catastrophic selections。随机选择为 20.2%。这表明当前预测足以避开明显风险，但不足以安全利用稀少的正向机会。

## 11. Ablations

E1 raw compact 的综合预测误差最低。E5 独立 specialist 的 contact MAE 0.1164、outcome Brier 0.0410，是相应轴最佳；加入 object-centric、contact-aware 或 history 没有带来增量，E4 history 最差。三 specialist 全组合不如 contact-only，说明在机会稀少且 outcome 极不平衡时，额外 motion/outcome noise 会破坏保守排序。

Abstention 相对未校准的 full combination 将 P90 regret 从 0.1311 降到 0.0841，fallback 率 14.3%，因此 uncertainty 对“组合路线内部的尾部控制”有用；但它仍没有超过 nominal，不能称整个 selector 已解决。

## 12. Engineering failures/fixes

第一次数据构建在创建 run 目录之前因未把仓库根目录加入 `sys.path` 而停止。修复只改变 import path，新 run ID 为 `exp12_a0_candidate_dataset_r1_20260815`。首次模型 run `exp12_a1_ranking_models_20260815` 完成后，审计发现两个 gate 把“与 baseline 相等”误写成“supported”；未修改其工件，收紧为必须有严格效用后在 `exp12_a1_ranking_models_r1_20260815` 重跑。所有预测值一致，只有 claim classification 被纠正。

## 13. Scientific interpretation

EXP12 支持“action-conditioned consequences 含有 pairwise preference signal”，但不支持“当前 consequence coordinator 能从现有候选集中选出比 nominal 更好的动作”。瓶颈首先是 candidate generation：两个任务没有可利用的 nominal improvement，第三个任务也只有 10 个 groups。继续扩大 coordinator 而不改善 candidates 没有科学价值。

## 14. Claim impact

支持：contact/motion/outcome 在该 pilot 上可预测；pairwise ranking 有信号；保守 consequence score 可避免明显灾难；三维 consequence card 与 620 维 richer future 在当前 setwise 指标上同样不优于 nominal。

不支持：setwise ranking improvement、跨任务 opportunity capture、consequence sufficiency、闭环 utility、policy independence。Richer-future 与 consequence-only 的平局是 inconclusive，因为二者都没有超过 nominal。

## 15. Module status

```yaml
contact_prediction: supported
motion_prediction: supported
outcome_prediction: supported
pairwise_ranking: supported
setwise_ranking: unsupported
tail_safety: supported_for_conservative_selection
uncertainty_calibration: supported_within_full_combination
consequence_sufficiency_vs_richer_future: inconclusive
offline_candidate_ranking: not_yet_supported
```

## 16. Recommended next experiment

EXP13 应解决 Candidate Generation Diversity。目标不是放宽 success gate，而是从同一状态提出更多有效、任务方向明确、跨至少两个任务经常优于 nominal 的动作块；在真实 oracle opportunity 达到足够支持前，不再训练更大的 coordinator。

## 17. Git commits / hashes / artifact audit

- Protocol freeze commit：`d886367`；
- data/model implementation commit：`a78fff4`；
- candidate dataset SHA256：`a876e0476868f52a79cd2252f0fdcb50fc99c99c7604e0ff2cd2b56462e4e82d`；
- final model run：`exp12_a1_ranking_models_r1_20260815`；
- output audit：`exp12_a2_output_audit_20260815`，全部检查通过；
- 13 required plots 均为非占位图；
- 最终完整 repository tests、pip check 与 compileall 在 final handoff 前执行。

