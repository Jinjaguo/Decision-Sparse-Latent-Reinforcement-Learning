# EXP10 实验报告：Phase-Conditioned Macro-Action Trajectory Response

日期：2026-08-14

状态：Stage A 完整执行；Stage B 按预注册 stop rule 未执行
最终分类：`no_macro_scale_structure`

主要 run：

- 数据/相位构造：`runs/exp10_a0_phase_macro_dataset_r2_20260814`
- 23 方法多路线比较：`runs/exp10_a1_a6_multiroute_models_r1_20260814`
- Track-D endpoint OR 门：`runs/exp10_a7_endpoint_route_20260814`
- 三种子复核：`runs/exp10_a7_seed_variance_20260814`
- 协议精确产物：`runs/exp10_a9_protocol_artifacts_20260814`
- 最终独立审计：`runs/exp10_a10_final_output_audit_20260814`

## 1. 结论摘要

EXP10 在同一锁定数据、同一整轨迹五折、同一 H5/H10 proper-score 口径下实现并比较了 23 个
模型/消融，覆盖协议的 A–F 六条路线。总计生成并写后锁定：

- 30 条 reference demonstrations；
- 17,280 个既有 EXP8 q 干预；
- 794,880 条 trajectory prediction rows；
- 51,840 条 event/terminal prediction rows；
- 34,560 条补充 endpoint prediction rows；
- 6 个 phase/route/source manifests、13 个精确命名的 Stage-A artifacts、16 张指定图；
- 3 个固定 CVAE seeds：101001、101002、101003。

结果不是“少试了几个模型”，而是六条路线均在独立门上失败：

| 路线 | 主要问题 | 授权 Stage B |
|---|---|---:|
| A multi-scale macro action | 所有 chunk/history 变体 H5/H10 energy 均劣于 EXP9 temporal reimplementation | 否 |
| B phase conditioning/MoE | phase hybrid energy 变差 33–36%；事件 AUROC CI 下界 0.641 | 否 |
| C compact trajectory latent | 16/32 维未改善；0 维常数模型虽 energy 低但 coverage 仅 0.128，属于退化 | 否 |
| D terminal consequence | AUROC 0.793 但 CI [0.348, 0.913]；endpoint error 反而恶化 20.67% | 否 |
| E coarse regime AR | AUROC 0.722、specificity 0.476、false-safe 0.524 | 否 |
| F world-action distribution | CVAE energy 改善约 46%，但 coverage 仅 0.621/0.627，严重欠分散 | 否 |

因此 `new_cohort_authorized=false`。没有生成 EXP10 新 reference、matched zero、q intervention 或
Stage-B confirmatory row。这个停止是协议要求，不是执行遗漏。

## 2. 前置审计与不可变来源

执行前读取了 PROJECT、EXP1–EXP9 报告、`next_exp_from9.md`、EXP9 实验摘要、证据表和最新
research log。仓库基线为 86/86 tests passed，`pip check` clean；最终加入 EXP10 known-answer
tests 后为 104/104 passed。

只读取以下锁定来源：

- EXP8 formal raw：`exp8_s11_formal_raw_locked_r1_20260814`；
- EXP8 reference contact frames：`exp8_s4_contact_frame_audit_r1_20260814`；
- EXP8 independent references：`exp8_s2_independent_refs_20260814`；
- EXP9 audited inputs/targets：`exp9_a0_hybrid_targets_r2_20260814`。

每个来源的 SHA-256 写入 `source_hash_manifest.json`。EXP8/EXP9 原文件未被修改。

## 3. 相位 schema 与支持

审计发现 EXP8 contact-frame 表中的 `physical_progress_clipped` 在 Drawer/Stove 实际是任务原生关节值，
并非通用 0–1 进度。EXP10 没有错误地把它当共享相位轴。冻结 schema 使用 reference-only 信息：

- 归一化 reference time 的候选阈值 0.12/0.28/0.48/0.68/0.86；
- 连续两步 target–gripper contact 可使相位至少进入 P2；
- first-crossing + 单调 hysteresis，禁止相位后退；
- 精确 reference task predicate 首次为真时进入 P6；
- 不使用任何 q-response、模型误差、任务 success flip 或 held-out outcome 选边界。

Reference boundary 支持如下：

| Task | P0 | P1 | P2 | P3 | P4 | P5 | P6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Drawer | 177 | 230 | 289 | 287 | 261 | 95 | 112 |
| Bowl | 115 | 146 | 184 | 184 | 146 | 7 | 146 |
| Stove | 107 | 133 | 170 | 169 | 152 | 12 | 110 |

P5 在 Bowl/Stove 很短，硬专家训练时按冻结的 minimum-support fallback 回退到 global model；没有根据
结果合并 phase。相位序列全部单调，transition matrix 与每条 first crossing 均已保存。

## 4. 输入、目标与对齐

历史窗口为 1/5/10 步和最近 10 步 phase history；action chunk 为 5/10/20 步以及
phase-to-next-landmark（上限 20）。左/右 padding 均显式携带 mask。

连续目标保持 task-specific 原维度：Drawer/Stove 26，Bowl 32；没有截断 Bowl，也没有加入随机填充通道。
目标包括 H5/H10 signed physical response、最终 endpoint response、terminal position/orientation error 和
success flip。

Coarse regime 使用冻结的 R0–R6 vocabulary，并由精确 MuJoCo contact identity 确定。10 步序列不足时使用
末个有效 regime 右填充并保存 mask；teacher-free rollout 的 safe probability 忽略 padding。宏观不利事件比例为：

- Drawer 0.49097；
- Bowl 0.89931；
- Stove 0.78872。

锁定 raw 没有 perturbation-side wrench/force trajectory。EXP10 没有从 reference force 或当前 force
伪造未来 wrench target。

## 5. 交叉拟合与实现方法

所有方法按完整 `(task, episode)` 五折隔离。训练只使用 `direction_role=basis`；测试保留该 fold 的全部
intent-to-perturb rows，包括 held-out random direction。标准化、残差尺度、phase-specific 尺度和分类阈值
均只用训练折。

一次实验中实际实现了 23 个方法/消融：

1. `baseline_b`、`exp9_temporal` reimplementation；
2. `no_action`、`no_history`；
3. chunk 5/10/20/phase-to-landmark；
4. history 1/5/10/phase-history；
5. hard phase MoE、soft phase MoE、hybrid phase、shuffled phase；
6. latent 0/16/32；
7. deterministic、phase-heteroscedastic、two-residual-mixture；
8. 真正的 16 维 conditional VAE，3 seeds 等权，不 cherry-pick；
9. direct terminal classifier/regressor；
10. 10 步 coarse-regime autoregressive model，测试时只反馈前一步预测。

线性/低秩候选使用同一冻结 compact projection，避免每个模型使用不同数据表示。CVAE 每个实例 31,392
参数，float32 GPU 训练，float64 scoring，RTX 4090，`CUBLAS_WORKSPACE_CONFIG=:4096:8`。

## 6. H5/H10 主要结果

下表的 improvement 均相对本 EXP10 中同 fold、同 target 的 `exp9_temporal` reimplementation；它不是把
EXP9 原 H1/H3/H5 数字直接搬入新 H10 estimand。

| 方法 | H5 energy | H5 改善 | H5 coverage | H10 energy | H10 改善 | H10 coverage |
|---|---:|---:|---:|---:|---:|---:|
| EXP9 temporal | 0.042875 | 0% | 0.7188 | 0.040159 | 0% | 0.7292 |
| Baseline B | 0.034965 | +18.45% | 0.8437 | 0.034805 | +13.33% | 0.8436 |
| A chunk 5 | 0.051472 | −20.05% | 0.6379 | 0.047882 | −19.23% | 0.6567 |
| A chunk 10 | 0.057214 | −33.44% | 0.6039 | 0.054610 | −35.98% | 0.6223 |
| A chunk 20 | 0.058649 | −36.79% | 0.5918 | 0.054227 | −35.03% | 0.6140 |
| A phase chunk | 0.055485 | −29.41% | 0.6176 | 0.050276 | −25.19% | 0.6419 |
| B hard MoE | 0.097417 | −127.21% | 0.4903 | 0.097129 | −141.86% | 0.5093 |
| B soft MoE | 0.073236 | −70.81% | 0.5520 | 0.073398 | −82.77% | 0.5697 |
| C latent 16 | 0.057083 | −33.14% | 0.6248 | 0.054546 | −35.82% | 0.6394 |
| F heteroscedastic | 0.058001 | −35.28% | 0.5645 | 0.055415 | −37.99% | 0.5717 |
| F residual mixture | 0.059920 | −39.76% | 0.5125 | 0.057176 | −42.37% | 0.5222 |
| F CVAE16 ensemble | **0.023111** | **+46.10%** | **0.6211** | **0.021765** | **+45.80%** | **0.6270** |

Baseline B 是唯一在 H5 达到 15% 平均改善且 demo CI 为正的简单方法，但它不属于 Track A–F 的新
macro route，H10 改善也只有 13.33%，因此不能替代 route gate。

CVAE 的 energy 改善在三任务均为正，且三种子稳定：

| Horizon | seed mean energy 的均值 | seed 间标准差 | 范围 |
|---|---:|---:|---:|
| H5 | 0.024118 | 0.000272 | [0.023831, 0.024373] |
| H10 | 0.022746 | 0.000080 | [0.022677, 0.022834] |

但其 90% marginal coverage 仅 62%左右，低于 Track F 冻结的 0.85–0.95 区间。低 energy 主要来自
目标高度集中在零附近时的窄预测；它没有提供合格的分布覆盖，因此不能包装成成功的 world model。

`C_latent0` 也呈现类似陷阱：H5/H10 energy 改善约 41.7%/41.8%，但 coverage 只有 0.1282，且零维
表示按定义完全 collapse。Track C 的 16/32 维都变差，故 compact latent 不合格。

## 7. Event、terminal 与 endpoint

| 模型 | AUROC | demo-bootstrap 95% CI | AUPRC | Sensitivity | Specificity | False-safe |
|---|---:|---:|---:|---:|---:|---:|
| B phase hybrid | 0.6891 | [0.6406, 0.7322] | 0.8260 | 0.5776 | 0.6458 | 0.3542 |
| E teacher-free AR | 0.7219 | [0.6667, 0.7755] | 0.8505 | 0.8287 | 0.4762 | 0.5238 |
| D terminal direct | 0.7926 | [0.3475, 0.9133] | 0.0529 | 0.5935 | 0.5048 | 0.4952 |

Track E 虽比 EXP9 exact-pair event 的约 0.30–0.35 specificity 更接近可用区，但仍同时失败
sensitivity≥0.85、specificity≥0.55、false-safe≤0.45 和 AUROC CI lower≥0.75。只能称为方向性改善，
不能称 coarse regime 已可复用。

Track D 的分类点估计接近 0.80，但 success flip 极少，CI 极宽且下界只有 0.348。补充 OR 门对 endpoint
做了独立 write-once cross-fit：

- EXP9 temporal mean endpoint L2：0.10180，p90 0.20883；
- direct terminal model mean：0.12284，p90 0.21704；
- relative improvement：−20.67%；
- demo-level absolute improvement CI：[-0.04053, -0.00361]。

因此 classification 与 endpoint 两个 Track-D 分支都失败。

## 8. 六条 route 的冻结授权判定

### Track A

要求 H5/H10 均至少改善 15%、demo CI 下界>0、至少 2/3 tasks 正向。所有 A 方法平均改善均为负；失败。

### Track B

要求 phase event AUROC CI lower≥0.75、specificity≥0.55 at sensitivity≥0.85、trajectory 至少改善 10%。
三项均未同时满足；失败。Hard MoE 比 global 更差，说明 phase sample fragmentation 很严重。

### Track C

要求 16/32 维 latent H5/H10 均改善至少 15%、terminal 同时改善、latent 不 collapse。16/32 维 energy
均变差；0 维虽 score 低但退化且 coverage 失败；失败。

### Track D

要求 terminal AUROC≥0.80 且 CI lower≥0.70，或 endpoint error 改善至少 15%。两分支均失败。

### Track E

要求 AUROC CI lower≥0.75、AUPRC≥0.70、specificity≥0.55、sensitivity≥0.85、false-safe≤0.45。
AUPRC 单项通过，其余关键选择性条件失败。

### Track F

要求 H5/H10 energy 均改善至少 20%、90% coverage 在合格带、p90 endpoint 改善。CVAE proper score
通过幅度门但 coverage 明确失败；其他分布模型 score 变差；失败。

## 9. 消融解释

- 最佳 A chunk 是 5 步，但仍比 EXP9 temporal 差 20.05%/19.23%；更长 chunk 没有帮助。
- `no_action` 比所有 A chunk 模型更好，说明未来 action summary 没有独立增量价值。
- `no_history` 很差，但 history1/5/10 在加入统一 phase/action 后也没有击败 baseline；不能声称长历史有独立价值。
- Shuffled phase 比正确 phase hybrid 更好，但仍劣于 temporal baseline。这强烈反对“当前 phase label 提供可复用动力学分区”。
- Hard/soft MoE 大幅恶化，符合 phase 内 demonstration 支持不足和跨 phase 边界不连续的现象。
- Two-residual mixture 未优于 Gaussian，EXP10 仍不能声称真实多模态需要 mixture。
- Temporal-only 没有再次战胜所有 contact/state-heavy baselines；本次 Baseline B 反而优于 temporal reimplementation。

## 10. 保留的失败与修正

1. `exp10_a0_phase_macro_dataset_20260814`：写审计 JSON 时把 `(task, episode)` 元组当任务键；在写最终
   metrics 前停止。修正纯汇总键，新 run 完整重建。
2. `exp10_a0_phase_macro_dataset_r1_20260814`：连续目标完整，但临近轨迹末端的离散 regime 序列没有
   明示 padding mask。没有模型结果；加入“末有效 regime 右填充 + mask”后用 r2 重建。
3. `exp10_a1_a6_multiroute_models_20260814`：连续 23×2 比较完成，但事件阶段发现非等长 regime 序列，
   在 prediction lock 和授权前停止。修复来自数据 schema，不改变模型/gate/seed/fold；r1 从头重跑。
4. 主脚本最初只把 Track D 的 AUROC 分支写入 route authorization。为完整执行冻结的 OR 规则，新增
   独立 endpoint prediction lock；两个 D 分支都失败，没有因此后验调整模型。

所有失败 run 都保留，没有覆盖。没有依据结果改变 architecture、fold、seed、epoch、threshold 或 gate。

## 11. 产物、锁与审计

主 prediction locks：

- trajectory SHA-256：`d589125a440d254df79eddc4099b0ad4eb0ce6bd9af05525e3e23a4dbe75d994`；
- event/terminal SHA-256：`ae1b1ba2a26de3e14d5e646183cd053a086f88dda235861faa4c1f1bff339c08`；
- Track-D endpoint SHA-256：`0334dde11724557a6ebe720419c02ffea56d7a77f6f87496afb7324a162767a1`。

最终独立审计 15/15 checks passed，包括 row counts、hash、精确 artifact names、16 张精确图、六条 route
均未授权和无 Stage-B run。最终 repository tests 104/104 passed；`pip check` clean；EXP10 compileall passed。

## 12. 多轴最终分类

```yaml
temporal_macro_action: unsupported
phase_conditioning: unsupported
trajectory_latent: unsupported
terminal_consequence: unsupported
coarse_event_sequence: unsupported
world_action_distribution: unsupported
summary_label: no_macro_scale_structure
```

这是针对“现有 local-q impulse + 锁定 EXP8 observation schema + 当前 macro targets”的结论，不是说所有
机器人宏动作世界模型普遍不可能。

## 13. 协议最终问题回答

| 问题 | 回答 |
|---|---|
| 是否有 route 授权新数据？ | 否，A–F 全失败。 |
| 哪个 chunk 最好？ | 5 步 A chunk；但仍不合格。 |
| 更长 chunk 是否优于 5-action？ | 否。10/20/phase chunk 均更差。 |
| Phase conditioning 是否有帮助？ | 否；正确 phase 不优于 shuffled phase。 |
| Phase experts 是否有帮助？ | 否；hard/soft MoE 明显更差。 |
| Compact latent 是否有帮助？ | 16/32 维没有；0 维是退化假象。 |
| 哪个 latent dimension 工作？ | 无。 |
| Terminal consequence 是否可预测？ | 点估计接近，但 CI/endpoint OR 门失败。 |
| Coarse regimes 是否优于 exact pair events？ | 有方向性选择性改善，但未过冻结门，不能称成功。 |
| Temporal-only 是否再次胜过 graph/contact-heavy？ | 否；Baseline B 在本次 H5/H10 更好。 |
| World-action trajectory 是否改善？ | CVAE energy 改善但严重欠覆盖，联合门失败。 |
| History/action chunk 有独立价值？ | 没有可靠证据。 |
| Mixture 有增量价值？ | 没有。 |
| 是否 task-specific？ | Event prevalence 和 phase support 明显 task-specific，但没有 task 达成足以授权的全局 route。 |
| 是否有独立 Stage-B cohort？ | 未审计、未收集；协议禁止。 |
| Stage-B 哪些路线复现？ | 无，因为 Stage B 未授权。 |
| 最强允许结论？ | 当前 local-q intervention 即使提升到 phase/macro scale，也未暴露出可校准、跨 demo 可复用的预测结构。 |
| Offline decision-allocation utility eligible？ | 否。 |
| Dedicated latent representation study eligible？ | 否；Track C 未通过。 |
| Latent RL eligible？ | 否。 |
| EXP11 应测试什么？ | 改变干预对象为结构化 action-chunk replacement，而不是继续增加 state encoder。 |

## 14. 最终 disposition

EXP10 已执行协议允许的全部 Stage-A 方法，并在明确的停止点结束。最有信息量的结果是：CVAE 和零维常数
预测器能利用近零目标获得好看的 energy score，却不能提供合格覆盖；更长 action chunk 和 phase experts
没有修复跨 demonstration 泛化。这表明下一步不应继续堆叠 encoder 或增加 q-impulse 模型容量，而应重新
定义因果干预本身。

下一实验建议见 `reports/next_exp_from10.md`。
