# EXP7 实验报告：Contact-Mode-Conditioned Local Response

日期：2026-08-14

最终状态：完整执行、正式 raw 已锁定、分析与产物审计通过

最终分类：`within_mode_nonsmoothness_persists`

## 1. 结论摘要

EXP7 回答了一个比 EXP6 更精确的问题：如果把接触切换造成的非光滑性隔离掉，只看“双符号扰动都保持 matched-zero 精确接触模式”的短时响应，局部 q-response 是否收敛、是否能预测 held-out 方向、以及是否能跨 demonstration 复用？

结果不是简单的“成功”或“失败”，而是一个清晰的两层结论：

1. **同一精确状态附近的一步局部算子非常稳定。** H1 通过，30/30 个 demo 的条件中位数满足全部四项门，top-1 分层 95% CI 为 `[0.99849, 0.99992]`。
2. **该算子不能只用“精确离散接触模式 + 1 mm margin 类别”跨 demo 复用。** H4 显著向反方向变化：同 mode+margin 相对最佳 time/progress 基线的平均改善为 `-0.03736`，95% CI `[-0.06554, -0.01040]`。

因此，EXP6 看到的非收敛确实有一部分来自接触模式切换，但“离散模式”不是充分状态。EXP7 支持的是**依赖更丰富连续接触几何与连续物理状态的局部 chart**，而不是仅由 mode label 索引的可复用全局/跨轨迹 response field。

H5 的 reference-only next-step mode-preservation predictor 通过预注册门：AUROC `0.89783`，demo-cluster 95% CI `[0.87495, 0.92000]`，ECE `0.02880`。但冻结阈值 0.5 的 specificity 只有 `0.41608`，所以该模型适合支持下一次“风险门验证”，不代表 EXP7 已授权控制或调度。

## 2. 与研究主线的关系

项目主线是判断是否存在可用于 decision-sparse / latent RL 的稳定局部作用结构。EXP3–EXP5 显示时间、标量进度和粗状态匹配不足；EXP6 在更小半径仍发现强非对称和半径依赖，但接触模式分歧对失败有显著解释力。

EXP7 沿主线做的是 estimand reconciliation，而不是引入新控制器：

- 用真实接触表面间隙与精确 named geom-pair mode 替代 contact count；
- 将 intent-to-perturb 与 preserved-mode conditional estimand 同时保留；
- 把 remaining horizon 拆为 H1/H3/H5/remaining；
- 用新独立 cohort 避免在 EXP5/EXP6 cohort 上重复确认；
- 在正式结果前冻结 predictor features、cross-fit folds 和统计门。

没有训练 scheduler、policy、MPC、VLA 或 latent RL。

## 3. 预注册假设与结果

| 假设 | 预注册判据摘要 | 结果 | 结论 |
|---|---|---:|---|
| H1：preserved-mode H1 收敛 | 至少 70% demo 同时满足 top1≥0.80、top2≥0.75、spectral≤0.20、asymmetry≤0.25；top1 分层 CI 下界>0.65 | 30/30；CI `[0.99849, 0.99992]` | 通过 |
| H2：interior 优于 near/ambiguous | demo-cluster CI 排除 0，BH q<0.05 | 均值 `+0.04169`；CI `[-0.00856, 0.11765]`；q=`0.5` | 失败 |
| H3：held-out 方向预测 | demo-median rho≥0.65；vector error≤0.35 | rho=`0.98788`；error=`1.64e-6` | 通过 |
| H4：mode+margin 跨 demo 复用 | 相对更优 time/progress 基线改善≥0.15，CI 下界>0 | 均值 `-0.03736`；CI `[-0.06554, -0.01040]` | 失败且方向相反 |
| H5：下一步 mode preservation 可预测 | AUROC demo-cluster 下界≥0.70；ECE≤0.10 | AUROC=`0.89783`，CI `[0.87495,0.92000]`；ECE=`0.02880` | 单独通过 |

按照事前冻结的优先级：

1. 类别 1 要求 H1+H3+H4；H4 失败，因此不能分类为 `within_mode_short_horizon_operator_converges`。
2. 类别 2 要求 H2；H2 失败。
3. 类别 3 要求 contact modes 有解释力但不可预测；H5 实际可预测，因此不适用。
4. 落入 `within_mode_nonsmoothness_persists`。

这里的“persists”主要指**跨 demo 的连续状态依赖仍然存在**，不应误读为同一状态的 H1 算子不收敛。

## 4. 运行环境

- OS：Windows，本机 PowerShell
- Conda 环境：`libero-exp1`
- Python：3.8.20
- NumPy：1.22.4
- SciPy：1.10.1
- PyArrow：17.0.0
- pandas：2.0.3
- Matplotlib：3.5.3
- MuJoCo：3.2.3
- robosuite：1.4.0
- PyTorch：1.11.0+cu113
- GPU：NVIDIA GeForce RTX 4090
- LIBERO SHA：`8f1084e3132a39270c3a13ebe37270a43ece2a01`
- robosuite source SHA：`fbee5844ff5632f5b5698e204ec5357ca50be0df`
- 正式执行时项目 SHA：`42c117bf8744be8847d9f85a31f69b03c968bd13`

MuJoCo 仿真保持 CPU；GPU 只在新版 scale-aware float64 CPU/GPU 等价门通过后可用于后处理。正式分析以 CPU 数值为记录真值。

## 5. Step 0：运行前基线

初始基线：

- `pytest -q`：56 passed；
- 增加 EXP7 纯指标测试后：59/59 passed；
- 最终测试使用独立可写 basetemp 与禁用 pytest cache：59/59 passed；
- `pip check`：No broken requirements found。

一次中间测试遇到本机历史 pytest 临时目录 ACL 的两个 setup error；没有断言失败。使用 `runs/exp7_pytest_final_20260814` 作为独立 basetemp 后完整通过。历史上无法访问的 `.pytest_tmp_exp6_*` 未被修改或删除。

## 6. Step 1：独立 cohort 审计与生成

三个任务的 HDF5 各包含 demo 0–49。程序化排除 EXP1–EXP6 已使用样本后：

| Task | 首个未使用 episode | 可用未使用数 | 实际接受 |
|---|---:|---:|---|
| Drawer | 21 | 29 | 21–30 |
| Bowl | 20 | 30 | 20–29 |
| Stove | 20 | 30 | 20–29 |

选择规则在执行前固定为：每任务按 episode 升序扫描，只接受成功、全部 finite、corrected-D integration/controller roundtrip 精确为 0 的轨迹；失败保留；每任务得到 10 条后停止。

结果：30/30 首批候选全部合格，无 replacement、无旧 cohort fallback。

关键 runs：

- `exp7_s1_cohort_audit_20260814`
- `exp7_s2_refs_drawer_21_30_r1_20260814`
- `exp7_s2_refs_bowl_20_29_r1_20260814`
- `exp7_s2_refs_stove_20_29_r1_20260814`
- `exp7_s2_independent_refs_20260814`

最初三个 reference 命令在沙箱内因 robosuite 尝试写 `C:\tmp\robosuite.log` 而失败；失败 runs 被保留。获得授权后在沙箱外用新 run id 重跑，均通过。

## 7. Step 2：接触几何与 signed-gap 审计

使用 MuJoCo 3.2.3 的 `mj_geomDistance`，其官方定义是两个 geom 的最小有符号表面距离，负值表示 penetration；若未在 `distmax` 内找到碰撞则返回上限。`mjContact.dist` 同样是最近表面距离，负值表示 penetration。参考：

- [MuJoCo 3.2.3 `mj_geomDistance`](https://mujoco.readthedocs.io/en/3.2.3/APIreference/APIfunctions.html#mj-geomdistance)
- [MuJoCo 3.2.3 `mjContact`](https://mujoco.readthedocs.io/en/3.2.3/APIreference/APItypes.html#mjcontact)

几何归属不是根据位置或模糊名称猜测：

- target：`env.env.obj_body_id` 加 model body ancestry；
- gripper：`env.robots[0].gripper.contact_geoms`；
- contact pair：参考轨迹真实 `mjData.contact` geom IDs；
- normal relative velocity：最近点方向/接触法向投影到两个 geom Jacobian 的相对速度。

冻结的相关 exact pairs：

| Task | target-gripper | target-environment | gripper-environment | task-object-environment | 总数 |
|---|---:|---:|---:|---:|---:|
| Drawer | 9 | 2 | 0 | 0 | 11 |
| Bowl | 42 | 11 | 3 | 11 | 67 |
| Stove | 20 | 0 | 0 | 0 | 20 |

没有使用 body-center 距离替代 signed gap。几何 schema 可辨识，未触发 `contact_schema_not_identifiable` 停止条件。

## 8. Step 3：boundary margin calibration

对独立 cohort 的全部 3,316 个参考边界恢复并测量几何；在每条轨迹的 4 个时间位置共 120 个边界做 4 次重复恢复。

- signed-gap 最大重复 range：0 m；
- normal-velocity 最大重复 range：0 m/s；
- exact mode 重复：全部精确相同；
- `m_near = max(1e-12 m, 10 × observed precision) = 1e-12 m`；
- `m_far = 1e-3 m`，作为事前固定物理阈值。

全部参考边界分布：2,809 interior、507 near、0 ambiguous。

冻结分支中的分布：275 interior、85 near、0 ambiguous；Stove 只有 4 个 near 分支，这限制 H2 的任务层级统计能力，但没有进行后验补采样。

## 9. Step 4：branch、radius、direction 与 horizon 冻结

每个 demo 选择 12 个 reference-only 分支：先覆盖稀有的 contact/margin/gripper/predicate 联合 strata，再补齐 12 个均匀时间 quantile 的最近未使用边界。任何 joint-limit 不可行边界只允许按冻结的最近未使用规则替换。

规模：

- 30 trajectories；
- 360 branches；
- radius fractions：0.0003125、0.000625、0.00125；
- 7 个拟合 basis 方向 + 1 个 held-out random 方向；
- `±` 两个符号；
- 总计 `360 × 3 × 8 × 2 = 17,280` 干预；
- H1、H3、H5、remaining 四个 horizon。

冻结 manifest 核心 SHA256：

| Manifest | SHA256 |
|---|---|
| cohort | `fc213fefedbf7ad2ec85aff6c58ee5ac20d600161ccf98c96e05b8facc6c3e1d` |
| contact schema | `4977559a18fd58ae09353e07baab1602b54adb92b16dbfb4eda1bf3bfc381d8c` |
| boundary margin | `8e09cc3d06da009c0b872c81222a1cc6b9d695cee679cf885f788b40ab198ab7` |
| branch | `6241978611c23dde15637a838dcd3bdfbc84087217bcfa360b5b851446062bb6` |
| radius | `8b24ea1de66ab39450719d61152f5f27b22170d7f028dc8c27976ce16f57ac8c` |
| directions | `dcbd440a46dac92632abd9f6773355a7c2ff68444aa3793afca1c8ab8467efc` |
| predictor folds | `9520e158db6c6ec67f11cd1256ef318b567c01991639c62b6253f4f6acaeea18` |

冻结提交：`93664a9`；正式分片/分析管线提交：`42c117b`。

## 10. Step 5：GPU scale-aware gate

EXP6 使用绝对误差门时，Gram/eigenvalue 的大尺度值会使微小相对误差错误失败。EXP7 在 formal outcome 前冻结：

`abs_error <= atol + rtol × max_abs_scale`

9 个 task-output-dimension/radius case 全部通过：

- operator 相对误差约 `1.16e-16`–`1.62e-16`；
- Gram 相对误差约 `1.35e-16`–`5.57e-16`；
- spectrum 相对误差约 `3.45e-16`–`1.18e-15`；
- dtype float64；
- GPU：RTX 4090；
- 未放宽任何阈值。

Run：`exp7_s7_gpu_gate_20260814`。

## 11. Step 6：dry run 与 matched-zero

单 trajectory、单 branch、单方向 dry run：

- 2 条 signed interventions；
- 256 per-step rows；
- non-arm max 0；
- gate passed。

正式 matched-zero：

- 360 branches；
- 每 branch 2 个 zero twins；
- 22,811 zero reference future rows；
- integration median/p95/max：全部 0；
- terminal object-pose p95：0；
- arm q/qvel、EEF、task-object pose：全部精确；
- contact mode/predicate：全部精确；
- signed gap 重复性由独立 120-boundary calibration 证明为精确。

运行器在完成 360 个不可变分片并合并 Parquet 后，写 metrics 时因 `exp7.json` 缺少 EXP4 runner 使用的四个 zero-threshold alias 而触发 `KeyError('zero_median_max')`。这不是仿真失败。处理方式：

1. 保留失败 run `exp7_s9_zero_controls_r1_20260814`；
2. 不重写任何完成分片；
3. 把协议已经规定的 exact-zero/1e-12 阈值显式映射到四个 runner alias；
4. 在新 run `exp7_s9_zero_controls_validated_20260814` 中独立重验全部 360 行和 raw hashes；
5. gate passed。

## 12. Step 7：正式干预执行与 raw lock

先按三任务分片启动。每任务首个 demo 完成后，在 demo 原子边界停止原长进程，保留已完成分片；剩余 27 个 demo 被拆成 9 个不重叠的 3-demo 进程。该重分片只改变调度，不改变 branch、direction、radius、sign、horizon 或分析规则。

全局 merge gate 对实际 key 与冻结 key 做集合完全相等验证：

- interventions：17,280/17,280；
- unique rows：通过；
- exact frozen coverage：通过；
- per-step rows：1,094,928；
- execution failures：0；
- branch removed：0；
- maximum non-arm integration change：0；
- maximum q-injection error：`2.220446049250313e-16`；
- terminal success flips：145（Drawer 78、Bowl 66、Stove 1）。

Raw lock run：`exp7_s11_formal_raw_locked_20260814`。

Raw SHA256：

| Artifact | SHA256 |
|---|---|
| zero controls | `de5e6376c2090d5193b88ba3a1f193be5b4cad153915b043cbce98105a4c456b` |
| zero steps | `de24c43cfe38c78d4f1337b6897fb3e1e6c669469086a66b15db63fc9d02b4f3` |
| interventions | `c4715750784a43114285b3285f19a3fe6a3c3ae469b03b399ae8a355c209a4f1` |
| per-step effects | `b49ff0697d3ca3535db9071c85f3651d5a3a997c314bd4a3776c9ea9071dd718` |

在 raw lock 完成前没有执行正式 q-response 分析。

## 13. 分析定义

对每个 direction、sign 和 horizon，signed output vector 是前 `min(H, continuation length)` 步的 duration-normalized mean；remaining 使用全部剩余步。

七列 operator：

`J_r[:, j] = (y_plus - y_minus) / (2r)`

H1 primary conditional inclusion 要求构成 operator 的七个方向，在相邻两个 radius 中，正负两个 sign 都在 H1 保持 matched-zero exact grouped mode。所有 intent-to-perturb 行仍保留并单独报告。

Mode transition 类别没有删除任何行：

- A：双符号保持；
- B：双符号进入同一新模式；
- C：双符号进入不同新模式；
- D：仅一个符号改变。

统计独立单位为 demonstration；bootstrap seed 和 folds 在 formal outcome 前冻结。

## 14. H1：一步 preserved-mode convergence

H1 通过。

- 30/30 demo 的条件中位数同时通过四项门；
- demo fraction：1.0；
- top-1 hierarchical 95% CI：`[0.99849, 0.99992]`。

H1 conditional branch-pair 描述：

| Task | N | median top1 | median top2 | median spectral discrepancy | median asymmetry | full pass rate |
|---|---:|---:|---:|---:|---:|---:|
| Drawer | 192 | 0.999997 | 0.999996 | 4.55e-6 | 0.00217 | 0.8281 |
| Bowl | 155 | 0.999996 | 0.999995 | 4.60e-6 | 0.00212 | 0.9161 |
| Stove | 159 | 0.999990 | 0.999991 | 6.36e-6 | 0.00236 | 0.9434 |

这说明 EXP6 的 unconditional remaining-horizon 非收敛不能外推为“任何固定模式的一步微小响应都不存在”。

同时，Drawer 的 p05 top2 仅 0.4277、p95 asymmetry 0.6380，说明分支级尾部仍很重；H1 通过是预注册的 demo-median 结论，不是每个 branch 的统一保证。

## 15. H2：boundary margin

H2 失败。

- 有可配对 interior/near 支持的 demo：15；
- interior-minus-near convergence 均值：`+0.04169`；
- 95% CI：`[-0.00856, 0.11765]`；
- one-sided p：0.5；
- BH q：0.5。

方向是预期的正方向，但证据不足，不能称 1 mm margin 已解释 residual nonsmoothness。可能原因包括：二元 1 mm bin 太粗、signed gap 只取组内最小值丢失了接触法向/多接触几何、以及 Stove near 支持太少。

## 16. H3：held-out direction

H3 通过：

- demo-median held-out magnitude-rank rho：`0.98788`；
- demo-median signed-vector relative error：`1.64e-6`。

但 branch-level 分布明显长尾：最小 radius、H1、held-out 双符号保持的 296 行中，中位 vector error 为 `7.90e-7`，均值 `0.2567`，p95 `0.8167`。因此允许的表述是“demo-median gate 通过”，而不是“任意 branch 都能准确向量预测”。

## 17. H4：跨 demo mode-conditioned reuse

H4 失败，而且失败方向稳定：

- 配对数：246；
- same mode+margin 相对 `max(time, progress)` 的平均改善：`-0.03736`；
- 95% CI：`[-0.06554, -0.01040]`；
- 中位数：0；
- p05：`-0.52660`；
- p95：`+0.03781`。

按任务均值：

- Drawer：`-0.04576`；
- Bowl：`+0.00061`；
- Stove：`-0.06387`。

这直接否定“精确离散 mode + margin class 已经是足够的跨轨迹局部状态”。最合理的解释是：同一个 named-pair set 内部仍存在连续接触位置、法向、penetration、接触力/约束激活程度、q 姿态和物体相对位姿差异。

## 18. H5：mode-preservation predictor

使用 demo-level 5-fold cross-fitting；输入只来自 reference/pre-intervention whitelist：radius、sign、direction index、signed gap、normal velocity、normalized time、physical progress、mode size、gripper state。没有使用 response vector、criticality、success flip 或 terminal outcome。

结果：

- positive rate：0.84450；
- AUROC：0.89783；
- demo-cluster 95% CI：`[0.87495, 0.92000]`；
- AUPRC：0.98078；
- Brier：0.09217；
- ECE：0.02880；
- threshold：0.5；
- sensitivity：0.95539；
- specificity：0.41608。

H5 按冻结 discrimination/calibration 门通过，说明 mode-transition risk 有 reference-side 可预测结构。但 specificity 很低，0.5 threshold 会放行过多不保持模式的动作。下一次实验应预注册高 specificity 或 conformal risk constraint，而不是直接把该阈值接入控制。

## 19. Horizon degradation

双符号贯穿 horizon 保持模式的比例：

| Horizon | Preservation rate |
|---|---:|
| H1 | 0.79815 |
| H3 | 0.72639 |
| H5 | 0.67616 |
| Remaining | 0.08553 |

Intent-to-perturb top1 median：

- H1：0.99873；
- H3：0.99393；
- H5：0.98807；
- remaining：0.64466。

Conditional survivor top1 median 在四个 horizon 都接近 1，但 remaining 只剩 42 个相邻-radius branch pairs。这个对比说明：随着 horizon 增长，主要变化是越来越多轨迹离开原 mode，不能只展示 survivor 曲线并声称长时算子稳定。

## 20. Exact-mode ablations

Individual-intervention preservation rate：

| Definition | H1 | H3 | H5 | Remaining |
|---|---:|---:|---:|---:|
| raw contact count | 0.86067 | 0.77892 | 0.72133 | 0.15596 |
| raw exact pair set | 0.84444 | 0.76955 | 0.71472 | 0.15283 |
| frozen all relevant groups | 0.84956 | 0.77182 | 0.71646 | 0.15288 |
| target-gripper only | 0.86768 | 0.79202 | 0.73630 | 0.16232 |

Contact count 更宽松，因为它允许 pair identity 改变时计数不变；target-gripper only 也忽略环境与 task-object contacts。四种定义的差异不大，说明 H4 失败不是由某一种 exact-mode 定义异常苛刻单独造成的。

## 21. 接触模式类别

H1 direction/radius pair（总计 8,640）类别：

- A both preserve：6,896；
- B same new：271；
- C signs different：672；
- D one sign changes：801。

Remaining endpoint 类别 A 仍有 5,213，但“贯穿 entire remaining horizon 都保持”的比例只有 0.08553；endpoint equality 不能替代完整 mode-path preservation。

## 22. 失败、修正与审计说明

保留的非科学失败：

1. 三个沙箱内 reference runs 因 `C:\tmp\robosuite.log` 权限失败；授权后新 run id 成功。
2. 首个 boundary calibration 因任务识别签名不唯一失败；所有三个任务场景包含同一组物体。修正为使用冻结 task-object body signature，新的 audit/calibration 通过。
3. 首个 freeze 错把 integration vector 用作 q 读取源，导致 joint-limit preflight 假失败；修正为从 audited boundary `panda_arm_q` 读取后冻结通过。没有 formal outcome 被观察。
4. zero run 完成数据后因 threshold alias KeyError 写 metrics 失败；完整数据被新 run 独立重验，未重跑/筛选。
5. 首轮 formal analysis 错把 intent-to-perturb 行用于 H1；该 run `exp7_s12_formal_analysis_cpu_20260814` 保留但不用于结论。修正为 preserved-mode conditional、缺支持 demo 保守失败，并增加 H5 demo-cluster AUROC CI；最终 run 为 `exp7_s12_formal_analysis_preserved_r1_20260814`。
6. 协议逐字审计发现早期实现用了语义等价但非指定的目录/文件名；最终目录改为 `exp7_contact_mode_conditioned`，并为 manifest/script/artifact 创建协议指定名称。内容未改变，正式 raw hashes 未改变。

## 23. 替代解释与限制

- **条件选择效应**：remaining 条件 survivor 极少，不能把其高相似度解释为总体长时稳定。
- **margin 粗糙**：1 mm 二元分类可能无法表示多接触几何的局部 manifold 坐标。
- **mode 不含力**：exact geom-pair set 不编码 normal/tangent force、constraint impedance 或 contact age。
- **任务支持不均**：Stove near 分支仅 4 个，H2 的任务层级功效有限。
- **H3 重尾**：demo median 极好但 branch p95 error 高。
- **predictor threshold**：AUROC/ECE 好，但 specificity 低，不宜直接当安全门。
- **模拟器范围**：结论限于这三个 LIBERO 任务、robosuite 1.4.0、MuJoCo 3.2.3 和 corrected-D substrate。
- **没有控制效用测试**：所有结果仍是 causal estimand/operator validation。

## 24. Claim impact

可支持：

- fixed-state、preserved-mode、H1 的小半径局部 operator 在 demo-median 层面收敛；
- 该 operator 可在 frozen H3 demo-median gate 下预测 held-out direction；
- mode preservation risk 可由 reference-side features cross-fit 预测；
- horizon 增长主要通过 mode-path 离开导致 unconditional 退化。

不可支持：

- exact mode+margin 已足够跨 demo 复用 operator；
- 1 mm margin 已显著解释 residual nonsmoothness；
- 每个 branch 都有可靠 held-out vector prediction；
- EXP7 已授权 scheduler/control/MPC/VLA/latent RL。

## 25. 正式产物

最终分析 run：`runs/exp7_s12_formal_analysis_preserved_r1_20260814`

20 个协议要求 artifact 全部存在，17 个协议要求 plot 全部存在；`output_audit.json` gate passed。主要文件：

- `reference_contact_geometry.parquet`
- `boundary_margin_calibration.parquet`
- `zero_controls.parquet`
- `interventions.parquet`
- `per_step_effects.parquet`
- `mode_outcomes.parquet`
- `horizon_operator_summary.parquet`
- `operator_matrices.parquet`
- `within_mode_convergence.parquet`
- `boundary_margin_analysis.parquet`
- `horizon_comparison.parquet`
- `heldout_direction_prediction.parquet`
- `mode_conditioned_crossdemo.parquet`
- `mode_predictor_predictions.parquet`
- `mode_predictor_metrics.json`
- `gpu_audit.json`
- `gpu_cpu_equivalence.json`
- `scientific_decision.json`
- `failure_examples.json`
- `raw_hash_manifest.json`

额外审计：`output_audit.json`、`descriptive_summary.json`、`exact_mode_ablation.parquet`、`exact_mode_ablation_summary.json`。

## 26. 最终测试

```text
59 passed in 1.69s
No broken requirements found.
```

测试命令使用：

```powershell
$env:PYTHONPATH = (Join-Path ([IO.Path]::GetFullPath('.')) 'src')
& 'C:\Users\Guoji\anaconda3\envs\libero-exp1\python.exe' -m pytest -q -p no:cacheprovider --basetemp=runs\exp7_pytest_final2_20260814
& 'C:\Users\Guoji\anaconda3\envs\libero-exp1\python.exe' -m pip check
```

## 27. 下一步

下一次不应直接做 latent RL。最小、主线一致的下一步是：在新的独立 cohort 上，用连续 contact-frame descriptor（nearest points、normal/tangent basis、signed gap、normal velocity、contact age、reference contact impulse/force、EEF/object relative pose）条件化局部 operator，再检验跨 demo reuse 是否显著优于 exact mode、time、progress 和 EXP5 state descriptor。具体协议见 `reports/next_exp_from7.md`。
