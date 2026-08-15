# Next experiment from EXP12

## Current evidence

EXP12 在 84 个 same-state candidate groups 上找到约 71.6% 的 pairwise ranking signal，但所有安全的 setwise route 都退化为 nominal。只有 10 个 groups 存在超过 0.05 的 nominal improvement，且全部来自 Stove；Drawer/Bowl 均为 0。620 维 richer-future baseline 也没有超过 nominal。

## Remaining bottleneck

当前首要瓶颈不是 predictor capacity，而是候选动作缺少真实正向机会。一个 selector 无法选择不存在的好动作。现有 analytic family 主要围绕 dominant channel 对称扰动；Bowl 又因 I-A fidelity gate 只剩 nominal 与两个 phase edits，candidate source 明显过窄。

## Next concrete module

EXP13：Task-Aware Candidate Generation Diversity。它只评估候选生成，不训练新的 coordinator，也不做闭环控制。

## Why highest value

只有先证明多个任务在同一状态下经常存在比 nominal 更优且执行有效的候选，EXP14 才有资格重新测试 action selection。否则任何 ranking 模型改进都只是在复现 nominal 或挑选较少坏的 replacement。

## What must not be repeated

不要只增加 amplitude、只放宽 clipping threshold、只改变 composite 权重、或再训练更大 ranking network。不要把更大的 macro effect 当成更好的动作；候选必须按 terminal success、task-native progress、contact risk 和执行 fidelity 共同审计。

## Proposed alternatives

比较多条独立路线：多通道解析 smooth modes、时间 warp/phase edits、training-only task×phase residual modes、跨 demo action library、任务进度定向 residual、EXP12 predictor-guided local proposal、精确 gripper timing，以及受限的两模式组合。先在 calibration states 上执行并按 fidelity、diversity、oracle opportunity 授权，再在 development-formal states 上验证。

## Data requirements

复用 corrected-D snapshots。EXP11 calibration episodes 41–42 只做 family authorization；formal episodes 43–49 做 development-level held-demo evaluation。任何 basis、nearest-neighbor 或 guided direction均不能使用目标 demo 的 outcome。保留 matched-zero、requested/executed、clipping、contact、task-object trajectory、predicate 和 raw hashes。

## Success/failure interpretations

如果至少 30% groups 存在有效 oracle improvement，且至少两个任务各自达到 20%，candidate generation 模块才算 solved，并进入新的 ranking experiment。如果只有一个任务改善，保留 task-specific generator，下一 EXP 解决跨任务缺口。如果所有多路线仍不能创造机会，说明围绕 successful demonstration 的局部 action chunks 不足，下一 EXP 应转向 task-space targeted planning或新的 proposal source，而不是停止研究。

