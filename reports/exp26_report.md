# EXP26 实验报告：冻结任务模块结构确认

EXP26 将 EXP25 中通过所有门槛的 U7 原样冻结，在最后未做 recovery 结果的 demos 28–29 上运行 192 次 rollout、18,126 步。确认集严格 24/24 唯一。

U7 safe success 79.17%，默认 62.5%，提高 16.67 点；headroom capture 80%，安全停止从 12.5% 降至 4.17%。但默认失败分支只恢复 5/9，即 55.56%，低于 60%，因此 EXP26 失败。Stove 候选 oracle 也只有 4/8，与 U7 相同，说明同质化 Stove 路线没有提供足够需求恢复机会。

EXP27 恢复默认前缀、soft-force、goal、response、phase-risk 和 medoid 等多种 Stove 机制，并在 demos 0–9 的全新分支时间上确认。运行：`exp26_s1_confirmation_frozen_structure_28_29_20260815`；分析：`exp26_s2_confirmation_analysis_20260815`。

