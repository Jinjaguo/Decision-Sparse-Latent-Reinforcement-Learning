# EXP25 实验报告：真实物理进展多专家协调

EXP25 用当前中层抽屉关节位移替代检索索引作为停滞信号，并在严格 120/120 唯一的 demos 0–9 确认集上运行 960 次 rollout、73,264 步。冻结 U0 safe success 85.83%，默认 62.5%；需求恢复 68.89%，安全停止 0.83% 对 10%。但 headroom capture 为 68.29%，所以 EXP25 失败。

关键消融 U7 没有物理停滞层，而是保留任务模块化的检索进展控制。它在同一新确认集上达到 89.17% safe success、77.78% demand recovery、78.05% headroom capture 和 0 安全停止，全部门槛通过。因为 U7 不是预先冻结主路线，不能追认 EXP25 成功。结果说明直接关节进展虽合理，但当前停滞切换会过早放弃可恢复专家；更简单的任务模块结构反而最好。

EXP26 将 U7 原样冻结，并在最后未做 recovery 结果的 demos 28–29 上前瞻确认；必须结合 EXP25 的 120 分支证据解释小确认集。运行：`exp25_s3_confirmation_physical_progress_0_9_20260815`；分析：`exp25_s4_confirmation_analysis_20260815`。

