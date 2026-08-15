# EXP23 实验报告：风险感知后果仲裁

EXP23 先在 120 个 leave-one-demo-out 分支上比较撤离防护、软力缩放、目标后果、平滑响应、进展停滞和当前状态路线选择。候选并集完成 119/120；冻结 `P5_soft_force_scaling`，校准 safe success 82.5%、需求恢复 73.91%，安全停止 2.5%，明显优于默认的 61.67% 和 15%。

原计划 demos 50–56 超出数据集索引，失败 run 被保留；在任何候选执行前，确认集前瞻性改为共同可用的 demos 21–27。确认共 84 分支、756 次 rollout、58,315 步。P5 达到 84.52%，默认为 65.48%；需求恢复 72.41%，安全停止从 10.71% 降到 0。Stove 完成 26/28，并捕获 86.67% headroom，说明风险防护成功。

总体 oracle-headroom capture 只有 57.14%，原因转移到 Drawer：P5 17/28，而候选 oracle 27/28；Bowl 为 28/28。EXP23 因唯一 headroom 门槛失败。EXP24 将任务模块解耦：Drawer 使用在该确认集达到 23/28 的 guarded-goal 模块，Stove 保留 P5，Bowl 保留稳定反馈，并在另一批未做 recovery 结果评估的 30 个成功参考上确认。

确认运行：`exp23_s4_confirmation_risk_guard_21_27_20260815`；分析：`exp23_s5_confirmation_analysis_20260815`；冻结主路线提交：`c58b7b7`。

