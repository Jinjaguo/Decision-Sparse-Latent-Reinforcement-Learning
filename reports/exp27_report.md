# EXP27 实验报告：成功保持的多样化 Stove 级联

EXP27 针对 EXP26 的候选同质化问题，恢复了 default、soft-force、goal-consequence、response、phase-risk 和 medoid 等多种 Stove 机制。校准在 demos 28–29 上选择并冻结 `V0_default70_soft_goal`：先执行默认模块最多 70 步，未成功则依次进入 soft-force 平滑、目标后果和响应对齐专家。Drawer 保留 goal→smooth→response 的检索进展回退，Bowl 保留稳定 smooth→goal。

正式集使用 demos 0–9 中与 EXP25 完全不相交的 0.28/0.68 新分支时间，共 60/60 唯一状态。7 条授权路线执行 420 次 rollout、25,047 步。冻结主路线 safe success 93.33%，默认 66.67%，提高 26.67 个百分点；默认失败需求恢复 90%，candidate-oracle headroom capture 80%，安全停止 0，默认为 6.67%。Drawer/Bowl/Stove 分别完成 16/20、20/20、20/20，三个任务均不劣。

最终审计 `exp27_s5_final_audit_20260815` 的 12 项检查全部通过：路线矩阵、60 个唯一分母、有限状态、目标未来隔离、专家路径隔离、目标 demo 排除、协议和分支哈希、冻结主路线、指标重算和 pooled gate 均有效。校准+正式 pooled safe success 为 91.67%，默认 65.48%，需求恢复 82.76%，headroom 78.57%，安全停止 1.19% 对 8.33%。

按 30 个 demo 聚类的 5,000 次 bootstrap 中，总体提升 95% CI 为 `[0.133, 0.400]`，需求恢复为 `[0.750, 1.000]`。headroom capture 点估计通过，但区间 `[0.591, 0.955]` 较宽，因此结论应限定为本开发基准上的结构成功，不宣称部署级稳定性。

协议提交 `42377d6`，主路线冻结提交 `b48e2a1`，最终审计代码提交 `308e594`。全仓验证为 158 tests passed，`pip check` 和 `compileall` 通过。

