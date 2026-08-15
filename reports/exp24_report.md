# EXP24 实验报告：任务模块化后果协调

EXP24 将 Drawer、Bowl、Stove 分别交给 guarded goal、稳定反馈和已确认 soft-force 模块。120 条 manifest 运行了 960 次 rollout、68,961 步；但一个 Stove 分支 ID 重复，分析只有 119 个唯一分支，因此完整分母门槛失败。

在这 119 个唯一分支上，冻结主路线 safe success 87.39%，默认 70.59%；需求恢复 74.29%，安全停止从 10.92% 降到 1.68%。Drawer/Bowl/Stove 分别为 31/40、40/40、33/39。总体 oracle-headroom capture 66.67%，仍低于 75%，所以即使忽略分母问题也失败。Drawer 所有路线并集达到 38/40，说明缺口仍是在线专家切换。

EXP25 改用真实抽屉关节位移作为进展后果，比较多个专家顺序和物理停滞窗口；新确认集 demos 0–9 必须严格 120/120 唯一。运行：`exp24_s3_confirmation_task_modular_20260815`；分析：`exp24_s4_confirmation_analysis_20260815`。

