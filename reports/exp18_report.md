# EXP18 实验报告：Recovery Action-Consequence Coordinator

EXP18 在 672 个候选、84 个 same-state groups 上做 complete-demo 五折。输入仅为当前状态、route identity 和第一动作；目标 future、post-action state、outcome 和真实 steps 均被禁止。

冻结主 conservative ensemble 的 safe success 为 66.67%，默认 58.33%，只提高 8.33 个百分点；需求恢复 34.29%，oracle headroom capture 24.14%，因此失败。context-kNN 最好学习路线为 71.43%（+13.10 点），但需求恢复仅 40%；固定 weighted-k9 为 75%，需求恢复 51.43%。所有学习路线 safety stop 均不高于默认，说明失败是 rescue discrimination，不是安全。

按任务看，主 ensemble 在 Stove 从 25% 提到 57.14%，Bowl 持平 100%，却把 Drawer 从 50% 降到 42.86%。EXP19 因此采用 demand-gated listwise branch-vector 模型，先保住预计成功的默认，再只在预计需求组中选择互补 rescue route。

首次 `_s1` 数据构建遇到已成功 Bowl 分支零步结束、没有第一动作；失败 run 保留。修复加入 `no_action_needed` 当前状态标记和零动作占位，`_r1` 完成。协议提交 `6218c96`，修复 `cb308db`。

