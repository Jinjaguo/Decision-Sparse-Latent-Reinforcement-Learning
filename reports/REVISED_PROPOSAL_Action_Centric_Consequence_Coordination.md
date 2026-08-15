# 修订提案：Action-Centric Consequence Coordination

## 核心假设

机器人恢复不需要先学习一个统一 latent RL policy。更可检验的假设是：从独立示范构造多个具有不同后果偏好的闭环动作专家，在线观察精确成功、任务进展、接触和力，再用任务模块化级联保持已有成功并恢复失败。

## 方法

协调器使用当前机器人/物体状态检索独立示范动作块。候选评分包括局部状态相容性、示范后继状态到目标状态的距离、经验响应与当前目标方向的一致性、平滑和 medoid 稳健聚合。每个任务冻结不同的专家顺序：Drawer 为 goal→smooth→response；Bowl 为 smooth→goal；Stove 为 default-70→soft→goal→response。所有候选隐藏目标 demo 的未来动作，成功即停，安全由任务级专家相对力包络和 1000 N 紧急上限约束。

## 主要比较

比较固定默认、固定专家、一次性路线选择、固定时间回退、检索进展切换、物理进展切换、任务统一策略、任务模块化、无 force guard、soft scaling、retract guard、默认前缀长度、专家顺序和候选 oracle。主指标为 safe success、default-demand recovery、oracle-headroom capture、安全停止和任务级非劣性。

## 验证设计

协议和主路线必须在正式结果前提交；分支 ID 必须完整唯一；目标 future、专家 suffix 和 post-action outcome 不得进入候选；同 demo 分支应聚类 bootstrap。开发成功门槛沿用 EXP27，但下一阶段必须加入全新任务/布局/示范的独立确认，不能只换同一 demo 的分支时间。

## 允许的主张

当前证据支持：在测试的三任务同运行时恢复基准中，成功保持、任务模块化的动作后果级联优于默认并通过预设效用和安全门槛。当前证据不支持通用决策稀疏、latent RL、部署安全或跨任务统一策略。

