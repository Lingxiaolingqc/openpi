# RedCubeToBox：从 legacy 到 polar 的问题定位与解决记录

## 1. 文档范围

本文只记录最终通向 `autogen_polar_retreat_transport` 的问题链。时间线从最初的 RedCubeToBox
`legacy` 专家开始，经过第一条继承 legacy 的 Autogen 风格搬运路线、独立重写的
`autogen_independent_retreat_transport`，最后到当前的
`autogen_polar_retreat_transport`。

没有展开记录 adaptive、servo、weighted servo、jaw-frame、axis-align slow-grasp 等并行专家的完整实验史；
只有当它们已经证明的结论直接影响 polar 设计时，才引用相应结论。例如：SO-101 的五个手臂关节难以稳定
满足完整 6D 末端位姿，完全 position-only 又会留下危险的自由姿态，因此 polar 最终采用“位置主任务 +
入口实测关节姿态的软零空间保持”。

本文根据 Git 历史、`examples/so101/0.originalCodeChanges.md`、保留下来的 smoke 日志以及本轮对话中的
逐次诊断重建。早期只有现象而没有完整逐帧日志的地方，会明确写成“当时观察”而不是事后伪造精确数据。

## 2. 路线关系

```text
RedCubeToBoxStateMachine（legacy）
├─ autogen_retreat_transport
│  └─ 继承 legacy 的抓取和抬升，只替换 retreat/transport
└─ autogen_independent_retreat_transport
   └─ 不继承 legacy 状态机，独立实现抓取、搬运、release gate 和诊断
      └─ autogen_polar_retreat_transport
         └─ 复用 independent 的基础阶段，重点重写持物路径和 IK 运行模式
```

设计过程中一直保留不同专家名称，没有用新实验覆盖旧专家。这样相同 seed 可以在独立进程中做公平对照，
失败方案也能继续作为消融基线。

## 3. 坐标和控制点约定

后续很多问题都来自“目标点到底属于谁”，因此先固定术语。

- `robot root`：机械臂根节点，当前审计位置约为 `(0.35, -0.64, 0.01)`。
- `gripper` / `gripper frame`：IK 默认末端控制点，对应 `ee_frame.target[0]`。
- `jaw detection frame`：闭合夹爪附近的检测点，对应 `ee_frame.target[1]`。
- `cube`：红色方块的刚体中心。
- `box floor center`：目标盒底板中心，当前为 `(0.20606, -0.40428, 0.04546)`。
- `wrist`：gripper 上游刚体。retreat 和圆弧阶段后来改为控制 wrist，目的是减弱 gripper 世界姿态对整臂的
  强耦合。

关键经验：把 gripper 原点送到 box center，不等于把 cube 送到 box center。持物时必须考虑
`cube/jaw - gripper` 的实时杠杆偏置。

## 4. legacy 基线：先证明场景、抓取和落盒本身可行

### 4.1 初版抓取失败：夹爪命令不够闭合

初版 legacy 使用九阶段绝对末端位姿 IK：接近、下降、闭合、抬升、横移、入盒、释放、撤离、稳定。
动态日志显示，抬升前 jaw 到 cube 距离已经约为 `18.26 mm`，但 gripper 关节停在约 `0.45639 rad`，高于
LeIsaac `pick_cube` 使用的 `0.26 rad` 阈值。

根因不是抓取位姿没到，而是闭合目标过松。

解决：把 RedCubeToBox 的闭合目标从通用的 `0.4 rad` 改到 `0.05 rad`，不改已经接近正确的末端位置。

### 4.2 闭合后 jaw frame 移位，打开状态下对准不代表闭合后对准

进一步运行时，gripper 能闭到约 `0.14367 rad`，但闭合后的 jaw frame 与 cube 距离变成约 `27.57 mm`。

根因是夹爪开合会改变 jaw detection frame 的位置。以打开夹爪时的 jaw 几何做抓取目标，会在闭合后产生
系统偏差。

解决：抓取世界 XY 各修正约 `10 mm`，同时降低约 `20 mm`，让闭合后的 jaw 而不是打开时的 jaw 对准 cube。

### 4.3 先把盒子碰撞和机械臂问题分离

当时不能确定失败来自机械臂，还是目标盒的碰撞/成功判据。因此增加独立 drop smoke：把 cube 放到盒子上方，
完全依靠重力落下。

验证结果：cube 下落约 `0.12492 m` 后，以约 `0.000779 m/s` 稳定在盒内。这证明盒子碰撞和成功判据可用。

随后校正后的 legacy 成功：jaw-cube 距离约 `4.12 mm`，gripper 约 `0.14649 rad`，运输阶段保持抓取，
最终 cube 相对盒中心约 `(36.34, 6.42, 19.07) mm`，速度约 `0.001097 m/s`。

这个成功基线非常重要：后续 Autogen 路线失败时，不能笼统归因于资产或盒子物理。

### 4.4 从 legacy 获得的两个长期结论

1. 完整固定世界姿态有助于保持抓取几何，但对五关节 SO-101 来说，长期同时要求 XYZ 和三轴姿态会过约束。
2. 控制点必须明确。早期代码曾用 body 列表最后一项，实际拿到 jaw；后续统一使用命名的 gripper frame，
   jaw 只用于检测或显式 offset。

## 5. 第一条 Autogen 风格路线：`autogen_retreat_transport`

这一变体仍继承 legacy，只替换抓取后的高层路径：先内缩 retreat，再 transport 到盒子。

### 5.1 最初释放 yaw，cube 在运输后半段掉落

第一版 retreat 把 gripper 相对 root 的 XY 半径缩为 `5/7`，并抬高；transfer 再从实际入口移动到实时 box
center。为了降低姿态约束，曾使用 yaw 自由的 `xyz_tilt`。

日志显示 `cube - gripper` 的 XY 关系从 late-lift 的约 `(22.8, 31.9) mm` 漂到 transfer 入口约
`(63.3, 43.9) mm`，随后掉落。

结论：简单释放 yaw 会让夹持杠杆方向漂移。即使 gripper 原点位置看似合理，cube 也可能被甩离。

当时解决：该变体恢复完整 pose IK，保留为与后续独立路线比较的基线。但完整 pose IK 的可达性问题并没有
真正消失，这也是后来独立重写和 polar 分段的动机。

## 6. `autogen_independent_retreat_transport`：独立重写后的问题

对应起始 commit：`c285d9d`。

### 6.1 为什么要独立重写

这一版本不继承 legacy 状态机，自己实现抓取关键帧、8 维 action 组装、阶段推进、安全 release 高度和失败
门控。目的不是立即替代 legacy，而是把 Autogen 风格 retreat/transport 与 legacy 内部历史逻辑分离。

初始路径在 close 后没有单独 lift，而是直接做组合 `retreat_to_safe`：

- 从实时 gripper 位姿开始；
- 同时上抬；
- root-relative XY 半径缩到 `5/7`；
- 组合参考速度从 `2.5 mm/step` 降到 `0.8 mm/step`。

### 6.2 上抬过程中掉 cube：开环参考与过约束 IK 产生实际运动偏差

第一轮日志中，cube 大约在 `retreat_to_safe` 第 76 个控制步掉落。当时曾把约 `70 mm` 增至 `82 mm` 的
Z 误差解释为“参考走得比真实机械臂快”，但复查源码后，这个说法并不严谨。

状态机每步实际发送的是按 `phase_step` 前进的瞬时参考：

```text
current_target = start + progress(step) * (final_target - start)
```

旧日志中的 `target_error` 却计算最终终点与实际 gripper 的距离：

```text
final_target_error = norm(final_target - actual_gripper)
```

它没有计算真正用于判断参考跟踪性能的：

```text
reference_tracking_error = norm(current_target - actual_gripper)
```

因此，`70–82 mm` 表示“距离最终终点还有多远”，不能证明真实机械臂落后当帧参考相同距离。这个指标适合
最终完成门，却不适合诊断瞬时跟踪延迟。它也不是掉落的直接原因：当时该误差只用于阶段完成和 timeout，
没有反馈到 IK action 的生成。

更根本的运动问题是三项叠加：

1. **参考按时间开环推进。** `progress` 只取决于 `phase_step`，不会因为机械臂未跟上而暂停或降速；IK、
   执行器、惯性和接触负载造成的正常跟踪延迟因而可能逐步积累。
2. **完整 6D 任务对五关节 SO-101 过约束。** XYZ 加三维世界姿态共有六个任务分量，五个手臂关节只能求
   最小二乘折中。上抬、内缩和姿态保持互相竞争，求解器可能用额外侧移或转动换取总任务误差下降。
3. **组合运动放大夹持载荷。** gripper/jaw 与 cube 之间存在杠杆臂；上抬和内缩同时发生时，IK 的姿态与
   侧向折中会转化为夹持接触上的复合载荷，最终使 cube 滑落。

所以不能把失败简单归因于“误差算错”或“IK 天生慢一帧”：误差公式误导了当时的诊断，但真正造成掉落的是
缺少跟踪误差调速的开环参考、欠驱动完整 6D IK 和持物接触动力学共同作用。

当时先增加两类安全修正：

1. retreat 完成从“精确 XYZ 到达”改为物理安全判据：高度进入安全带、root-relative 半径进入目标加余量。
2. 增加抓取锁存和丢失保护：retreat 入口 jaw-cube 距离不超过 `15 mm` 才确认；确认后超过 `25 mm`
   立即写入 `servo_abort_reason` 并停止。

这些修正能够区分“参考仍在推进”“最终目标不可达”和“cube 已掉落”，但没有从控制结构上消除根因。后来的
polar 进一步通过垂直上抬与径向运动分段、降低 IK 约束、切换 wrist/gripper 控制点、软姿态保持和累计关节
目标，才真正减少不可达折中与夹持侧向载荷。

### 6.3 两段式 retreat 曾经实现，后来又回退

针对复合运动掉落，曾把 independent retreat 拆成先垂直、再水平两个内部段（`9cf63bf`）。但为了做严格的
速度单变量对照，随后又在 `cc40169` 回退为单段组合 retreat，只保留安全门和慢参考。

这个回退不是“两段式被证明错误”，而是当时实验目的要求先确认速度本身的影响。后来的 polar 又重新采用
明确分段，并最终证明分段更适合当前机械臂。

### 6.4 只看半径和高度会产生假阳性

independent 后续日志显示，目标 X 约为 `0.3505 m`，实际 gripper X 却漂到约 `0.4528 m`；如果只检查
root-relative 半径和 Z，仍可能错误进入下一阶段。

解决思路：后来的 polar 不只检查半径和高度，还检查真实控制点、参考是否走完、bearing 误差及连续稳定步数。

### 6.5 hover 高度过高造成不必要运动

independent 的 retreat/transfer hover 从 `floor + 0.25 m` 降到 `floor + 0.22 m`。盒沿顶约为
`0.10946 m`，目标 gripper Z 约 `0.26546 m`，仍保留约 `61 mm` jaw 净空，同时减少额外抬升和后续下降。

## 7. polar 初版：为什么采用极坐标分段

对应起始 commit：`07652a6`。

polar 保留 independent 作为对照，把持物路径拆成：

1. retreat；
2. 围绕 robot root 的定半径圆弧；
3. 沿 box bearing 的径向接近；
4. lower、release、retract。

圆弧实现不是预计算关节轨迹，而是在世界坐标中计算连续笛卡尔参考：

- 入口半径 `r = ||start_xy - root_xy||`；
- 起点和 box 的 bearing 使用 `atan2`；
- 通过 smootherstep 逐步插值 bearing；
- 每步目标为 `root + r * [sin(bearing), cos(bearing)]`；
- Z 保持入口实际安全高度；
- 每步再由差分 IK 转成关节目标。

这样将“改变朝向”和“改变半径”分开，理论上比一条直线同时穿过多个约束更容易诊断。

## 8. polar 早期 IK 约束路线

### 8.1 完整 6D pose IK：不是路径公式错，而是任务过约束

第一轮 polar 日志中 cube 没掉，jaw-cube 最终约 `11.8 mm`，但 retreat 失败：

- 实际 X 从约 `0.3507 m` 漂到最高约 `0.4582 m`；
- bearing 误差超过 `1 rad`；
- Z 从约 `0.26546 m` 的目标越到约 `0.3558 m`；
- shoulder-pan 从约 `0.1165 rad` 漂到约 `1.5170 rad`。

根因是五个手臂关节无法稳定满足 XYZ 加完整三轴世界姿态。状态机看起来像“往目标反方向走”，实际上是 IK
在不可兼容约束之间选折中分支。

### 8.2 改为 `xyz_tilt` 后改善，但仍会掉落

`c605bd3` 将 retreat 改为 XYZ + world roll/pitch，释放 yaw，并加入三类保护：

- Z 超调 `20 mm`；
- bearing 偏差过大；
- 参考结束后误差连续恶化。

位置误差从约 `119.5 mm` 降至 `52.5 mm`，但 X 仍差约 `45 mm`、bearing 约 `13.6°`；控制步约 473
时 jaw-cube 距离从 `3.15 mm` 增到 `25.96 mm`，cube 掉落。

结论：释放 yaw 只减少一部分过约束；同步运动、自由底座分支和夹持侧向载荷仍存在。

### 8.3 尝试 `vertical_lift + 向内 30 mm` 和硬 shoulder-pan

随后将 retreat 拆成：

- 固定入口 XY 的垂直上抬；
- 固定安全 Z/bearing、向内 `30 mm` 的短径向 retreat。

两段使用 `xyz_pitch_joint`：XYZ、base-frame pitch 和入口 shoulder-pan。这个五行任务把五个手臂自由度几乎
全部耗尽，接近 pickup 构型时容易病态，jaw 会在名义垂直上抬中横摆。

解决：取消硬 shoulder-pan 任务，先恢复 `xyz_tilt`，随后进一步转向 wrist position-only + 软姿态。

### 8.4 不再固定“向内 30 mm”，恢复 root-relative `5/7`

实测中短退目标可能把机械臂推入不利分支。按日志具体 target 对比后，放弃固定向内 `30 mm`，恢复按 robot
root 的 X/Y 相对位置缩放到 `5/7`。这是几何比例目标，不是简单对某一个世界轴减常数。

## 9. close_gripper 为什么会“停住”

### 9.1 固定时长或要求命中名义闭合角都不可靠

夹到 cube 后，真实 gripper 关节会因接触停在某个折中角，不一定达到无负载时的名义闭合目标。如果阶段只看
目标角误差，就会一直 close；如果只用固定步数，又可能在夹爪仍高速运动时直接进入 lift，导致 cube 被甩出。

### 9.2 改为反馈式 settle gate

`0c76320` 移除 close 的固定阶段长度，改成：

- 至少执行 80 步；
- gripper 至少过半闭合；
- jaw-cube 距离在抓取确认阈值内；
- 连续稳定若干步后才能 retreat；
- 最多 200 步，否则安全失败。

早期还把 gripper 低速度作为硬条件，但接触抖动会让它长期无法满足。后续速度改为诊断信息，不再阻挡已经
几何确认的抓取。核心教训是：接触系统不能只用“达到空载目标角”判断完成。

## 10. retreat 改用 wrist 控制

### 10.1 动机

gripper 世界姿态会通过较长杠杆影响整条机械臂。retreat 的目标本质是把持物系统抬到安全区，不需要强制
gripper 在世界中保持完整角度。因此改用 wrist 作为 IK 控制刚体，让 gripper/jaw 的姿态变化不再直接成为
主任务硬约束。

### 10.2 最终形式：wrist XYZ position-only + 入口关节软姿态

`8c5aae2` 扩展 `phase_aware_ik_action.py`，允许运行时切换 control body。retreat 和 arc 控制 wrist；主任务只
解 XYZ，同时把入口五关节实测姿态作为软 nullspace posture：

```text
delta_q = primary_XYZ_delta + (I - J#J) * posture_delta
```

软姿态不是额外硬任务行，只在不破坏 XYZ 主任务的零空间中发挥作用。它抑制 position-only 的自由关节漂移，
但不会像完整世界姿态那样占满自由度。

### 10.3 放开 Y 的实验为什么偏得更严重

曾尝试在 retreat 中不约束 Y，希望避免为了同时满足 Y/Z 而走到另一 IK 分支（`d4382b5`）。日志显示偏移反而
更大：一旦 Y 不进入主任务，IK 可以用更大的侧向运动换取 X/Z 和软姿态折中。

解决：恢复 Y 约束（`45bd712`）。垂直 lift 固定入口 X/Y；后续 root-relative retreat 同时约束 X/Y/Z。

这个实验说明“少一个约束”不一定更稳定。应该释放的是不必要的世界姿态，而不是路径定义本身需要的平移轴。

## 11. 阶段完成条件必须围绕实际安全目标设计

### 11.1 lift 不应被微小 XY 折中卡住

垂直 lift 的安全目的只是达到足够 Z。若参考已经走完且 wrist Z 进入容差，微小 XY 误差不应该让阶段无限
等待。因此 lift 完成改为看实际 wrist Z，并在进入下一段时从实际位置 rebase。

### 11.2 root-relative retreat 需要半径、Z 和 bearing

第二段不能只看三维欧氏误差，也不能只看半径：

- 半径保证退到安全尺度；
- Z 保证净空；
- bearing 防止进入“半径正确但在另一条射线”的错误分支；
- 连续稳定步数避免瞬时穿过阈值。

### 11.3 日志按阶段关注真正的控制点

为了避免打印一大串难读字段，日志后来按阶段切换关注点：

- retreat、arc：打印 wrist；
- radial、lower、release、retract：打印 gripper；
- 抓取阶段：打印 gripper/jaw/cube 和夹爪反馈。

所有小字段分行，包括 motion start、motion target、current target、actual focus、target error、bearing、稳定计数、
IK singular values、关节目标和实际差值。这个变化多次帮助区分“笛卡尔目标反了”和“关节执行目标在漂”。

## 12. position-only 已启用，为什么后半段仍会反向走

### 12.1 原始差分 IK 目标每步重新锚定实际关节

原始 action term 的典型形式是：

```text
q_target(k+1) = q_actual(k) + limited_delta_q(k)
```

当执行器明显滞后或带负载漂移时，下一步目标会跟着实际关节一起移动。日志中即使 IK delta 已请求反向制动，
目标仍可能被新的实际位置向前带走，于是视觉上出现“后半段突然往目标反方向走”。

这不是单纯的 target XYZ 算错，而是关节目标参考系不持久。

### 12.2 增加逐关节诊断

`0356394` 增加以下日志：

- `controlled_joint_names`；
- `last_joint_position_target`；
- `current_controlled_joint_position`；
- `joint_target_minus_actual`；
- `actual_joint_velocity`；
- primary/nullspace/final IK delta；
- task singular values。

这些字段把“求解器请求了什么”“发送的关节目标是什么”“执行器实际怎么动”分开。

### 12.3 使用独立累计关节目标

`45b1255` 改为：

```text
q_accum(k+1) = q_accum(k) + limited_delta_q(k)
```

每次 IK 应用最大关节分量限制为 `0.005 rad`，累计参考不再每步重新锚定 live joint。这样反向制动目标能持续
存在，而不是被实际漂移吞掉。

## 13. radial 阶段仍需切回 gripper，并在 handoff rebase

retreat/arc 控制 wrist 是为了安全抬升和圆弧；最终放置却必须知道 gripper/jaw 的实际位置。因此 radial 不能
继续只控制 wrist。

`0bd8b68` 做了三件事：

1. radial 切回 gripper XYZ position-only；
2. 在 radial handoff 捕获当时实际五关节姿态作为新的软 nullspace posture；
3. 累计关节目标重置到 handoff 的实际关节位置。

这避免把 wrist 阶段已经积累的大关节 target gap 原封不动带进 gripper radial 阶段。

## 14. tracking-gap / anti-windup：合理意图，错误结果

### 14.1 为什么加 `0.15 rad` 限制

累计目标解决了 live reanchor，但会产生较大的 `target - actual`。为防止无限积累，曾增加 `0.15 rad` tracking
gap 限制，随后改成 anti-windup：当某关节 gap 已到上限且新的 IK delta 会进一步扩大 gap 时，冻结累计目标。

### 14.2 实际日志证明冻结切断了必要纠偏

anti-windup 按代码工作，但 shoulder-pan 目标被冻结在约 `0.03786 rad`，实际 shoulder-pan 继续从约
`0.1928 rad` 漂到约 `0.50 rad`。圆弧目标 wrist 为约
`(0.27866, -0.52317, 0.28783)`，实际停在约
`(0.37887, -0.56410, 0.28985)`，误差约 `108 mm`，bearing 误差约 `0.912 rad`。

根因：冻结策略只阻止 target 继续远离 actual，却没有让执行器更快回到 target；关键关节失去持续纠偏后，
整条圆弧无法跟踪。

### 14.3 最终解决

`6e0d548` 完全移除 tracking-gap clamp/anti-windup，恢复不受 gap 限制的累计目标，但保留：

- 每步 `0.005 rad` 限速；
- soft joint limits；
- radial handoff 的实际关节 rebase。

这里的关键是边界放在“每步变化”和“阶段 handoff”上，而不是在运动中冻结一个已落后的目标。

## 15. lower 卡在折中解

### 15.1 现象

一次日志中 lower 目标为：

```text
(0.20606, -0.40428, 0.17946)
```

实际最终约为：

```text
(0.23235, -0.38928, 0.19538)
```

Z 只差约 `15.9 mm`，XY 合计偏约 `30 mm`，三维误差因此稳定在约 `34.2 mm`，状态机一直等待到 timeout。

### 15.2 根因和修复

lower 重新使用完整世界姿态后，IK 在下降位置和姿态之间选折中解。修复 `0bc130f`：

- lower 也使用 gripper XYZ position-only；
- 入口捕获实际关节姿态作为软 nullspace posture；
- 重置累计关节目标；
- 阶段完成只看实际 gripper Z 是否进入 `15 mm` 容差并连续稳定 10 步。

这是“完成条件围绕阶段目的”的应用：lower 的主要安全目的为达到 release 高度，不能让不重要的 XY 折中误差
永久阻塞。

但这项修复还没有解决最终落点，因为当时 radial 的目标控制点本身选错了。

## 16. 最终关键问题：把 gripper 对盒中心，而不是把 cube/jaw 对盒中心

### 16.1 Windows 原生 run2 给出的证据

有效失败日志：

```text
D:\Sim\results\expert-smoke\polar\polar-seed42-0bc130f-run2.stdout.log
```

radial 入口：

```text
gripper = (0.23806, -0.50220, ...)
cube    = (0.20575, -0.45119, ...)
```

cube 相对 gripper 的 XY 偏置约为 `(-32 mm, +51 mm)`。旧代码却直接令：

```text
gripper_target_xy = box_center_xy
```

因此即使 gripper 完美到 box center，cube 也会落在 box center 的左前方。lower 又把 gripper XY 拉回中心，
进一步破坏 radial 已形成的持物对齐。release 前 cube 已约在 `(0.158, -0.330)`，最终相对盒中心约
`(-38 mm, +109 mm)`。

### 16.2 修复 `85d19d8`

radial 入口读取实时 jaw 与 gripper：

```text
jaw_from_gripper_xy = jaw_xy - gripper_xy
gripper_target_xy = box_center_xy - jaw_from_gripper_xy
```

并同步修正后续阶段：

- lower 从 radial 实际 handoff XY 垂直下降，不再把 gripper 原点拉到 box center；
- release 在实际 lower handoff 位姿原地打开；
- retract 从实际 release XY 垂直上抬；
- retract 完成同样只看 Z 容差。

这次修复将“轨迹目标属于哪个刚体/控制点”彻底明确下来。

## 17. Windows 原生环境审计与最终成功

### 17.1 安装审计

本机环境：

- Python `3.11.15`；
- PyTorch `2.7.0+cu128`；
- Isaac Sim `5.1.0.0`；
- IsaacLab `2.3.0`；
- LeIsaac commit `1651c321e9b0c1bb54233211fc7b3cd70d8373d5`；
- GPU：RTX 4060 Laptop 8GB，`cuda:0`；
- SO-101 USD SHA-256：
  `64A877C3B82CDC4A48AB8A1F321A2DD3EF7C55D4B10BCE222B58C530D978AE58`。

环境没有 `pip` 模块，但包元数据、pytest、CUDA tensor、Isaac Sim 和 IsaacLab 均可用。这是 uv 创建环境的
状态，不需要为了运行仿真从头重装。`isaaclab_tasks` 的顶层可选导入提示也没有阻止 OpenPI 自定义 task 创建。

### 17.2 scene audit

Windows 原生 scene audit 在单环境、headless、performance、`cuda:0` 下输出：

```text
RED_CUBE_TO_BOX_SCENE_AUDIT_OK
```

证明资产、场景、相机、FrameTransformer 和 GPU 路径可用。

### 17.3 第一次带最终修复的成功 smoke

日志：

```text
D:\Sim\results\expert-smoke\polar\polar-seed42-placement-offset-run3.stdout.log
```

结果：

```text
completed_steps: 1560
cube_final_pos_w: (0.19670, -0.38771, 0.06454)
cube_offset_from_box: (-0.00936, 0.01657, 0.01908)
cube_final_speed: 0.000277
servo_timeout_phase: None
release_block_reason: None
expert_success: True
RED_CUBE_TO_BOX_EXPERT_SMOKE_OK
```

相对 run2 的 `(-38 mm, +109 mm)`，最终 XY 偏移缩小到约 `(-9.36 mm, +16.57 mm)`，cube 位于盒内且已稳定。

### 17.4 带 recordings 的重复成功

为了排除“只成功一次”或录制造成不同负载的问题，又以 `record_every=4`、`record_fps=15` 重跑。结果与 run3
一致，生成 394 个文件、约 10.9 MiB：

```text
D:\Sim\results\expert-smoke\polar\recordings\red_cube_20260813-120216
```

离线 viewer：

```text
D:\Sim\results\expert-smoke\polar\recordings\red_cube_20260813-120216\
autogen_polar_retreat_transport-seed42-20260813-040256-pid4072\index.html
```

## 18. 最终有效的设计原则

### 18.1 路径层

- 抓取后先垂直达到安全高度，再改变水平半径或 bearing。
- 圆弧只改变 bearing，固定入口实际半径和安全 Z。
- radial 只改变沿 box bearing 的半径。
- lower/retract 使用实际 handoff XY 的垂直路径，不在低高度引入横移。

### 18.2 IK 层

- retreat/arc 控制 wrist XYZ，而不是强制 gripper 完整世界姿态。
- radial/lower 控制 gripper XYZ。
- 使用入口实测五关节姿态的软 nullspace posture，避免 position-only 自由漂移。
- 关节 target 使用独立累计参考，每步最大 `0.005 rad`。
- 不使用会冻结关键纠偏的 tracking-gap anti-windup。
- 每次控制刚体或阶段语义切换时，从实际关节状态 rebase 累计目标。

### 18.3 状态机门控层

- close 由接触几何和最小/最大步数门控，不强求空载名义闭合角。
- grasp 确认后持续监控 jaw-cube 距离，超过阈值立即失败。
- lift 完成看实际安全 Z。
- retreat 看 radius、Z、bearing 和连续稳定。
- lower/retract 看实际 Z 和连续稳定。
- release 必须在上游 placement 已对齐后原地发生。

### 18.4 控制点层

- 日志和目标必须明确属于 wrist、gripper、jaw 还是 cube。
- 最终放置对齐的对象是 cube/jaw，不是 gripper 原点。
- jaw/gripper 偏置必须在持物实际姿态下测量；不能长期复用打开夹爪或早期姿态下的固定 offset。

## 19. 已证明不应直接重复的方向

1. **完整 6D 世界姿态贯穿 retreat/arc**：五关节 SO-101 会在位置与姿态之间选错误折中。
2. **完全 position-only 且不做软姿态保持**：自由关节/姿态会漂移，通过杠杆把 cube 甩离目标。
3. **硬锁 shoulder-pan 加 XYZ/pitch**：五行硬任务耗尽自由度，容易近奇异并横摆。
4. **为了减少约束而放开 retreat Y**：Y 会成为大幅侧向补偿方向，实测偏得更严重。
5. **每步用 `q_actual + delta_q` 生成新目标**：执行器滞后时目标会随实际漂移，反向制动无法积累。
6. **累计目标达到 gap 后冻结**：冻结不能改善执行器跟踪，只会切断必要纠偏。
7. **把 gripper 原点直接送到 box center**：忽略持物偏置，cube 必然系统性偏离。
8. **lower/retract 同时重新对 box XY 和 Z**：会在盒沿附近引入不必要横移，并可能卡在姿态折中解。
9. **只看单帧到达或固定阶段时长**：接触抖动和动态越过会产生假阳性，需要连续稳定门。

## 20. 关键提交索引

| Commit      | 作用                                                             |
| ----------- | ---------------------------------------------------------------- |
| `c285d9d` | 新增 independent retreat/transport 专家                          |
| `b450c9d` | 增加 independent retreat 安全判据、抓取丢失保护和诊断            |
| `9cf63bf` | independent retreat 两段式实验                                   |
| `cc40169` | 回退单段 retreat，保留为受控速度对照                             |
| `07652a6` | 新增 polar 圆弧/径向路径专家                                     |
| `c605bd3` | retreat 降维到`xyz_tilt` 并增加安全停止保护                    |
| `cc226bf` | 垂直 + 短径向 retreat 与 shoulder-pan 硬任务实验                 |
| `0c76320` | 移除硬 pan，加入 close feedback settle gate                      |
| `380d9a5` | 调整 polar 抓取闭合门控                                          |
| `8c5aae2` | retreat 改用 wrist 控制，扩展 phase-aware IK                     |
| `d4382b5` | retreat 放开 Y 的消融实验                                        |
| `45bd712` | 根据日志恢复 Y 约束                                              |
| `4dfaa0f` | 调整 polar 阶段完成门                                            |
| `351cdb7` | wrist/gripper XYZ position-only 与软姿态路线完善                 |
| `0356394` | 增加逐关节 IK target/actual/velocity 诊断                        |
| `45b1255` | 增加独立累计关节目标                                             |
| `0bd8b68` | radial 切回 gripper position-only 并在 handoff rebase            |
| `750c746` | tracking-gap anti-windup 冻结实验                                |
| `6e0d548` | 移除 anti-windup，恢复持续累计纠偏                               |
| `0bc130f` | lower position-only，并以实际 Z 完成                             |
| `85d19d8` | jaw/gripper 偏置反算、垂直 lower/retract、原地 release；最终成功 |

## 21. 后续复测建议

当前成功结论来自固定 seed 42 的 Windows 原生单环境重复运行，证明该状态机在当前资产和初始状态上可行，
但不等于随机化成功率已经充分验证。后续应保持以下顺序：

1. 先用不同 seed 的独立单环境进程复测，不在同一个长寿命 Isaac 进程中混用随机序列。
2. 统计 radial handoff 时的 `jaw_from_gripper_xy`、最终 cube offset 和最小盒沿净空。
3. 失败时先判断 grasp loss、IK timeout、placement offset 还是 success predicate，不直接放宽阈值。
4. 8GB 显存仍优先单环境、headless、performance；OOM 时先降低录制频率或关闭不必要录像，不改变物理逻辑。
5. 保留 legacy、independent 和 polar 三条路线，避免用新实验覆盖可比较基线。

## 22. State machine 目录整理

最终成功路线与仍在开发的 Autogen reference 路线保留在
`examples/so101/red_cube_to_box_task/`。失败、被取代或尚未证明成功的消融专家移动到同级
`failed/` 包，但 smoke 与 batch 的 `--expert` 名称保持不变，因此历史对照仍可直接运行。

`autogen_independent_retreat_transport` 是一个特殊情况：它自身属于失败对照，但成功的 polar 原先直接继承
它。为避免成功实现依赖 `failed` 包，原实现抽成当前目录的 `polar_base_state_machine.py`，类名改为
`RedCubeToBoxPolarBaseStateMachine`；polar 继承该活动基类，而 `failed/` 中保留使用旧类名的薄兼容子类。
这只改变代码组织和导入路径，不改变原 independent 专家的状态、动作或参数。

## 23. 随机 batch 的 close gate 集中失败（2026-08-13）

初次成功率验证中，polar 明显优于其他路线，但 7 个失败全部集中为
`gripper_not_settled_before_retreat`。这说明当时应先处理抓取后的物理判定，而不是修改已经能工作的运输路线。

逐帧日志显示两个旧假设不可靠：jaw detection 会在继续闭合时移动，因此已经建立的接触几何可能随后暂时
离开 `15 mm` 阈值；任务的 `pick_cube` 也可能只是短暂为真。最终保留方案是：

- jaw 进入确认距离后锁存“曾建立抓取几何”，不因随后 jaw frame 移动而清除；
- `pick_cube` 必须连续 3 帧才锁存，单帧反馈不直接放行；
- 仍要求 12 帧夹爪角窗口的跨度不超过 `0.01 rad`，连续稳定 8 帧；
- close 最长等待从 200 增至 320 步，但超时仍明确失败，不靠无限等待掩盖 miss；
- 把 target error、速度、角窗口、jaw 距离、feedback streak 和 stable streak 一并写入失败原因。

核心教训：接触成立、任务 proxy 为真和执行器孔径稳定是三个不同事实。可以锁存前两者，但进入运动前仍须
等待第三者；不能只放宽 jaw 距离或只延长阶段时间。

## 24. Batch 录像全黑但物理仿真仍在运行（2026-08-13）

录制目录和 JPEG 正常生成，但 batch 帧全部为黑。对比已验证的非 batch smoke 后确认，问题不在图片编码：
问题运行的原始 `policy.front` 张量本身就是全零。根因是启用离屏相机但未显式指定 rendering mode，
AppLauncher 的空覆盖继承了本机禁用 RTX 的持久设置。

解决方式：

- smoke/batch 在启用相机且用户未指定时自动选择 `performance`；
- 用户显式指定 `balanced/quality` 时不覆盖；
- 不添加 `--renderer_device`；
- 首帧打印 dtype/shape/min/max/mean，若 `max=0` 立即失败，不继续生成整批黑录像。

修复后首帧恢复到 `max=240`、均值约 `153.79`，非 batch smoke 同时保持成功。核心教训：录像 QA 必须检查
传感器原始像素，文件存在和相机 tensor shape 正常都不能证明 renderer 真正在输出图像。

## 25. 随机位置暴露盒子遮挡和远端不可达（2026-08-13）

录像显示部分失败并非 state machine 控制器本身：cube 离盒子太近时，固定爪会先被盒壁挡住；cube 离 robot
root 太远时，五关节机械臂无法让 gripper 与 cube 在 XY 对齐，只能进入折中 IK 构型。

原随机 batch 的 cube 中心距托盘外沿最小约 `31 mm`，同时存在相对 root 向前约 `335 mm` 的样本。最终：

- box center 从 `(0.20606, -0.40428)` 移到更安全且可达的 `(0.18, -0.43)`；
- cube reset 限制为 `x=(-0.02, 0.05)`、`y=(-0.06, -0.04)`；
- yaw 保留 `[-30°, +30°]`，没有通过取消姿态随机化规避抓取问题。

第一次几何修正后，10 个 episode 全部能建立抓取，整体成功率由 `3/10` 提高到 `5/10`。核心教训：随机化
范围是任务可行域的一部分。应先排除静态障碍重叠和真实工作空间不可达，再用控制器调参解释剩余失败。

## 26. Cube yaw 随机化后固定夹爪方向会夹飞方块（2026-08-13）

即使位置可达，夹爪闭合轴保持固定时，旋转后的方块会让爪面先撞到相邻边角，将 cube 横向挤开。reference
路线验证了可用的几何：以 gripper local `+X` 为闭合轴，在 wrist-roll 软限位内选择与
`{+cube X, -cube X, +cube Y, -cube Y}` 转角最小的无向轴。

polar 最终加入：

```text
descend
-> measured position settle
-> freeze other four arm joints
-> direct wrist_roll axis alignment
-> recenter while preserving measured aligned quaternion
-> close while holding aligned arm target
```

直接从固定时长 descend 跳到 align/close 的实验被否决，因为实际末端仍落后参考约 `43 mm`。必须先按实测
位置稳定，再旋转并 recenter。alignment 每步 target 增量仍限制为 `1°`，实测角误差门仍为 `5°` 且需连续
10 帧稳定；只把 target 相对 actual 的最大 lead 从 `2°` 增至 `4°`。同一 10 回合 batch 中，平均 alignment
由 `255.2` 降到 `188.8` 步、最大由 `564` 降到 `348` 步，最终最大实测误差仍为 `0.32°`。两版都抓到
10/10；剩余 4 个失败发生在 retreat，因此此时没有继续放宽 alignment 精度。

核心教训：单关节 direct control 能避免 IK 与另一个 writer 争用 wrist roll；速度优化应增加有界 target lead，
而不是放宽最终实测误差或稳定门。

## 27. Cube 旋转后仍用世界固定抓取偏移（2026-08-13）

### 27.1 新发现：jaw detection 不是夹持中心

为修复 axis alignment 后的抓取偏心，曾尝试：

```text
recenter_gripper_xy = actual_gripper_xy + (cube_xy - actual_jaw_xy)
```

这个公式隐含假设 `actual_jaw_xy` 是两爪间隙中心。实际 LeIsaac 配置中，`ee_frame.target[1]` 是 jaw
刚体加 `(-0.021, -0.070, +0.020) m` 偏移得到的检测点，位于可动夹爪最远端。上述公式会把可动爪端点
送到 cube 中心，从而把真正的夹持开口整体移到一侧。动态 smoke 因此在
`recenter_after_alignment` 超时并推动 cube。该实验被撤销，没有进入提交历史。

### 27.2 更根本的问题：标定偏移没有随夹爪旋转

原成功抓取目标使用世界系固定偏移：

```text
gripper_xy = cube_xy + (-0.020, 0)
```

它只在 gripper local `+X` 与世界 `+X` 重合时成立。新增 wrist-roll axis alignment 后，夹爪闭合轴会随
cube yaw 旋转，但该偏移仍固定指向世界 `-X`，于是 cube yaw 越大，夹持开口越偏。旧 10 回合 batch 中
4 个 retreat grasp-loss（episode 2、5、6、7）正是这一类抓取几何不一致的后果；不应先修改 retreat 路线。

### 27.3 核心解决方式：旋转已有成功标定，而不是追踪 jaw 端点

将既有 `20 mm` 标定解释为 gripper local closing-axis offset，并在 alignment 完成后用实测 gripper
姿态转到世界系：

```text
closing_axis_xy = normalize(world_direction(gripper_local_+X).xy)
target_gripper_xy = live_cube_xy - 0.020 * closing_axis_xy
```

这里使用 alignment 后的实时 cube XY，保留原抓取 Z；`jaw_detection` 不参与 recenter。零旋转时该公式
严格退化为原来的 `(-0.020, 0)` 成功目标，因此只修正姿态变化带来的坐标系错误。

### 27.4 验证结果

- Windows 原生 seed-42 单环境 smoke：`1793` 步，`expert_success=True`，无 timeout/abort；
- 相同 seed-42 随机序列的 10 回合录像 batch：从旧版 `6/10` 提升到 `10/10`；
- 原失败 episode 2、5、6、7 全部成功；
- `failed_episodes=[]`、`servo_abort_episodes=[]`、`success_rate=1.000`；
- batch 与全部 10 个录像目录位于
  `D:\Sim\results\expert-batch\polar\rotated-grasp-offset-20260813-203016`。

核心教训：先确认一个 frame 是 IK 原点、刚体原点、接触点、检测端点还是夹持中心。经过姿态对齐后，
局部标定向量必须随末端姿态旋转；不能继续把它当作世界系固定 XY，也不能用单个可动爪端点替代夹持中心。

## 28. 服务器 50 回合把 jaw 端点阈值误当成抓取丢失（2026-08-14）

服务器 scene audit、非黑 polar smoke 均通过；相同进程的 50 回合 batch 为 `44/50`。六个失败全部在
`retreat_to_safe`，旧原因都是 `jaw_cube_distance > 0.025 m`。其中 episode 33 仅为 `0.025043 m`，episode 43
仅为 `0.025007 m`，分别只越界 `0.043 mm` 和 `0.007 mm`。batch 又在 abort 瞬间结束回合，所以较高的最终
Z/速度只表示正在上抬，不能证明 cube 已经从夹爪脱落。

更根本的问题是：该 jaw frame 位于可动爪远端，会随夹爪闭合、接触受力和腕部姿态改变；单帧世界距离既没有
滞回，也不是抓取不变量。修复后在 retreat 入口锁存 cube 在 gripper 局部坐标系中的位置：

```text
p_cube_in_gripper = R_gripper_world^-1 * (p_cube_world - p_gripper_world)
relative_position_error = norm(p_cube_in_gripper_live - p_cube_in_gripper_at_retreat_entry)
```

初版曾仅用相对位置漂移超过 `12 mm` 连续 5 帧确认丢失。Windows seed-42 动态 smoke 给出反证：局部漂移
达到 `15.18 mm` 且 `pick_cube` 瞬时为假时，jaw 距离仍只有 `20.56 mm`；新门过早中止了旧版能够继续的已知
成功轨迹。因此最终采用双证据：相对位置漂移超过 `12 mm` 且 jaw 距离超过 `25 mm`，连续 5 帧才确认丢失；
相对漂移回落到 `8 mm` 内或 jaw 距离回落到 `22 mm` 内即清零 streak，形成滞回。未确认抓取时的入口保护仍
保留。真正确认丢失后仍立即安全终止，不会继续运输。

最终双证据版本需重新通过 polar 静态回归和 Windows seed-42 smoke；是否把服务器 `44/50` 提高到目标门槛，
仍需用同一 seed 的 50 回合 batch 动态验证，静态检查不能替代动力学结论。
