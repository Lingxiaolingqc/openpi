# Windows 本地控制 SO-101 关节角度

本文说明如何在 Windows 上使用
`examples/so101/hold_joint_angles_windows.py`，控制连接在当前电脑上的 SO-101 机械臂移动到六个指定关节角度，并保持该姿态。

## 功能与限制

脚本会：

- 复用现有 `leader_arm.json` 标定文件。
- 接收六个关节角度。
- 先读取机械臂当前位置，再显示计划移动量。
- 经过两次人工确认后才启用扭矩。
- 以受限速度逐步移动到目标位置。
- 依靠舵机闭环位置控制保持目标姿态。
- 持续监控舵机温度和位置跟踪误差。
- 在 `Ctrl+C`、异常、过热或持续误差过大时自动卸力。

脚本不会修改：

- 电机 ID。
- 波特率。
- PID 参数。
- Homing Offset。
- 已有标定文件。

六个输入值的顺序固定为：

```text
shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_roll gripper
```

这里的角度是经过标定的关节角度。`0°`表示相应关节已记录活动范围的中点，不是舵机原始编码零点，也不表示保持启动时的位置。

## 安全要求

正式启用扭矩前：

1. 停止所有正在使用 `COM7` 的发布、标定或控制程序。
2. 打开舵机所需的外部电源。
3. 托住机械臂，防止上力瞬间下坠。
4. 清空机械臂的运动范围。
5. 手指远离夹爪、关节和连杆夹缝。
6. 检查 USB 线和电源线不会被机械臂拉扯。
7. 第一次测试不要使用接近标定极限的目标角度。

如果机械臂猛烈运动、堵转、持续异响或程序无法卸力，应立即切断舵机电源。

## 1. 打开新终端

打开一个新的 Windows PowerShell，不需要激活 Conda，也不需要连接服务器。

进入 OpenPI 仓库并记录专用 Python 路径：

```powershell
Set-Location 'D:\Documents\Xprogram\HuiXIONG\EmbodiedAI\openpi'

$python = '.\tmp\leisaac-remote-env\python.exe'
```

确认解释器和控制脚本存在：

```powershell
& $python --version

Test-Path $python
Test-Path '.\examples\so101\hold_joint_angles_windows.py'
```

两个 `Test-Path` 都应输出：

```text
True
```

## 2. 释放 COM7

如果 Leader 数据发布器仍在运行，先回到它的终端并按：

```text
Ctrl+C
```

可以在新终端中检查相关进程：

```powershell
Get-CimInstance Win32_Process |
  Where-Object {
    $_.CommandLine -match 'so101_joint_state_server|leader_remote_windows'
  } |
  Select-Object ProcessId, CommandLine
```

如果仍有发布器进程，应回到对应终端正常停止它。不要让两个程序同时打开 `COM7`。

SSH 隧道本身不占用 `COM7`，可以继续运行，但本地角度控制不需要 SSH、tmux、服务器或 Isaac Sim。

## 3. 检查本地运行环境

运行只读审计：

```powershell
& $python `
  '.\examples\so101\leader_remote_windows.py' `
  audit `
  --port COM7 `
  --leisaac-root '.\tmp\leisaac-v0.4.0'
```

重点确认输出包含：

```text
requested_port_detected: True
module_serial: available
module_scservo_sdk: available
calibration_file_exists: True
calibration_json_valid: True
```

如果 `COM7` 未检测到，应先检查 USB 连接和 Windows 设备管理器。如果标定文件不存在，不要进入执行模式。

## 4. 预检目标角度

先运行默认的安全预检模式：

```powershell
& $python `
  '.\examples\so101\hold_joint_angles_windows.py' `
  --port COM7
```

看到提示后，输入六个目标角度。例如：

```text
0 -20 35 0 0 10
```

预检模式会：

- 读取并检查标定文件。
- 计算每个关节的安全角度范围。
- 检查六个目标值是否为有限数字并位于安全范围内。

预检模式不会打开 `COM7`、启用扭矩或移动机械臂。成功时最后会显示：

```text
DRY_RUN_OK: no serial port was opened and torque was not enabled.
```

也可以直接通过命令行提供六个角度：

```powershell
& $python `
  '.\examples\so101\hold_joint_angles_windows.py' `
  --port COM7 `
  --angles 0 -20 35 0 0 10
```

## 5. 正式连接并控制机械臂

完成安全检查后运行：

```powershell
& $python `
  '.\examples\so101\hold_joint_angles_windows.py' `
  --port COM7 `
  --execute
```

再次按固定顺序输入六个目标角度。

### 第一次确认

程序显示以下提示时：

```text
Type MOVE_AND_HOLD to continue:
```

检查机械臂周围安全后，准确输入：

```text
MOVE_AND_HOLD
```

程序随后连接 `COM7`，但暂时不会启用扭矩。它会输出：

```text
current_deg:
requested_move_deg:
max_requested_move:
```

含义分别是：

- `current_deg`：当前读取到的六个关节角度。
- `requested_move_deg`：每个关节从当前位置到目标位置需要移动的角度。
- `max_requested_move`：本次运动中幅度最大的关节。

如果数值或方向不符合预期，不要进行第二次确认。输入任意其他内容即可取消，程序会关闭串口且不启用扭矩。

### 第二次确认

只有确认当前位置和计划位移均合理后，才在以下提示中输入：

```text
APPLY_TARGET
```

完整提示为：

```text
Type APPLY_TARGET to enable torque and move:
```

此后程序才会：

1. 将当前位置设置为初始目标，减少上力瞬间的跳变。
2. 启用六个舵机的扭矩。
3. 以默认不超过 `15°/s` 的速度逐步移动。
4. 到达目标后持续保持并监控状态。

## 6. 保持期间的输出

保持姿态时，程序会定期输出：

```text
present_deg:
hottest_motor:
max_tracking_error:
```

- `present_deg` 是当前六个关节位置。
- `hottest_motor` 是当前温度最高的电机。
- `max_tracking_error` 是目标位置与实际位置之间最大的角度误差。

默认保护条件：

- 温度达到 `65°C` 时卸力。
- 位置误差持续超过 `15°` 大约两秒时卸力。
- 默认目标移动速度上限为 `15°/s`。

## 7. 正常停止和卸力

按：

```text
Ctrl+C
```

正常情况下会看到：

```text
STOP_REQUESTED: disabling torque
TORQUE_DISABLED_AND_PORT_CLOSED
```

确认机械臂已经卸力后再接触或手动移动它。

## 可选参数

### 在命令中直接指定角度

```powershell
& $python `
  '.\examples\so101\hold_joint_angles_windows.py' `
  --port COM7 `
  --angles 0 -20 35 0 0 10 `
  --execute
```

### 降低运动速度

例如将速度降低到 `8°/s`：

```powershell
& $python `
  '.\examples\so101\hold_joint_angles_windows.py' `
  --port COM7 `
  --speed-deg-s 8 `
  --execute
```

脚本允许的速度范围是大于 `0` 且不超过 `30°/s`。第一次测试建议使用默认速度或更低速度。

### 查看所有参数

```powershell
& $python `
  '.\examples\so101\hold_joint_angles_windows.py' `
  --help
```

## 常见问题

### COM7 打不开

常见原因：

- Leader 发布器仍在运行。
- 标定程序仍在运行。
- 其他串口工具占用了 `COM7`。
- USB 线断开或设备重新枚举成了其他端口。

先关闭占用串口的程序，然后在设备管理器中重新确认端口号。

### 所有电机 ID 都报告缺失

检查：

- 舵机外部电源是否开启。
- USB 串口适配器是否连接正常。
- 三线总线的方向和极性是否正确。
- 电机 ID 是否确实为 `1` 到 `6`。
- 波特率是否为 `1,000,000`。

### 目标角度被拒绝

脚本会根据现有标定文件计算范围，并默认在两侧各保留 `2°` 安全余量。目标超出范围时不会打开串口或启用扭矩。请降低对应目标角度，不要通过修改标定文件绕过保护。

### 出现持续跟踪误差

可能原因包括：

- 机械臂发生碰撞或被阻挡。
- 负载过重。
- 舵机供电不足。
- 关节装配或线缆限制运动。

程序会自动卸力。排除机械和供电问题后再重新测试。

### 自动卸力失败

如果终端出现：

```text
CRITICAL: automatic torque disable failed
```

应立即切断舵机电源，不要依赖串口继续发送命令。
