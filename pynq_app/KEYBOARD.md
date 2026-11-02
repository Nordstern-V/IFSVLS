# USB 键盘验证与类别命名（2026-11-01 / r21，A/B r17 + 键盘 r20）

USB Host 启动修复已通过用户实板验证：USB3320、ChipIdea Host、板载 HUB 和 CX 2.4G Wireless Receiver 已枚举。本功能直接读取 Linux evdev，使用 Python 标准库，不需要额外安装 evdev/pynput，也不依赖桌面窗口或串口标准输入。

## 先验证键盘

将 `usb_keyboard.py` 与 `keyboard_validate.py` 一起放到板端 `pynq_app`，执行：

```bash
cd /home/xilinx/jupyter_notebooks/PEFSL_MIX_PLUS/pynq_app
sudo -E /usr/local/share/pynq-venv/bin/python3 keyboard_validate.py --seconds 60
```

默认按键能力自动选择包含 A-Z/Enter 的键盘接口，避开复合无线接收器的鼠标、系统控制和多媒体接口。不要假定 event 编号固定。如需指定已确认的键盘接口：

```bash
sudo -E /usr/local/share/pynq-venv/bin/python3 keyboard_validate.py --device /dev/input/event0 --seconds 60
```

依次输入 `Apple 01_test`、Backspace、`t`、Enter，并测试 Shift/Caps Lock、主键盘数字和开启 Num Lock 后的小键盘数字。下划线通过 `Shift` 加 `-` 输入，空格直接按空格键。预期显示：

```text
[keyboard] connected device=... name=CX 2.4G Wireless Receiver
raw: KEY_A down
raw: KEY_A up
edit: text text='a'
...
raw: KEY_ENTER down
raw: KEY_ENTER up
confirmed: 'Apple 01_test'
```

`--seconds 0` 持续运行；串口 Ctrl-C 或物理键盘 Ctrl-C 都可退出。默认独占选中的键盘，退出后自动释放；验证脚本与 bbox 应用应分开运行。`--no-grab` 仅用于诊断，届时字母还会进入 Linux 控制台。

## 运行命名应用

首次部署键盘时更新四个文件：`usb_keyboard.py`、`keyboard_validate.py`、`pefsl_touch_ui.py`、`pynq_bbox_demo.py`，保留当前数字/运算符/计算器的其他依赖文件。若板端已是 r20，本次 A/B 恢复只更新 `pefsl_touch_ui.py` 与 `pynq_bbox_demo.py`，具体见 `AB_R17_RESTORE.md`。板端现有的学习 NPZ 和数字/运算符缓存可继续使用。

```bash
cd /home/xilinx/jupyter_notebooks/PEFSL_MIX_PLUS/pynq_app
sudo -E /usr/local/share/pynq-venv/bin/python3 pynq_bbox_demo.py
```

启动 UI 版本为 `mix-plus-ab-r17-20261101-r21-keyboard-calc10`，键盘连接成功打印 `[keyboard] connected ...`。

1. A 模式拖动框选物体，松手后开始键盘输入。
2. 输入名称，字符逐个显示在框上方；输入期间末尾额外的下划线是编辑光标，Enter 后消失。允许 A-Z/a-z、数字 0-9、空格和下划线，最多 24 个字符（包含空格）。支持 Shift、Caps Lock 和 Backspace；下划线通过 `Shift+-` 输入，主键盘数字直接输入，小键盘数字随 Num Lock 状态，Shift 临时反转小键盘数字/导航含义。其他标点及 Shift+主键盘数字形成的符号不录入。按住键的自动重复不会重复录入。
3. Enter 确认名称，输入结束；首尾空格自动去除，名称内部空格保留。全空格名称按空名称处理，沿用已有名称或默认类别 ID。随后触摸类别 1/2/3/4，启动现有八帧学习采集。输入过名称却未按 Enter 时，类别按钮会提示先确认。
4. 采集成功后，类别 ID 与显示名称关联，例如 `class1 -> apple`。Enter 本身不触发拍摄；名称与特征在采集成功时一起提交。应在命名、采样期间保持框内物体稳定。
5. 切换 B 模式，手动识别和自动扫描均使用名称替换原 `class1`。r21 恢复旧版 A/B 计算流程和百分比显示：手动为 NCM 类间分数，扫描为旧版余弦分数。
6. SAVE 将名称与学习特征一起写入 `pefsl_learned.npz`；下次 LOAD 恢复。旧文件没有名称字段时继续显示原类别 ID。

Esc 取消当前名称，可重新输入或直接选择类别。重新画框、切换 A/B/D、CLEAR、LOAD 会取消未提交命名。给已有类别补样本时，不输入新名会保留原名称；显式输入新名称并完成采集时更新该类别的显示名，但已有特征仍属于这个类别，不能借改名把旧物体样本当作新物体样本。

键盘断开或输入事件丢失时，当前尚未采集的命名框会取消，重连后重新画框即可。已确认并开始采集的名称继续跟随本次采样；失败不改旧类名。B/D 模式不会将普通字母当作命名输入；D 模式的数字/运算符及计算器保持各自类别。

主入口支持 `--keyboard-device /dev/input/eventX` 指定设备、`--no-keyboard` 关闭键盘读取，以及诊断用的 `--keyboard-no-grab`。无键盘时仍可按原流程使用触摸类别按钮。

## 实现与验证边界

键盘线程只读 evdev 并排队；UI 主线程按单调时钟时间戳合并触摸和键盘事件。触摸 IIC 子进程启动顺序保留，重启子进程会关闭继承的键盘 fd 副本，以免保留额外键盘引用。

主机测试涵盖复合设备筛选、按下/释放、大小写、输入丢失、拔插重连、退出释放、事件顺序、名称持久化、学习成功/失败及原 B/D/计算器回归。USB Host、r19 大小写字母及 A/B 命名已由用户实板确认；r20 的空格、数字和下划线仍需板端按上述步骤验收。

## r20 字符扩展

r19 的大小写字母及 A/B 命名流程已由用户实板确认。r20 增加空格、主键盘数字、小键盘数字和下划线，并将相同字符规则接入实时输入、学习提交和 NPZ 保存/加载。若板端已经是 r19，本次只需覆盖 `usb_keyboard.py`、`keyboard_validate.py`、`pefsl_touch_ui.py`；主入口不需要更新。可用 `Apple 01_test` 完成 A 采样、SAVE/LOAD 和 B 手动/自动显示测试。

## r21 A/B 恢复

按照用户指定的 `PEFSL_MIX_PLUS_digital_10.31` 恢复 A/B，键盘字符与名称保存机制继续使用 r20。不要从旧目录覆盖当前键盘或计算器代码。部署和旧版对照验证见 `AB_R17_RESTORE.md`。
