# MIX_PLUS PYNQ bbox application

## 当前：r21 恢复指定 r17 A/B，保留计算器和键盘（2026-11-01）

版本为 `mix-plus-ab-r17-20261101-r21-keyboard-calc10`。A/B 已按照用户指定的
`PEFSL_MIX_PLUS_digital_10.31/pynq_app` 恢复：A 原32×32采样，B 手动 NCM 分类，
自动扫描回到 r17 的窗口轮换、余弦评分及周期确认。C124 身份门和结果过期策略
不再参与 A/B。当前 D 数字/算符识别、10数字/9算符计算器和 r20 键盘完整保留。

已在 r20 板端部署的用户只需更新 `pefsl_touch_ui.py`、`pynq_bbox_demo.py`；
保留当前数字/运算符/键盘模块和学习文件。启动命令不变。旧学习文件及名称可继续
LOAD，无需重建模型或数字缓存。部署和验收见 [A/B r17 恢复说明](AB_R17_RESTORE.md)。

123 项当前回归通过，其中10项直接对照指定旧 r17；数字/计算器/键盘方法已做
AST 保留比对。实际学习效果仍需同场景板端验证。以下为历史实施记录。


## 计算器扩容至 10 个数字、9 个算符（2026-10-31）

历史版本 `mix-plus-bmode-20261031-c124-r18-calc10` 将 D 计算器容量扩为
X1～X10、OP1～OP9。两版预览布局均支持完整表达式和结果显示；容量用统一常量管理，
达到上限后拒绝继续存入，UNDO 后可补回。每个 X 仍保存一个 0～9 数字。
本轮只修改 `digit_calculator.py` 和 `pefsl_touch_ui.py`，无需重新建库。
已同步当前 r18 的板端覆盖这两个文件即可；较早版本须先同步 r18 依赖，
具体检查、传输和验收命令见 [10 数字扩容部署说明](../plan/calculator_10_board.md)。

## 计算器计划第 4～5 步（2026-10-31）

历史版本 `mix-plus-scan-20261030-r17-calculator-s45` 已实现 D 模式表达式存储和四则运算，
提供 `--digit-preview on`（保留二值化预览）和 `off`（隐藏预览）两种布局。
新增 `digit_calculator.py`，更新 `pefsl_touch_ui.py`；同时修复 `digit_recognizer.py`
的缓存 evaluate 在 ARM 32 位平台上 `np.repeat` 拒绝 int64 counts 的问题。无需重建缓存。

- D 默认识别数字，`OP MODE` 亮起时只识别四个算符；存入不会自动切换类型。
- FREEZE 后画框，连续 3 次有效预测稳定且原始/平滑结果一致，显示 READY 后核对并按 Xn 或 OPn。
  只存入屏幕已显示且仍有效的候选；重复触摸不重复追加。
- 最多 5 个数字、4 个算符，先乘除后加减，同级从左到右；OUT 显示结果，DIV/0 保留表达式。
  每个 X 仅一位数字，小数结果最多显示 6 位，近似值带 `~`。
- UNDO 撤销最后一项；RESET 清空计算表达式。两者保留冻结图和类型，但需重新画框再存入。
  LIVE/FREEZE、A/B/D 切换保留已确认表达式；重启程序后清空。
- 原 SAVE/LOAD/CLEAR 仍管理 A/B 学习数据；CLEAR 不代替计算器 RESET。
- D 可画框区域：预览 on 为 x0..639、y58..363；off 为 x0..799、y58..363。
  y364..421 显示表达式/结果/候选，y422..479 为五个计算按钮；A/B 区域未缩小。

PC 软件验证使用替代 TCU，板端计算交互仍需验收。准确的传输清单、备份、两版启动命令和测试流程见
[第 4～5 步板端部署说明](../plan/calculator_steps45_board.md)。以下为前期实现记录。

## 算符计划第 1～3 步（2026-10-30）

版本 `mix-plus-scan-20261030-r17-operator-s3` 已接入独立四类算符识别。
D 模式默认数字；顶部 `OP MODE` 点亮后只识别 plus/sub/mul/div，再按回数字。
两者共用串行 TCU，分别使用数字/算符数据、均值、分类器及平滑历史。
冻结/解冻保持类型；切换类型保留冻结图与完成的 ROI，并丢弃旧候选。

算符支持集在 `operator_support/`，四类各 5 张。新参数 `--class-set operators`
用于建库；默认生成 `operator_support_r17.npz`。`inspect`、`evaluate` 可检查缓存
及进行自匹配/留一验证，无须操作 TCU。UI 新增 `--operator-cache`、
`--operator-min-score`、`--operator-min-margin`、`--operator-history` 和
`--digit-preview on|off`。算符默认门限 0.45/0.08 是试验值，需实拍标定。

该历史版本尚无 X/OP 存储和 OUT 计算按钮。其预览 on 时有效 ROI 区域为
x0..639、y58..421；off 时为 x0..799、y58..421，A/B 区域沿用原配置。
完整文件传输表、备份及串口命令见 [板端步骤](../plan/operator_steps123_board.md)。

This is plan step 5. It keeps one Overlay instance alive for the camera/LCD video hierarchy, touch IIC, and TCU DMA. The first UI deliberately supports one ROI at a time:

- A: draw an ROI, then touch class 1..4 in the bottom bar to collect eight TCU features.
- B: tap SCAN to search the image automatically, or draw an ROI for manual
  recognition. Automatic scan uses bounded multi-scale windows, cosine
  matching, and overlap suppression. HSV statistics are retained in the
  learned file but the hard color gate is disabled by default because camera
  exposure and background pixels make it reject valid windows. The same manual ROI
  remains active and is reclassified periodically, so changing the object
  updates the result without drawing another box.
- D: draw one ROI containing a single handwritten digit. D mode uses the
  prebuilt `digit_support_r17.npz` 80D support cache and the same TCU feature
  extractor as A/B, while preserving the existing A/B learning and recognition
  data file. Its top-right `FREEZE` button holds the current LCD image for ROI
  drawing and recognition; the same button becomes `LIVE` to resume the camera.
- SAVE/LOAD/CLEAR: manage the learned feature file. SAVE writes the current
  in-memory classes to `pefsl_learned.npz`; LOAD restores that file; CLEAR
  clears the in-memory classes and can be followed by LOAD to restore them.
- SCAN: toggle automatic scanning. B mode starts it automatically when learned
  data is available; drawing a manual B ROI pauses it, and tapping SCAN again
  resumes automatic scanning.

Copy pynq_app/ together with the already deployed artifacts/, touch_deploy/, and tcu_deploy/ directories. On PYNQ run:

    cd /home/xilinx/jupyter_notebooks/PEFSL_MIX_PLUS/pynq_app
    sudo -E /usr/local/share/pynq-venv/bin/python3 pynq_bbox_demo.py \
        --touch-poll-ms 100 --predict-ms 1000 --scan-ms 100

For a bounded smoke run use --seconds 30. The application expects LCD mode 800x480 and LCD ID 0x7084. It modifies the frame returned by video_in.readframe() in place, returns ownership with video_out.writeframe(frame), and only calls freebuffer() when ownership was not transferred.

`--predict-ms` controls the background B-mode refresh interval. The TCU call runs
in its own worker, so changing the object inside the active ROI does not block
the camera or touch loop. `--touch-poll-ms 100` is the normal setting; the
driver recovers a failed AXI-IIC transaction in the same poll cycle.
`--scan-ms` is the delay between automatic scan windows and `--scan-windows`
limits the number of windows in one cycle. Because each window consumes one
serial TCU call on the PYNQ DMA path, the scan worker intentionally processes
one window at a time while the video and touch loops remain responsive. Tap
SCAN to toggle the locator; drawing a manual B ROI temporarily stops scanning.
`--scan-hist-th` can re-enable a histogram gate for a controlled camera setup;
the default value is 0, which leaves cosine matching as the decision gate.

Learning samples use the same background TCU worker as B-mode recognition.
The video and touch loops therefore continue to run while the eight samples
for a class are collected.

## Fix for the first-frame/slow-touch symptom

The current entry point polls touch IIC in a worker thread, uses an interruptible VDMA wait, discards one warm-up frame, and starts the vendor video channels with video_in.tie(video_out). Replace the whole pynq_app/ directory and the updated touch_deploy/touch_driver.py on the board before rerunning. A healthy run should report frames increasing continuously every few seconds; touching A/B or class buttons should change the status line without stopping the camera. Ctrl-C should exit without repeated key presses.

The UI assumes the camera buffer is in the BGR channel order used by the desktop preprocessing. The screen should be observed for live motion before collecting samples.



The bbox app gates touch IIC polling on the FT INT input (idle high, touch low) to avoid idle receive failures. FT status and first-point coordinates are read in one transaction, and the worker defaults to a 100 ms poll period. A failed PYNQ AXI-IIC transaction is recovered with the AXI-IIC SOFTR register and retried once; this keeps the normal poll period and avoids the long-lived busy state seen during long drags. It prints [touch-int] transitions and [touch] events. A stable INT-high interval also closes the local contact state with an `up` event, so the next tap starts a fresh `down` even when the controller emits INT pulses and no invalid status read is available. Use `--touch-poll-ms 150` for a slower diagnostic rate or `--no-touch-int-gate` only if the panel does not pull INT low.


## r13 IIC 卡顿诊断

r13 必须同时替换 `pynq_app/pynq_bbox_demo.py` 和 `touch_deploy/touch_driver.py`。发生总线异常时终端会立即打印 `[touch-iic] error #...`，随后打印 `recovery=ok` 或 `recovery=unavailable`；恢复后打印 `[touch-iic] recovered after: ...`。五秒心跳会保留 `touch_iic=... errors=... recoveries=...`，不再用 `touch_error=None` 表示健康状态。

## r14 IIC 卡顿看门狗

r14 在触摸轮询线程外增加 AXI-IIC 看门狗。一次真实 FT 读事务超过约 0.8 秒时，另一个线程写 AXI-IIC SOFTR（0x40=0x0A）释放内部忙状态，并打印 `[touch-iic] watchdog recovery=ok`；每次事务最多触发一次。正常触摸轮询周期不变。心跳会附带 `watchdogs=` 计数，便于判断是否发生过卡死恢复。

r14 必须同时替换 `pynq_app/pynq_bbox_demo.py` 与 `touch_deploy/touch_driver.py`。驱动版本应显示 `mix-plus-touch-20261023-r14`。

r14 同时限制 `move` 串口日志为每 250 ms 一条；所有 `down/move/up` 事件仍完整进入 UI 事件队列。这样长时间拖框时不会因为串口 `flush` 反向拖慢 IIC 事务。

## r15 触控芯片硬恢复

r14 日志中 `watchdog recovery=ok` 后仍保持 `state=stalled`，说明只写 AXI-IIC SOFTR 没有释放 FT 控制器。r15 在看门狗触发时先拉低/释放 `touch_rst_gpio`，再写 SOFTR；事务返回后调用 `initialize(reset=False)` 重新检测 FT 并写回配置寄存器。恢复期间清空旧触摸事件和 INT 状态，避免旧 `down/move` 事件进入 UI。驱动版本为 `mix-plus-touch-20261024-r15`。

## r16 进程隔离触摸 IIC

r15 日志显示三次 `device_reset=ok` 后仍保持 `stalled`，说明旧 PYNQ C IIC 调用无法被线程复位打断。r16 在 Linux PYNQ 环境中使用 `fork` 启动独立触摸进程；触摸进程独占 `TouchController.poll()`，父进程通过管道接收事件和心跳。超过 0.8 秒无心跳时，父进程终止旧触摸进程，执行触控 GPIO 复位与 AXI-IIC SOFTR，再 fork 新进程并执行 `initialize(reset=False)`。进程启动放在 UI predictor 线程和摄像头初始化之前，避免 fork 时继承应用线程。

r16 必须同时替换 `pynq_app/pynq_bbox_demo.py` 和 `touch_deploy/touch_driver.py`。驱动版本为 `mix-plus-touch-20261024-r16`。正常卡顿恢复日志应出现 `process watchdog timeout`、`child restart reason=heartbeat-timeout` 和新的 `[touch-iic] child ready`。

## r17 C104 扫描误检抑制（2026-10-24）

自动扫描已按 `PEFSL_C104` 的轻量门控链更新。扫描分数采用 `0.7*Top1 + 0.3*Top5`，多类别分数差小于 0.04 时拒绝；默认余弦门限提高到 0.56。候选窗口还会检查中心与外环 HSV 直方图距离（默认 0.14）并拒绝超过注册框 2.3 倍的巨型窗口。每个扫描周期按类别只保留最高分框，并要求同一类别在相近位置连续两个周期确认后才显示。

这些检查只使用现有 80D 特征、图像和 `scan_sizes`，因此 r11/r12 的 `pefsl_learned.npz` 可以直接加载。板端更新时需同时替换本目录的 `pefsl_scan.py`、`pefsl_touch_ui.py` 和 `pynq_bbox_demo.py`；已稳定的 r16 `touch_driver.py` 不需要改动。若物体较远或中心与背景颜色非常接近，可用 `--scan-cos-th`、`--scan-confirm-cycles` 和 `--scan-contrast-th` 调整召回与误检的平衡。

## 数字支持集 80D 原型缓存（计划第三步）

数字支持图片位于 `digit_support/0` 至 `digit_support/9`。建库程序
`digit_recognizer.py` 只读取数字目录 `0` 至 `9` 中的两种已知命名：
`<digit>_<index>.png` 和更新 support 使用的 `debug_binary_*.png`，并排除
运算符目录；它使用当前 r17 bitstream/tmodel 的
`TcuFeatureExtractor`，不会读取或修改 `pefsl_learned.npz`。

先用一个样本/类确认 TCU 建库路径，再生成完整缓存：

    cd /home/xilinx/jupyter_notebooks/PEFSL_MIX_PLUS/pynq_app
    sudo -E /usr/local/share/pynq-venv/bin/python3 digit_recognizer.py build --limit-per-class 1
    sudo -E /usr/local/share/pynq-venv/bin/python3 digit_recognizer.py build

默认输出为 `pynq_app/digit_support_r17.npz`。建库过程不需要摄像头、LCD
或触摸屏，但需要现有 bitstream、`.hwh`、tmodel、`.tprog` 和 `.tdata`。

## D 模式数字识别（第四步已实现）

启动 `pynq_bbox_demo.py` 后点击顶部 `D`，在 LCD 上框选一个数字并松开触摸。
程序会保留 ROI 的宽高比，使用与支持集相同的二值化/32x32 预处理，再调用
当前 r17 TCU 得到 80D 特征。连续五帧概率默认做均值平滑；当最高类别概率低于
`--digit-min-score` 或最高、次高类别的差值低于 `--digit-min-margin` 时，状态栏显示
`?`，避免把不确定结果当成数字。D 模式不读取或修改 `pefsl_learned.npz`，A/B
学习与识别流程保持不变。

D 模式顶部右侧按钮用于控制画面：实时状态显示 `FREEZE`，点击后保存当前摄像头
画面并显示 `LIVE`；冻结后可以继续在 LCD 上画框，D 识别仍使用这张保存的画面。
程序仍持续读取和释放摄像头帧，避免视频 DMA 停止。再次点击 `LIVE` 恢复实时画面，
切换到 A 或 B 模式也会自动恢复实时输出。

默认缓存路径是 `pynq_app/digit_support_r17.npz`。如果缓存放在其他位置，可以在板端
启动时指定：

    sudo -E /usr/local/share/pynq-venv/bin/python3 pynq_bbox_demo.py \
        --digit-cache /home/xilinx/jupyter_notebooks/PEFSL_MIX_PLUS/pynq_app/digit_support_r17.npz \
        --digit-min-score 0.25 --digit-min-margin 0.04 --digit-history 5

其中 `--digit-min-score` 的范围为 0 到 1，`--digit-min-margin` 不小于 0，
`--digit-history` 至少为 1。缺少缓存时 A/B 仍可启动，点击 D 会提示缓存不可用；
建库完成后重新启动程序即可加载缓存。

## 更新 PC support 到板端

更新后的 PC support 只把 `0` 至 `9` 目录复制到板端；`plus`、`sub`、`mul`、
`div` 不属于第一版 D 模式。`2` 至 `9` 中的 `debug_binary_*.png` 是有效的
32x32 支持图片，不能删除或改成调试文件。上传前应替换整个
`digit_support/0..9`，不要把旧的正式图片和新图片混在同一类别中。

板端先备份旧目录和缓存：

    cd /home/xilinx/jupyter_notebooks/PEFSL_MIX_PLUS/pynq_app
    stamp=$(date +%Y%m%d_%H%M%S)
    mkdir -p backup_digit/$stamp
    cp -a digit_support backup_digit/$stamp/
    cp -a digit_support_r17.npz backup_digit/$stamp/ 2>/dev/null || true

然后上传本项目最新的 `pynq_app/digit_support`、`digit_recognizer.py`、
`digit_preprocess.py` 和 `pefsl_touch_ui.py`。先做单样本建库：

    sudo -E /usr/local/share/pynq-venv/bin/python3 digit_recognizer.py build \
        --limit-per-class 1 --support-dir digit_support \
        --output digit_support_r17_smoke.npz

单样本建库成功后生成完整缓存，并覆盖旧缓存：

    sudo -E /usr/local/share/pynq-venv/bin/python3 digit_recognizer.py build \
        --support-dir digit_support --output digit_support_r17.npz

预期结果为 `10 classes / 49 images / 80D`，类别计数为
`[5,5,5,5,5,4,5,5,5,5]`。新版缓存 schema 会拒绝旧缓存，避免忘记重建时
继续使用旧的 51 张支持集。建库后必须重启 `pynq_bbox_demo.py`，程序启动时才会
加载新缓存。

## r18：C124 B 模式识别优化（2026-10-31）

本版保留 D 模式、运算符和计算器，普通 B 手动 ROI 与扫描使用同一拒识链。
新增 `pefsl_bmode.py`、`pefsl_identity.py`、`pefsl_proposals.py`；部署时与
`pefsl_touch_ui.py`、`pynq_bbox_demo.py` 一起更新。无需重编 bitstream 或数字缓存。

完整用法、旧库兼容说明、参数、实板验收项目及限制见 [B_MODE_C124.md](B_MODE_C124.md)。
新 A 学习保存完整 ROI 小图，B 会核验整个区域内的分布式纹理与多视图颜色；
旧文件仍可 LOAD，但要获得新的空间身份门，需在新版 A 重新学习并 SAVE。
B 显示 `sim=` 相似度，未知内容显示 `no match`。自动扫描增加区域候选和有截止
时间的局部回访；结果过期或直接核验失败会撤销旧标签。触摸恢复路径保持 r16。

## r19 USB 键盘命名（2026-11-01）

USB Host/PHY 修复已实板通过。现加入标准库 evdev 键盘读取、独立 keyboard_validate.py 验证器，以及 A 学习类别名称。命名顺序为 A 画框 → 输入 A-Z/a-z（框上方实时显示）→ Enter → 触摸类别1..4 → 采集成功。B 手动识别与自动扫描显示该名称，内部类别 ID 不变；SAVE/LOAD 同时保存名称，旧学习文件兼容。新版本为 mix-plus-bmode-20261101-c124-r19-keyboard-calc10。

部署与验收命令、大小写/退格/热插拔规则见 KEYBOARD.md。验证器只需 usb_keyboard.py 和 keyboard_validate.py，使用 sudo -E /usr/local/share/pynq-venv/bin/python3 keyboard_validate.py --seconds 60；先退出验证器再运行 bbox。应用更新需要 usb_keyboard.py、pefsl_touch_ui.py、pynq_bbox_demo.py，并保留当前 C124/D/计算器依赖。主机共112项测试通过，新增键盘命名尚需板端验收。

## r20 命名字符扩展（2026-11-01）

r19 的大小写及 A/B 命名已由用户实板确认。r20 加入空格、主键盘0-9、小键盘数字和Shift+-下划线；名称验证、实时显示、学习提交与NPZ保存/加载使用同一字符规则。Enter去除首尾空格、保留内部空格，全空格名称沿用原名称或默认ID。24字符上限包含空格。版本为mix-plus-bmode-20261101-c124-r20-keyboard-calc10，完整124项主机测试通过。r19板端本次只需覆盖usb_keyboard.py、keyboard_validate.py、pefsl_touch_ui.py，重启应用；以Apple 01_test验证新增字符。详见KEYBOARD.md。
