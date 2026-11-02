# PEFSL_MIX_PLUS 模块连接与流程图

整理日期：2026-11-02；对应当前版本：`mix-plus-ab-r17-20261101-r21-keyboard-calc10`。

依据 [model.md](model.md)、[conclusion.md](conclusion.md) 与当前活动源码绘制，覆盖模块 1～32。

节点中的编号与 model.md 一致；实线表示图像、特征或输入事件流，虚线表示配置、管理、恢复或旁路验证。图是功能连接示意，组合节点不代表每个功能都在同一个线程执行。

PS 负责交互、预处理、NCM、扫描筛选及算术；PL 负责视频硬件通路、TCU 和触摸接口。历史 C124 身份门、区域候选、局部回访和结果过期策略没有画入当前流程。

## 一、系统总体连接

摄像头提供图像，触摸和键盘提供操作事件；UI 产生学习或识别任务，统一调用 TCU，处理结果后显示到 LCD。底部支撑模块在图 4 展开。

![一、系统总体连接](flowcharts/01_system_overview.svg)

[SVG 矢量图](flowcharts/01_system_overview.svg) · [Mermaid 源文件](flowcharts/01_system_overview.mmd)

<details>
<summary>查看可编辑的 Mermaid 连接图</summary>

```mermaid
flowchart TB
    camera["OV5640 摄像头<br/>外部图像输入"]:::ext
    video["#20 视频采集 / VDMA<br/>PL 通路与 DDR 帧缓冲"]:::pl
    ui["#1 实时画面与界面<br/>原始帧读取 → UI / 结果叠加"]:::ps
    lcd["RGB LCD · 800×480<br/>经 #20 视频输出显示"]:::ext
    touch["触摸屏 → #21 IIC / GPIO<br/>独立触摸进程生成坐标事件"]:::ps
    keyboard["USB 键盘 → #22 输入<br/>Host / evdev / 键盘线程"]:::ps
    events["#27 输入同步与状态保护<br/>按时间顺序合并触摸和按键"]:::ps
    control["#2 / #6 / #10 交互控制<br/>单框 ROI · 类别命名<br/>模式切换 · 冻结 / 字形预览"]:::ps
    jobs["#3 / #4 / #5 / #8 / #9<br/>A 学习 · B 手动 / 自动扫描<br/>D 数字 / 算符识别"]:::ps
    engine["#25 / #13 / #14 / #19 / #18<br/>共享特征处理通路<br/>任务串行 · 图像预处理<br/>适配 / DMA ↔ 同一个 PL TCU"]:::pl
    result["#15 / #16 / #11 / #12<br/>学习提交 · NCM / 扫描筛选<br/>候选确认 → 表达式 → PS 运算"]:::ps
    data["#7 / #17 数据管理<br/>A/B 学习数据及名称<br/>数字库 / 算符库独立加载"]:::data
    watchdog["#26 触摸看门狗<br/>父进程监测心跳<br/>超时复位并重建触摸进程"]:::tool
    support["#23 平台 · #24 初始化 · #28 诊断 · #29 退出 · #30 构建 · #31 验证 · #32 USB 修复<br/>为上层链路提供配置、运行保障与工程维护，展开关系见图 4"]:::tool
    camera --> video
    video -->|"图像帧"| ui
    ui -->|"叠加后的画面"| lcd
    ui -->|"原始帧 / 冻结帧选择"| control
    touch --> events
    keyboard --> events
    events --> control
    control -->|"ROI / 模式 / 名称"| jobs
    jobs --> engine
    engine -->|"80 维特征"| result
    data <-->|"样本"| result
    result -->|"识别 / 计算结果返回 #1"| ui
    watchdog <-.->|"心跳监测 / 复位重建"| touch
    classDef ps fill:#EDF4FB,stroke:#93B2CB,color:#173C5B
    classDef pl fill:#EDF6EF,stroke:#96BBA2,color:#275639
    classDef data fill:#FFF5DF,stroke:#D3BA85,color:#624B22
    classDef tool fill:#F3F4F6,stroke:#B2BAC4,color:#364457
    classDef ext fill:#FFFFFF,stroke:#9CAABA,color:#25394C
```

</details>

## 二、A/B 学习与物品识别

A 模式建立 A/B 学习数据；B 手动使用 NCM，SCAN 使用余弦匹配与候选筛选，两条路径共享学习数据和特征提取服务。学习数据不会被一次识别消耗。

![二、A/B 学习与物品识别](flowcharts/02_ab_learning_recognition.svg)

[SVG 矢量图](flowcharts/02_ab_learning_recognition.svg) · [Mermaid 源文件](flowcharts/02_ab_learning_recognition.mmd)

<details>
<summary>查看可编辑的 Mermaid 连接图</summary>

```mermaid
flowchart TB
    a0["#3 A 学习请求<br/>画框 → 可选 #6 命名 / Enter → 点 1～4"]:::ps
    a1["#13 ROI 裁剪 / 缩放<br/>取未叠加 UI 的原始图像 → 32×32"]:::ps
    a2["#25 / #19 / #18 共享 TCU 调用<br/>输出 80D 特征；默认累计 8 帧"]:::pl
    a3["#3 学习完成并提交<br/>特征 + HSV + 框尺寸 + 显示名称"]:::ps
    bank["#7 A/B 内存学习数据<br/>class1～class4；供手动与扫描重复使用"]:::data
    npz["#7 pefsl_learned.npz<br/>SAVE 写文件 / LOAD 恢复 / CLEAR 清内存"]:::data
    b0["#4 B 手动 ROI<br/>同一个框按周期刷新"]:::ps
    b1["#13 ROI 裁剪 / 缩放<br/>原始图像 → 32×32"]:::ps
    b2["#25 / #19 / #18 共享 TCU 调用<br/>输出 80D 特征"]:::pl
    b3["#15 A/B NCM 分类<br/>全局居中 / 类别均值 / 距离分数"]:::ps
    b4["#4 名称与类间分数<br/>读取 A/B 数据中的类别名称"]:::ps
    b5["#1 LCD 叠加显示<br/>框内物体变化 → 再次识别"]:::ps
    s0["#5 多尺度轮换窗口<br/>使用已学习框尺寸生成候选区域"]:::ps
    s1["#13 窗口裁剪 + 颜色统计<br/>32×32 图像 / HSV / 中心与周围差"]:::ps
    s2["#25 / #19 / #18 共享 TCU 调用<br/>输出 80D 特征"]:::pl
    s3["#16 余弦匹配与背景筛选<br/>类别分差 / 颜色 / 对比 / 尺寸限制"]:::ps
    s4["#16 候选确认<br/>每类优选 → NMS → 连续周期确认"]:::ps
    s5["#1 LCD 叠加显示<br/>已确认目标框 + 名称 + 匹配分数"]:::ps
    shared["共用特征服务：32×32 BGR → 数值预处理 / 12 路打包 → DMA ↔ PL TCU → raw84 拆分 → feature80<br/>调用框表示同一服务的多个入口；B 手动使用 NCM，SCAN 使用独立的余弦匹配与筛选流程"]:::tool
    a0 --> a1
    a1 --> a2
    a2 --> a3
    a3 --> bank
    b0 --> b1
    b1 --> b2
    b2 --> b3
    b3 --> b4
    b4 --> b5
    s0 --> s1
    s1 --> s2
    s2 --> s3
    s3 --> s4
    s4 --> s5
    bank <--> npz
    bank -->|"特征"| b3
    bank -->|"学习特征 / 颜色 / 框尺寸"| s3
    bank -.->|"框尺寸"| s0
    classDef ps fill:#EDF4FB,stroke:#93B2CB,color:#173C5B
    classDef pl fill:#EDF6EF,stroke:#96BBA2,color:#275639
    classDef data fill:#FFF5DF,stroke:#D3BA85,color:#624B22
    classDef tool fill:#F3F4F6,stroke:#B2BAC4,color:#364457
    classDef ext fill:#FFFFFF,stroke:#9CAABA,color:#25394C
```

</details>

## 三、D 数字、算符与计算

D 先截取原尺寸字符 ROI，再做字形预处理；识别结果经过稳定检查并由用户确认后才写入表达式。数字和算符各自使用独立支持库，算术在 PS 执行。

![三、D 数字、算符与计算](flowcharts/03_digit_operator_calculator.svg)

[SVG 矢量图](flowcharts/03_digit_operator_calculator.svg) · [Mermaid 源文件](flowcharts/03_digit_operator_calculator.mmd)

<details>
<summary>查看可编辑的 Mermaid 连接图</summary>

```mermaid
flowchart TB
    frame["#10 实时 / 冻结原始帧<br/>FREEZE 保存原始帧副本"]:::ps
    roi["#2 D 字符 ROI<br/>先取原尺寸的单个字形"]:::ps
    glyph["#14 字形预处理<br/>灰度 / 二值 / 形态 → 32×32"]:::ps
    input["#19 输入适配<br/>数值预处理 / 12 路打包 / DMA"]:::ps
    tcu["#18 同一个 PL TCU<br/>现有 ResNet 骨干提取特征"]:::pl
    output["#19 输出适配<br/>按模型分段：raw84 → feature80"]:::ps
    preview["#10 字形预览<br/>显示 32×32 二值化输入"]:::ps
    cache["#17 独立缓存文件<br/>数字 0～9 支持库<br/>加 / 减 / 乘 / 除支持库"]:::data
    banks["#17 启动加载<br/>数字 / 算符独立数据区<br/>OP MODE 选择其中一套"]:::data
    classify["#8 / #9 / #15 NCM 分类<br/>数字分类器 / 算符分类器<br/>各自的均值、样本与预测历史"]:::ps
    stable["#11 候选稳定检查<br/>平滑 / 门限 / 连续一致<br/>屏幕结果与当前候选一致"]:::ps
    confirm["#11 READY 后用户确认<br/>冻结 + Xn / OPn；防止重复存入"]:::ps
    expression["#11 表达式编辑<br/>数字 / 算符交替；UNDO / RESET"]:::ps
    calc["#12 PS 四则运算<br/>OUT → 有理数计算 / 错误处理"]:::ps
    display["#1 LCD 结果显示<br/>表达式、结果或错误提示"]:::ps
    build["#17 支持图片 → #14 相同字形预处理 → #18 实板 TCU → 数字 / 算符缓存文件<br/>建库命令独立运行；A/B 的 SAVE / LOAD 不会加载或修改这两套支持库"]:::tool
    frame --> roi
    roi --> glyph
    glyph --> input
    input --> tcu
    tcu -->|"raw84"| output
    output -->|"80D"| classify
    glyph --> preview
    cache --> banks
    banks -->|"支持样本"| classify
    classify --> stable
    stable -->|"READY"| confirm
    confirm --> expression
    expression --> calc
    calc --> display
    classDef ps fill:#EDF4FB,stroke:#93B2CB,color:#173C5B
    classDef pl fill:#EDF6EF,stroke:#96BBA2,color:#275639
    classDef data fill:#FFF5DF,stroke:#D3BA85,color:#624B22
    classDef tool fill:#F3F4F6,stroke:#B2BAC4,color:#364457
    classDef ext fill:#FFFFFF,stroke:#9CAABA,color:#25394C
```

</details>

## 四、启动、触摸恢复与工程保障

左列为启动顺序，中列为运行中的独立触摸恢复，右列为系统及工程支撑；右列各框不是一条依次执行的流水线。

![四、启动、触摸恢复与工程保障](flowcharts/04_startup_recovery_tools.svg)

[SVG 矢量图](flowcharts/04_startup_recovery_tools.svg) · [Mermaid 源文件](flowcharts/04_startup_recovery_tools.mmd)

<details>
<summary>查看可编辑的 Mermaid 连接图</summary>

```mermaid
flowchart TB
    build["#30 构建与部署<br/>Vivado / Tcl → 匹配的 bitstream / HWH"]:::tool
    overlay["#24 加载单个 Overlay<br/>初始化同一套视频 / 触摸 / TCU 硬件"]:::ps
    touch["#21 / #26 初始化触摸并 fork<br/>先启动独立 IIC 子进程，再创建其他线程"]:::ps
    models["#24 / #25 加载模型并创建 UI<br/>加载 #17 支持库，启动单个 TCU 后台线程"]:::ps
    video["#22 键盘 → #20 视频初始化<br/>启动键盘线程 → 摄像头 / LCD / 预热帧"]:::ps
    loop["#1 / #20 / #27 主循环<br/>读帧 → 同步输入 → 提交任务<br/>叠加 UI → 归还视频帧 → 下一帧"]:::ps
    monitor["#26 父进程监测<br/>读取心跳 / 检查子进程是否退出"]:::tool
    timeout["出现心跳超时或进程退出<br/>正常时继续监测，异常才进入恢复"]:::tool
    kill["#26 结束旧 IIC 子进程<br/>释放阻塞在底层调用中的执行单元"]:::tool
    reset["#21 触摸与 IIC 复位<br/>GPIO 硬复位 → AXI IIC SOFTR"]:::tool
    restart["#26 重建并初始化触摸子进程<br/>恢复坐标读取和心跳"]:::tool
    sync["#27 恢复输入处理<br/>继续同步触摸 / 键盘事件<br/>取消失效的旧输入状态"]:::ps
    usbfix["#32 USB 启动修复 / 回退<br/>匹配镜像 → 校验 / 备份 → 设备树修复"]:::tool
    host["#22 Linux USB Host<br/>重启后枚举键盘 → evdev 输入"]:::ps
    validate["#31 模块验证 / 回归<br/>触摸、键盘、TCU 边界及应用软件检查"]:::tool
    platform["#23 7020 板级平台<br/>PS / DDR / AXI / 时钟 / 复位 / 管脚"]:::pl
    health["#28 运行状态 / 诊断<br/>主循环、触摸与键盘健康 → 串口日志"]:::tool
    exit["#29 安全退出<br/>Ctrl-C / 时限 → 停止任务与进程<br/>释放视频帧、键盘和运行资源"]:::tool
    build --> overlay
    overlay --> touch
    touch --> models
    models --> video
    video --> loop
    monitor -.-> timeout
    timeout -.-> kill
    kill -.-> reset
    reset -.-> restart
    restart -.-> sync
    sync -.-> monitor
    touch -.->|"心跳"| monitor
    usbfix -.->|"重启后"| host
    validate -.->|"旁路验证，不是每帧步骤"| touch
    platform -.->|"底层平台支撑"| overlay
    loop -.->|"状态观测"| health
    loop -.->|"Ctrl-C / 运行时限"| exit
    classDef ps fill:#EDF4FB,stroke:#93B2CB,color:#173C5B
    classDef pl fill:#EDF6EF,stroke:#96BBA2,color:#275639
    classDef data fill:#FFF5DF,stroke:#D3BA85,color:#624B22
    classDef tool fill:#F3F4F6,stroke:#B2BAC4,color:#364457
    classDef ext fill:#FFFFFF,stroke:#9CAABA,color:#25394C
```

</details>

## 五、读图时需要注意的连接关系

- **一个 TCU，多条调用路径**：A、B 手动、SCAN、D 都由同一个后台线程串行调用，图 2 的三个 TCU 调用框不是三个硬件核。
- **原始图像先用于识别，再叠加 UI**：A/B 的 ROI 先缩放到 32×32；D 则先保留原尺寸 ROI 做字形处理，随后进入同一 TCU 数值预处理和输入打包。
- **SCAN 与手动分类分开**：B 手动使用 NCM；SCAN 使用独立余弦匹配、背景筛选、NMS 和跨周期确认。
- **三套数据相互独立**：A/B 学习数据由 SAVE/LOAD/CLEAR 管理；D 数字和算符库由独立建库工具生成、启动时加载；计算式使用 UNDO/RESET/OUT。
- **键盘只先确认名称**：Enter 完成名称确认，随后点击类别 1～4 才启动采样，特征和名称在采集成功后一起提交。
- **冻结不停止视频输入**：FREEZE 保存原始帧副本供识别和显示，主循环仍持续读取并归还摄像头帧。
- **看门狗不串入正常输入链**：它由父进程独立监测心跳，异常时复位和重建触摸进程，正常触摸事件直接进入输入队列。
- **构建与 USB 修复不属于每帧处理**：Vivado 生成硬件交付，USB 修复更新匹配镜像的启动设备树，二者分别为运行初始化和 Host 枚举提供条件。

## 六、32 个模块的图中位置

| 编号 | model.md 模块 | 所在图 |
|---|---|---|
| 1 | 实时画面与 LCD 界面 | 图 1、2、3、4 |
| 2 | 触摸交互与 ROI 框选 | 图 1、3 |
| 3 | A 模式少样本学习 | 图 1、2 |
| 4 | B 模式手动持续识别 | 图 1、2 |
| 5 | B 模式自动扫描 | 图 1、2 |
| 6 | USB 键盘类别命名 | 图 1、2 |
| 7 | 学习数据管理 | 图 1、2 |
| 8 | D 模式数字识别 | 图 1、3 |
| 9 | D 模式算符识别 | 图 1、3 |
| 10 | 画面冻结与字形预览 | 图 1、3 |
| 11 | 候选确认与表达式编辑 | 图 1、3 |
| 12 | 四则运算与结果显示 | 图 1、3 |
| 13 | ROI 图像预处理 | 图 1、2 |
| 14 | 数字与算符字形预处理 | 图 1、3 |
| 15 | 少样本 NCM 分类 | 图 1、2、3 |
| 16 | 扫描候选筛选与确认 | 图 1、2 |
| 17 | 数字与算符支持库 | 图 1、3、4 |
| 18 | TCU 硬件特征提取 | 图 1、2、3 |
| 19 | TCU 数据适配与 DMA 驱动 | 图 1、2、3 |
| 20 | 摄像头采集与视频传输 | 图 1、4 |
| 21 | 触摸 IIC 与 GPIO 驱动 | 图 1、4 |
| 22 | USB Host 与键盘输入 | 图 1、4 |
| 23 | 7020 板级 PS/PL 平台 | 图 1、4 |
| 24 | Overlay 与模型初始化 | 图 1、4 |
| 25 | TCU 后台任务调度 | 图 1、2、3、4 |
| 26 | 触摸看门狗与自动恢复 | 图 1、4 |
| 27 | 输入事件同步与状态保护 | 图 1、4 |
| 28 | 运行状态与故障诊断 | 图 1、4 |
| 29 | 安全退出与资源清理 | 图 1、4 |
| 30 | 硬件构建与部署交付 | 图 1、4 |
| 31 | 模块验证与软件回归 | 图 1、4 |
| 32 | USB 启动修复与回退 | 图 1、4 |

## 七、源码对应

- [主入口与输入同步、触摸恢复](../pynq_app/pynq_bbox_demo.py)：`main()`、`_dispatch_ui_inputs()`、`TouchProcessWorker`。
- [模式状态与任务调度](../pynq_app/pefsl_touch_ui.py)：`process_frame()`、`_prediction_worker()`、`LearnedData`、`PefslNcmClassifier`。
- [扫描筛选](../pynq_app/pefsl_scan.py)：窗口生成、余弦评分、颜色统计与重叠抑制。
- [字形处理](../pynq_app/digit_preprocess.py)、[支持库](../pynq_app/digit_recognizer.py)、[计算器](../pynq_app/digit_calculator.py)。
- [TCU 数据适配](../tcu_deploy/tcu_bbox_backend.py)、[触摸驱动](../touch_deploy/touch_driver.py)、[键盘输入](../pynq_app/usb_keyboard.py)。
