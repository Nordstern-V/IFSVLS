# PEFSL_MIX_PLUS / 领航者 7020 案例

此页是从本次工程提取的已知案例快照（2026-11-02），用于寻找参照文件和解释曾发生的错误，不是通用默认配置。路径或源码已变时以用户当前主工程和现场文件为准。

## 核对入口

本次主工程曾位于 `E:/FPGA-3up/PEFSL_MIX_PLUS`，原始 PEFSL 在 `E:/FPGA-3up/PEFSL/PEFSL-main-origin`。Linux 部署在 `/home/xilinx/jupyter_notebooks/PEFSL_MIX_PLUS`。这两个路径来自案例，不能替换新用户的路径。

| 证据 | 工程相对路径 |
|---|---|
| 当前版本与主循环、输入合并、触摸进程恢复 | `pynq_app/pynq_bbox_demo.py` |
| UI、后台任务、NCM、学习数据、候选提交 | `pynq_app/pefsl_touch_ui.py` |
| 扫描评分与重叠抑制 | `pynq_app/pefsl_scan.py` |
| 特征预处理、打包与输出边界 | `tcu_deploy/tcu_bbox_backend.py` |
| IIC 事务、GPIO、FT/GT、SOFTR | `touch_deploy/touch_driver.py` |
| 字形、缓存、计算、键盘 | `pynq_app/digit_preprocess.py`、`digit_recognizer.py`、`digit_calculator.py`、`usb_keyboard.py` |
| 触摸硬件 | `scripts/add_touch_interfaces.tcl`、`artifacts/*.hwh` |
| USB 修复和回退 | `usb_host_fix/README.md`、`apply_usb_host_fix.py` |
| 当前阶段与验证范围 | `conclusion/conclusion.md`、`conclusion/model.md`、`pynq_app/AB_R17_RESTORE.md` |
| 旧构建报告 | `scripts/mix_plus_impl_utilization.rpt`、`scripts/mix_plus_impl_timing.rpt` |

## 已知配置与边界

- 目标器件 `xc7z020clg400-2`；现有工程记录为 Vivado 2025.2，镜像为正点原子 PYNQ 2.7.0 / Python 3.8 / ARMv7 / Linux 5.4.0-xilinx-v2020.2。
- 启动解释器曾为 `/usr/local/share/pynq-venv/bin/python3`；用它的实际输出确认，不依赖系统 `python3` 恰好相同。
- 视频为 OV5640 → PL/VDMA/DDR → PS UI → RGB LCD，显示 800×480、24 bpp，LCD ID `0x7084`；未再合并触摸示例的第二套 LCD 驱动。
- 触摸 AXI IP 位于 PS GP0 M04/M05/M06，连接 FCLK0 与外设复位；HWH IIC 频率 100 kHz。

| 接口 | 本例管脚 | 本例 AXI 地址 |
|---|---|---|
| `touch_iic` SCL / SDA | R19 / P20 | `0x41610000` |
| `touch_int_gpio` | U19 | `0x41620000` |
| `touch_rst_gpio` | M19 | `0x41630000` |

R19/P20 由移除旧 HDMI DDC 外部端口后复用；新板须重新核对复用关系。实板验证控制器为 FT、7 位地址 `0x38`、version 1；GT 分支存在但不代表实板验证过。五点坐标能力不代表 UI 支持多指编辑。

模型输入为 32×32 uint8 BGR，按现有运算顺序预处理，打包为 `(1024,12)` 前三路有效、其余补零。驱动拼接输出段 72+12=84；第一段全 72 与第二段前 8 组成 feature80，尾 4 用于诊断。不是所有 TCU 都输出 80/84，也不能假定 DRAM 输出天然连续。

## 这次排障中应保留的判断

| 现象 | 实际修正方向 | 不应从中推导的结论 |
|---|---|---|
| GPIO write 缺 mask | 检查旧 channel API，按真实签名调用 | 所有 PYNQ GPIO 都需 mask |
| repeated START 后 wait bus free 超时 | 保持地址阶段到接收的事务边界 | 所有 IIC 超时都来自同一原因 |
| 独立触摸通过，整机长拖动卡住 | 事务修正后仍需独立进程与真实心跳恢复 | 只延长轮询即可解决；无异常文本即健康 |
| watchdog 打印成功仍无输入 | 复验事件前进，处理旧 C 调用未释放 | 复位返回成功就证明触摸已恢复 |
| LCD 只有首帧、Ctrl-C 慢 | 视频通路关联、后台推理、有限等待与帧所有权 | 画面不动必然要换摄像头 |
| lsusb/sysfs 为空 | 证据指向 USB0 DT 缺 PHY 引用/Host 模式，制作匹配启动补丁 | 任意镜像使用相同 image.ub 即可 |
| 新 A/B 效果弱，用户要求回退 | 恢复指定 r17 算法，保留新键盘/数字/计算器 | 整目录覆盖旧版，顺带删缓存 |

## 当前应用语义快照

版本曾为 `mix-plus-ab-r17-20261101-r21-keyboard-calc10`，触摸驱动为 `mix-plus-touch-20261024-r16`。A/B 恢复指定 r17；C124 身份门、区域候选、局部回访和结果过期策略不在活动流程。

A 默认采集 8 帧、4 个物体类别；名称与特征成功后提交。B 手动 NCM 持续刷新，SCAN 用独立余弦/背景/尺寸/NMS/周期确认；进入 B 有数据时启动扫描，手动画框会暂停。

D 使用独立数字/算符支持库，共用 TCU。FREEZE 后等 READY，由用户 Xn/OPn 确认后组成表达式；最多 10 个一位数字和 9 个算符，PS 使用 Fraction 计算。它不是多位数 OCR，也不是新 PL 算术核。

SAVE/LOAD/CLEAR 管理 A/B 数据；CLEAR 清内存而不删磁盘文件，不替代计算器 RESET。A/B 本身能持久化，D 是固定支持库复用；当前没有将 A/B 自动巩固为 D 库的机制。

本例 USB0 为 MIO28～39、复位 MIO9，USB3320 → GL850G HUB；P4 Host 接法 3–5、4–6。USB UART 为独立串口路径。用户实板日志已出现 PHY/root hub/HUB/CX 键盘，但不要将此接线套用到其他型号。

用户验证了视频实时、FT 四角/五点、触摸恢复、TCU gradient/impulse raw84/feature80/重复一致、USB 键盘枚举和字母命名。后加字符、回退后准确率及数字计算性能应按现有日志分别判断，不能一概标为整机全面通过。

2026-10-18 旧实现记录为 LUT 30240、FF 34802、BRAM Tile 70.5、DSP 159；虽 WNS +0.414 ns，同时有 cam_pclk 无时钟 324 项与未约束内部端点 615 项。此记录不代表后来重新构建，也不是完整时序验收。

## 示例来源的使用方式

组织方式参考用户提供的 `ross-ai-assistant-main/skills` 中 IP 配置、ILA 和时序检查示例：先识别状态、分层证据、读回与结果验证、按需加载参考资料。这里的指令为本次工程经验重新整理，不复制其专有 MCP、helper 或版本限制；不要求 Vivado 2026.1、chipscope、Python 3.10 或“只能用某一个工具”。
