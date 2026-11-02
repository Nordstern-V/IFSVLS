# 板级、Overlay 与 USB 分层定位

## 板级 / IP 移植

先找目标板实际配置：part、DDR、MIO/EMIO、外设电源/时钟、复位极性、管脚复用，再对照 Block Design、XDC 和 HWH。相同 Zynq 型号不意味着两块板的 PS 配置和接口可直接互换。

检查新增 AXI IP 的地址、PS master 可达性、数据位宽、接口时钟、复位同步和必要的 CDC。复制示例 RTL 前确认已有视频/显示链路，避免再接入一套互相竞争的 LCD 控制逻辑。

需要改硬件时先保存当前基线；验证 BD 后生成匹配的硬件描述与 bitstream，再部署该对文件。不要只更新 bit 而让旧 HWH 继续描述错误地址，也不要看到 Python 异常就默认硬件必须重建。

优先从实际文件读取 IP 参数；设置后核验读回与 `validate_bd_design`。若工具忽略参数、接口不兼容或地址未分配，修正该原因，再重跑必要阶段。既有 IP 升级和整库重生成不是每次修改的前置步骤。

资源结论注明器件、报告时间及综合/实现阶段；Slice、LUT、FF、BRAM、DSP 分开报告。时序至少同时看 clocks、no_clock、unconstrained endpoints、跨域路径与 WNS/TNS，不把“生成 bitstream”当成完整时序验收。

## PYNQ 包装与部署一致性

先检查实际解释器和包，而不是直接升级系统 Python：

```bash
python3 -c 'import sys; print(sys.executable); print(sys.version)'
```

用工程原来的板端 Python 再检查 PYNQ 版本及有关驱动源码/函数签名。独立脚本和整机应用可能使用了不同解释器、驱动副本或同名导入路径。

对 IP 版本告警，先核对 VLNV 与该 PYNQ 驱动寄存器契约。只有兼容性已经核实，才使用对应的版本兼容选项；不要全局压制告警或默认所有新版 IP 都兼容旧驱动。

采集脚本会记录本地文件哈希、同名 bit/HWH 是否存在和 HWH 中的 IP/地址。它**不会**检查板上当前配置寄存器，也无法只凭两份哈希证明文件来自同一构建；与部署清单、启动输出和实际加载路径一起判断。

## USB 没有键盘：先定位消失在哪一层

用户描述“USB 插入无反应”时，可按缺失信息选用下列只读命令，不必每次全部执行：

```bash
lsusb
lsusb -t
ls -l /sys/bus/usb/devices/
cat /proc/bus/input/devices
ls -l /dev/input/event* /dev/input/by-id/*-event-kbd
dmesg | grep -iE 'usb|ci_hdrc|ehci|ulpi|hid|input'
```

`dmesg` 被权限拒绝时记录这一事实；其他检查继续，不把读取失败解释为没有 USB 日志。确需内核配置时读取运行镜像的配置，确认是否有 Host/控制器/PHY/HID/evdev，不能仅看“注册了 usbcore”。

| 证据 | 下一层检查 |
|---|---|
| 没有 root bus，sysfs USB 目录空 | 控制器 DT status/compatible/时钟、dr_mode、PHY 引用、驱动绑定与内核配置 |
| root hub 存在，板载 HUB 不出现 | 本板 Host 接线、VBUS/HUB 电源、PHY/HUB 复位和走线 |
| HUB 存在，设备不枚举 | 端口、电缆/接收器、供电及枚举错误 |
| 键盘已枚举，无输入 event 节点 | HID、输入子系统及 evdev 支持 |
| event 节点存在，应用不收字符 | 设备选择、权限、事件 ABI、抓取者、事件解码与线程存活 |

缺少 `usb-phy` 不是所有平台的统一补丁：绑定可能使用不同 PHY 框架，路径和属性须与目标内核及设备树一致。USB UART 与 USB Host 是否独立也由原理图决定，不把一个串口连接猜成 USB0 被占用。

只有证据指向 DT 且修改已在任务范围内，才制作候选：识别 DTB/FIT 和真实启动分区；保留内核、内存保留区、原有属性；按该镜像格式处理哈希/签名；保存原镜像并验证目标哈希。FIT 有签名时，不能只重算普通哈希代替签名。不要把单板 `image.ub` 复制到另一套镜像试运气。

## evdev 键盘接入

复合接收器会提供多个 event 接口；按字母键及 Enter 能力选择，不固定 `event0`，也不把 Consumer Control 当主键盘。PYNQ 常见 ARMv7 为 32 位，读取 `input_event` 时使用运行平台 ABI，避免直接套用 x86-64 的事件长度。

键盘线程只读取和排队，由 UI 线程更新状态；需要独占时处理抓取/释放、拔插重连、SYN_DROPPED 和修饰键重置。应用使用物理键盘 Ctrl-C 时需显式解码；它与串口 SIGINT 不一定是同一条路径。
