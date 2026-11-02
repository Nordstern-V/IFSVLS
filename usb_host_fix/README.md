# USB Host 修复包（2026-11-01）

适用于已检查的正点原子 `ZYNQ-7020-2.7.0.img`，以及 boot `image.ub` 与其 SHA256 完全一致的开发板。P4 保持 Host 接法（3-5、4-6）；USB UART 可继续与电脑连接，作为串口控制台。

现场证据：`/proc/device-tree/amba/usb@e0002000/` 缺少 `usb-phy` 与 `dr_mode`，USB0 无 driver 绑定，`/sys/bus/usb/devices` 为空。原镜像内核的 ChipIdea OF/Host、ULPI、HID、evdev 均已内建，因此本包修复启动设备树。

修复内容：在原始 DTB 根节点增加 ULPI PHY（寄存器范围 0xe0002000/0x1000，view-port 0x170，drv-vbus），USB0 增加 `usb-phy` 引用和 `dr_mode="host"`，重新计算 FIT 中的 FDT SHA1。内核 payload 逐字节保留，其他原始设备树属性、内存保留区域和 FIT 参数均已比对保持。原始镜像没有 FIT 签名。

官方参考（同为 xilinx-v2020.2）：
- https://github.com/Xilinx/linux-xlnx/blob/xilinx-v2020.2/drivers/usb/chipidea/ci_hdrc_usb2.c （读取 Zynq usb-phy，缺少引用即返回）
- https://github.com/Xilinx/linux-xlnx/blob/xilinx-v2020.2/arch/arm/boot/dts/zynq-zc702.dts （ULPI PHY 与 usb-phy/host 配置）

## 部署

把整个 `usb_host_fix` 文件夹复制到：

```text
/home/xilinx/jupyter_notebooks/PEFSL_MIX_PLUS/usb_host_fix
```

在串口终端执行：

```bash
cd /home/xilinx/jupyter_notebooks/PEFSL_MIX_PLUS/usb_host_fix
sudo python3 apply_usb_host_fix.py --check
sudo python3 apply_usb_host_fix.py --apply
```

安装器会校验源和目标 SHA256，先备份原始启动文件，再写入经校验的临时文件并替换 `image.ub`。看到 `USB Host DT fix installed successfully.` 后执行：

```bash
sudo reboot
```

若报告 `boot image not found`，先用 `lsblk -f`、`findmnt /boot`、`ls -l /boot` 确认 SD 卡启动分区实际挂载位置，再通过 `--boot-image /实际启动分区/image.ub` 指定；不要指向工作目录中无关的 image.ub。如果 SHA256 不匹配，安装器拒绝写入，应提供实际启动用的 image.ub 来生成匹配修复，不能跳过校验。

原始 SHA256：

```text
b502ca462cb57a43834c5cd0756ac38c6a28d6760cf27e942ea97dacd36c91f1
```

修复后 SHA256：

```text
b72b9ee15590152d88239b17887ba5aa0b8699e4fe53cbebcac406beea4ff3aa
```

## 重启后验证

先在串口检查 USB 枚举：

```bash
sudo dmesg | grep -iE 'usb|ci_hdrc|ehci|ulpi'
lsusb
cat /proc/bus/input/devices
ls -l /dev/input/event*
```

预期先出现 USB root hub，再出现板载 GL850G USB HUB 与键盘，随后出现键盘输入节点 `eventX`（编号不固定）。正常 PHY 日志通常包含 USB3320 与 ULPI integrity check。若 root hub 出现而没有板载 HUB，继续查 HUB 路径/电源；若出现 root hub、HUB、键盘却没有 event 节点，继续查输入驱动。保留完整启动日志用于区分层级。

本包已经完成离线结构、FIT 哈希、内核不变性和安装/回退流程的主机测试；尚未进行实板启动验证。本包解决 USB Host 枚举，类别命名输入属于后续应用功能。

## 回退

可进入 Linux 时：

```bash
cd /home/xilinx/jupyter_notebooks/PEFSL_MIX_PLUS/usb_host_fix
sudo python3 apply_usb_host_fix.py --restore
sudo reboot
```

原始文件备份名为 `image.ub.before-usb-host-20261101`，与启动分区中的 `image.ub` 放在同一目录。若启动未能进入 Linux，可关机取出 SD 卡，在电脑打开启动分区，用该备份的内容恢复为 `image.ub`。保留备份用于再次回退。
