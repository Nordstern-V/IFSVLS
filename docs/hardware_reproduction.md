# 硬件工程复现

本仓库保留当前 `PEFSL_MIX_PLUS` 使用的摄像头、LCD、触摸和 Tensil TCU 设计输入，以及配套的已部署 Overlay。硬件目标是正点原子领航者 PYNQ/Zynq-7020，器件为 `xc7z020clg400-2`，OV5640 使用 800×480@30 配置；引脚约束以本仓库 `base.xdc` 为准。

## 工具与依赖

- AMD Vivado **2025.2**，安装 Zynq-7000 器件支持及有效工具许可；无需安装其他板卡定义，工程使用器件型号和显式 PS/引脚配置。
- Tcl 构建脚本通过自身位置寻找仓库，允许 checkout 到其他目录。
- 自定义 IP 原始文件已保留：Digilent `axi_dynclk`、Alientek `cmos_capture_data`、Alientek `rgb2lcd` 及 LCD 接口 XML。保留原始版权声明，未给第三方代码重新授予许可。
- 其余 AXI、视频、PS7、时钟/复位和 DMA IP 使用 Vivado 安装自带的 IP 目录；仓库中的 `.xci` 用于固定 IP 配置。
- TCU 的 `top_custom_perf.v` 和两个 BRAM 模块位于 `sources/tcu/`，不依赖机器外部的 HDL 或初始化文件。运行模型和驱动见 `tcu_deploy/`。

## 直接使用现有 Overlay

板端部署使用同名的一对文件：

```text
artifacts/pefsl_mix_plus_touch.bit
artifacts/pefsl_mix_plus_touch.hwh
```

两者必须成对更新。配合 `pynq_app/`、`touch_deploy/` 和 `tcu_deploy/` 使用，详见仓库根 README。重新生成 FPGA 不要求重新生成 TCU 模型；本仓库提供与该 TCU 架构匹配的现有模型运行文件，并不包含训练数据集或模型训练全过程。

## 从工程源重建

先在 Vivado 2025.2 的命令环境中进入仓库根目录，运行依赖检查：

```bash
vivado -mode batch -source scripts/rebuild_hardware.tcl -tclargs --check
```

该命令打开所选 `.xpr`、刷新本仓库 IP 目录、验证 Block Design，并重新生成 IP 目标产物；它不运行综合或实现。成功时输出 `HARDWARE_DEPENDENCY_CHECK_PASS`。

完整构建：

```bash
vivado -mode batch -source scripts/rebuild_hardware.tcl -tclargs --jobs 4
```

完成后检查 `build/hardware/timing_summary.rpt` 与 `utilization.rpt`，并在实板上验证新生成的文件：

```text
build/hardware/pefsl_mix_plus_touch.bit
build/hardware/pefsl_mix_plus_touch.hwh
```

脚本不会自动替换已发布的 `artifacts/`。不同工具版本或重新布局布线可能导致不同的二进制哈希；时序和板端运行验证应以本次构建结果为准。

如偏好图形界面，打开：

```text
vivado_project/ZYNQ_7020_PYNQ/6_ov5640_lcd/vivado/ov5640_lcd_tcu_touch.xpr
```

Windows 用户应把仓库放在较短路径（例如 `E:/IFSVLS`），避免 Vivado 生成文件超过 Windows 路径长度限制。

## 本次精选文件的处理

- 保留 BD、XPR、XDC、XCI、实际使用的 IP 源码、TCU RTL、顶层 wrapper 与现有 Overlay；不上传 `.runs`、`.cache`、`.gen`、`.hw`、`.Xil`、日志和历史备份。
- 将原先位于 `ov5640_lcd_tcu_merged_fix3.gen` 的小型顶层 wrapper 固定到 `ov5640_lcd_tcu_touch.srcs/sources_1/hdl/base_wrapper.v`，避免依赖被排除的生成目录。
- 发布版 XPR 删除原机器绝对工程路径并修改 wrapper 引用。BD/XCI 保留原始字节和参数，沿用相对的 `ov5640_lcd_tcu_merged_fix3.gen` 作为可重新生成的输出目录；不上传该生成目录。
- `sources/touch_reference/` 是触摸控制器参考 RTL；当前活动设计通过标准 AXI IIC 和 GPIO 驱动触摸，不把这些参考模块直接接入视频链路。
- 此次发布不重新运行完整综合/实现，也未重新进行实板验证；板端复现仍需匹配的 PYNQ 镜像、板卡、摄像头、LCD 和接线。
