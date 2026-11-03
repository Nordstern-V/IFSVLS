# IFSVLS

基于正点原子领航者 ZYNQ-7020 的交互式少样本视觉学习工程：OV5640 摄像头、800×480 RGB LCD、触摸框选、Tensil TCU 特征提取、A/B 少样本学习与识别、USB 键盘命名，以及数字/算符识别和四则运算。

整理于 2026-11-02。当前应用版本为 `mix-plus-ab-r17-20261101-r21-keyboard-calc10`。A/B 使用恢复后的 r17 流程；历史 C124 身份门等策略未启用。PC 参考程序与板端当前算法的区别见软件复现说明。

## 从这里开始

1. 克隆本仓库；板端目录建议命名为 `PEFSL_MIX_PLUS`，保持各子目录之间的相对位置。
2. 执行 `python tools/verify_reproduction.py` 检查交付文件的 SHA256、TCU 模型配套文件和 Python 语法。
3. 按 [软件复现说明](docs/software_reproduction.md) 准备对应的厂家 PYNQ 镜像、摄像头、屏幕、触摸和模型，部署到板卡。
4. 如需重新综合硬件，按 [硬件复现说明](docs/hardware_reproduction.md) 使用 Vivado 2025.2；日常运行可直接使用已经配套的 `artifacts/*.bit` 和 `.hwh`。
5. 使用 [LCD 按键说明](conclusion/button.md) 操作 A/B/D 模式、SAVE/LOAD/CLEAR、SCAN 和计算器。

板端默认 Python 环境为厂家镜像的 `/usr/local/share/pynq-venv/bin/python3`。主应用入口：

```bash
cd /home/xilinx/jupyter_notebooks/PEFSL_MIX_PLUS/pynq_app
sudo -E /usr/local/share/pynq-venv/bin/python3 pynq_bbox_demo.py --seconds 30
```

首次运行前须完成软件说明中的依赖检查、视频驱动核对以及数字/算符缓存建库；仅运行此命令不能代替环境准备。数字支持集包含名为 `debug_binary_*.png` 的正式输入图片，应一并保留。

## 目录用途

| 路径 | 内容 |
| --- | --- |
| `pynq_app/` | 当前板端应用、交互/分类/扫描模块、支持图片与主机回归测试 |
| `touch_deploy/` | 触摸 IIC 驱动及板端验证程序 |
| `tcu_deploy/` | Tensil PYNQ 驱动、模型元数据、程序、权重及特征验证程序 |
| `ov5640_lcd/` | 相机/LCD Python 支撑代码和示例 |
| `artifacts/` | 当前配套的 bitstream 与 HWH，便于直接部署 |
| `vivado_project/`、`sources/` | 可重建硬件的工程描述、BD、RTL、约束及所需自定义 IP |
| `usb_host_fix/` | 仅适用于匹配厂家启动镜像的可选 USB Host 修复，使用前阅读其 README |
| `PEFSL_C124/` | 可选 PC 对照程序及其原有许可证、模型，不是板端当前 A/B 推理入口 |
| `skill/pc-engineering/`、`skill/pynq-engineering/` | 项目配套的两套工程技能及全部相对引用资料 |
| `conclusion/` | 当前功能、模型、按键及流程说明 |
| `docs/` | 本复现包的环境、部署和硬件构建说明 |
| `reproduction_manifest.json` | 交付文件、大小与 SHA256 清单 |

## 选取与验证范围

包含源码、实际依赖的 IP/约束/BD、模型和必要运行资源。没有收录 Vivado 综合/实现缓存、自动生成的 IP 输出、历史备份、聊天回复、主机日志、Python 缓存、个人学习结果、完整 SD 镜像、无关第三方技能合集和压缩副本。构建过程会重新生成所需输出；预编译 bit/HWH 和模型数据作为明确的部署输入保留。

`tools/verify_reproduction.py` 提供离线完整性验证；硬件路径与 BD 依赖检查、主机测试的实际结果见 `docs/package_validation.md`。本次打包不代表重新完成了实板测试或精度/性能测量。原有功能和验证范围见 [工程总结](conclusion/conclusion.md)。

## 依赖与来源

需要领航者 ZYNQ-7020（`xc7z020clg400-2`）、适配的 OV5640、800×480 RGB 触摸 LCD、厂家匹配 PYNQ 镜像，以及 USB Host 正常枚举的键盘（键盘命名功能）。通用 PYNQ-Z1 镜像不能直接代替这套板级环境。

工程沿用 PEFSL 的预训练骨干/少样本分类路线、Tensil TCU 及相应驱动，包含厂家视频/板级 IP；各组件保留原文件内版权说明及已有许可证。`PEFSL_C124/LICENSE` 仅适用于其对应组件，本仓库未额外声明覆盖所有第三方组件的统一许可证。完整系统镜像、原始训练数据与重新训练/重新编译 TCU 的完整工具链不在本次交付范围；复现当前运行功能使用仓库内匹配的模型文件。
