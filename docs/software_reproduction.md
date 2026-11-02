# 软件复现：PYNQ 板端与可选 PC 对照

## 当前工程与环境

主工程是 `pynq_app/pynq_bbox_demo.py`，应用版本为
`mix-plus-ab-r17-20261101-r21-keyboard-calc10`：A/B 使用恢复后的 r17
学习与分类流程，保留 r20 USB 键盘命名、D 数字/算符识别和最多 10 个数字、9 个算符的计算器。
`PEFSL_C124/` 是独立、可选的 Windows PC 对照程序；当前板端不导入它，也不使用其 C124 身份门。

板端前提是与本工程相同的正点原子 ZYNQ-7020 开发板、OV5640 摄像头和
800×480 LCD（ID `0x7084`），使用厂家 PYNQ 2.7 镜像及其 ARM Python 环境。
本地部署文档使用 Python 3.8+ 与 `/usr/local/share/pynq-venv/bin/python3`。
镜像需提供 `pynq`、`numpy`、`cv2` 和 `cffi`；USB 键盘直接使用 Linux evdev，
无需安装名为 evdev 的 Python 包。不要将 `PEFSL_C124/requirements.txt` 安装到板端。
不同板卡、普通 PYNQ-Z1/Z2 镜像或不同内核不属于已验证的可直接替换环境。

仓库不包含完整 SD 镜像、厂家工具安装包或系统 Python 环境。接收者需要自行取得匹配的厂家
`ZYNQ-7020-2.7.0.img`；板端文件路径由代码相对 `__file__` 计算，克隆目录可命名为 `IFSVLS`。

## 从干净 checkout 部署

将整个精选仓库复制或克隆到板端，例如
`/home/xilinx/jupyter_notebooks/IFSVLS`，保留仓库内相对目录。下面命令从仓库根执行：

```bash
cd /home/xilinx/jupyter_notebooks/IFSVLS
PYTHON=/usr/local/share/pynq-venv/bin/python3
"$PYTHON" -c 'import pynq, numpy, cv2, cffi; print("Python dependencies available")'
"$PYTHON" -c 'import pynq.lib.ov5640, pynq.lib.video2lcd; print("Vendor camera/LCD drivers available")'
```

本仓库附有 `ov5640_lcd/ov5640/` 与 `ov5640_lcd/video2lcd/` 的源码副本。
厂家镜像已提供这两个包时直接使用原有安装；只有缺少包时才从副本安装。
下列命令只补充缺少的目录，不覆盖现有驱动：

```bash
sudo "$PYTHON" - <<'PY'
from pathlib import Path
import shutil
import pynq.lib
repository = Path.cwd()
target = Path(pynq.lib.__file__).resolve().parent
for name in ("ov5640", "video2lcd"):
    source = repository / "ov5640_lcd" / name
    destination = target / name
    if destination.exists():
        print("Keep existing:", destination)
    else:
        shutil.copytree(str(source), str(destination))
        print("Installed:", destination)
PY
"$PYTHON" -c 'import pynq.lib.ov5640, pynq.lib.video2lcd; print("Vendor drivers available")'
```

当前 Zynq-7020 摄像头/LCD 路径使用附带的 Python 驱动与 PYNQ 底层驱动；
`video2lcd/drm.py` 中的 `libdisplayport.so` 仅用于其他 DisplayPort 使用路径，
本工程不提供该其他硬件路径的编译库。

## 必需运行数据

| 文件或数据 | 用途 | 原始字节数 |
| --- | --- | ---: |
| `artifacts/pefsl_mix_plus_touch.bit` 与同名 `.hwh` | 当前摄像头/LCD、触摸、TCU Overlay，必须成对使用 | 见仓库文件清单 |
| `tcu_deploy/resnet9_strided_16fmaps_onnx_custom_perf.tmodel` | Tensil 模型/架构及数据文件描述 | 638 |
| 同名前缀 `.tprog` | TCU 指令 | 798480 |
| 同名前缀 `.tdata` | TCU 常量 | 468912 |
| `pynq_app/digit_support/0..9/*.png` | 49 张数字支持集输入 | 小型 PNG |
| `pynq_app/operator_support/{plus,sub,mul,div}/*.png` | 20 张算符支持集输入 | 小型 PNG |
| `pynq_app/digit_support_r17.npz` | 已生成的 49 样本/80D 数字缓存 | 11822 |

`debug_binary_*.png` 在这两个支持集目录内是正式模型输入，不是可删除的调试输出。
随附数字缓存记录的 bitstream 和 tmodel SHA256 与仓库中的现有文件一致；
它使用旧的 v2 数字缓存元数据格式，现有程序能够读取。
本地工程没有 `operator_support_r17.npz`，首次复现需要在板端生成。
重新生成 Overlay、修改模型或修改预处理/支持集后，应在板端重建对应缓存。

先停止正在运行的 bbox 应用，逐项进行硬件烟测；各命令会下载 Overlay：

```bash
sudo -E "$PYTHON" tcu_deploy/tcu_feature_validate.py \
  --bitstream artifacts/pefsl_mix_plus_touch.bit \
  --tmodel tcu_deploy/resnet9_strided_16fmaps_onnx_custom_perf.tmodel --repeats 2
sudo -E "$PYTHON" touch_deploy/touch_iic_validate.py \
  --bitstream artifacts/pefsl_mix_plus_touch.bit --seconds 30
```

TCU 预期显示 `raw84=84`、`feature80=80` 和 `TCU bbox feature validation PASS`。
之后生成算符缓存；数字缓存可先直接 inspect，需要重建时再运行 digits 的 build：

```bash
sudo -E "$PYTHON" pynq_app/digit_recognizer.py build --class-set operators
"$PYTHON" pynq_app/digit_recognizer.py inspect --class-set operators
"$PYTHON" pynq_app/digit_recognizer.py inspect --class-set digits
# 修改了模型/支持集/预处理，或希望完整重新生成数字缓存时：
sudo -E "$PYTHON" pynq_app/digit_recognizer.py build --class-set digits
```

数字预期为 `10 classes / 49 images / 80D`，计数 `[5,5,5,5,5,4,5,5,5,5]`；
算符预期为 4 类、每类 5 张，共 20 张。生成的缓存是板端运行输出；不需要 PC 权重来建库。

## USB 键盘前置与精确镜像修复

键盘命名需要 USB Host/PHY 正常枚举及 `/dev/input/event*` 输入节点。
如果镜像已经能够使用键盘，不需要修改启动文件。
`usb_host_fix/` 保留了原项目中修复厂家镜像缺失 USB0 Host/PHY 设备树属性的部署输入；
它不是通用内核更新，也不应直接用于其他镜像。

| 文件 | 用途 | 字节数 |
| --- | --- | ---: |
| `IMAGE_USB_HOST.ub` | 保持原内核 payload、仅修正设备树及 FIT 哈希的启动镜像 | 5892776 |
| `candidate_usb_host.dtb` | 对应设备树，供核验 | 19646 |
| `apply_usb_host_fix.py`、`manifest.json`、`README.md` | SHA256 检查、备份、安装和回退 | 小型文本 |

适用条件：正点原子 `ZYNQ-7020-2.7.0.img`，实际启动的原始 `image.ub` SHA256 为
`b502ca462cb57a43834c5cd0756ac38c6a28d6760cf27e942ea97dacd36c91f1`，
或已经是本修复的 `b72b9ee15590152d88239b17887ba5aa0b8699e4fe53cbebcac406beea4ff3aa`。
先确认 SD 卡启动分区挂载位置与板卡 P4 Host 接法（3-5、4-6），再按
[`usb_host_fix/README.md`](../usb_host_fix/README.md) 执行只读 `--check`。
只有完全匹配且确实需要修复时才执行 `--apply`；保留脚本建立的启动备份用于回退，
不要绕过哈希校验或直接将候选文件盲写到启动分区。

原 USB 包的 README/manifest 保留了候选生成时的“尚未实板验证”记录；
后续 [`pynq_app/KEYBOARD.md`](../pynq_app/KEYBOARD.md) 记录 USB Host/PHY、HUB
和键盘枚举已在原开发板通过用户验证。这不证明其他板卡或镜像同样适用。
没有附带完整 SD 镜像、原机恢复档案或用户启动分区备份。

## 启动与操作验收

```bash
# 可先独立验证 USB 键盘；退出验证器后再运行应用
sudo -E "$PYTHON" pynq_app/keyboard_validate.py --seconds 30
sudo -E "$PYTHON" pynq_app/pynq_bbox_demo.py --seconds 30
# 持续运行
sudo -E "$PYTHON" pynq_app/pynq_bbox_demo.py
```

没有键盘时可传 `--no-keyboard`，A/B 与触摸数字计算仍可使用。
确认 LCD 活动图像、触摸拖框与按钮响应：A 框选学习，输入名称并 Enter 后选择类别；
B 手动/SCAN 识别；SAVE/LOAD 保存和恢复学习；D 数字与 OP MODE 算符识别，
FREEZE/LIVE、10 数字表达式、UNDO/RESET/OUT。键盘扩展字符可用 `Apple 01_test` 验证。
个人学习库 `pefsl_learned.npz` 未随仓库上传，首次 A/B 需要自己采样。
应用产生的学习文件、验证报告及其他运行输出不属于精选源码。

## 主机回归与可选 PC 包

主机回归需要 NumPy 和 OpenCV；不需要实际 PYNQ 硬件。仓库保留了当前无外部历史源码依赖的
单元测试。此前 `test_ab_r17_restore.py` 依赖本机另一份指定历史 r17 工程作逐字节对照，
未纳入精选包；不影响板端运行，当前 r21 行为说明保留在 `pynq_app/AB_R17_RESTORE.md`。

```bash
cd pynq_app
python -m unittest -v test_digit_calculator test_calculator_ui \
  test_operator_cache test_operator_mode test_usb_keyboard \
  test_keyboard_ui test_keyboard_dispatch
# 兼容数据及保留的辅助模块；不代表当前 A/B 在启用 C124 身份门
python -m unittest -v test_bmode test_b_identity test_b_proposals
cd ..
python touch_deploy/test_touch_driver.py
```

精选准备时，前一组 94 项测试、兼容模块 19 项测试全部通过；触摸脚本的 FT/GT 两项协议检查通过。
验证主机为 Python 3.13.5、NumPy 2.1.3、OpenCV 4.13.0。
触摸脚本使用普通函数并在 `__main__` 中调用，需直接执行；`unittest discover` 不会运行其中的检查。
这类测试使用替代的图像/TCU 或协议模拟，不能代替板卡启动、时序、实时图像及识别效果验收。
完整硬件重建步骤见仓库的硬件复现说明。

可选 Windows PC 对照位于 `PEFSL_C124/`，包含全部源码、MIT LICENSE、原始模型权重
`resnet9_strided_16fmaps.pt`（782219 字节）、版本/哈希清单和安装启动脚本。
按照其 `README_先读我.md` 使用 64 位 Python 3.12、运行 `install.bat` 与 `start.bat`。
该包记录的依赖为 torch 2.6.0+cpu、NumPy 2.5.3、opencv-contrib-python 5.0.0.93、Pillow 12.3.0；
保留原有依赖清单，本次精选未重新联网安装或验证这些 Windows 依赖。
板端不会载入 `.pt` 权重；PC 模型的特征缓存不能代替 TCU 支持集缓存。

## 保留的第三方说明

`ov5640_lcd/` 源码保留 Alientek/Xilinx 原作者及 BSD 版权声明；
`tcu_deploy/tcu_pynq/` 保留 Tensil Apache-2.0 头与许可证；
`PEFSL_C124/LICENSE` 保留原 PEFSL 代码许可证。
没有给不同来源的厂家 IP、系统镜像或运行环境统一重新授权。
