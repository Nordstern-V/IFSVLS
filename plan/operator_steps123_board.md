# 算符计划第 1～3 步：部署与板端验证

后续更新（2026-10-31）：用户已确认建库及算符实拍成功。本文保留第 1～3 步部署记录；当前计算器版本和 ARM evaluate 修复请按 [calculator_steps45_board.md](calculator_steps45_board.md) 更新，无需重复建库。

日期：2026-10-30。电脑端版本：`mix-plus-scan-20261030-r17-operator-s3`。

已完成：算符素材核对和 manifest、预处理有效标志、独立算符缓存建库/加载/自匹配/留一验证、D 数字/算符手动切换、两种预览布局。X/OP 存储、表达式和输出计算属于后续第 4～6 步，本版尚未提供。

## 电脑端验证记录

- `operator_support/` 四类各 5 张，共 20 张，32×32；与项目 `support/` 对应图片逐个 SHA-256 一致。
- 20 张支持图通过实际 NumPy/OpenCV 预处理验证，15 张原本二值图逐像素保持，减号横线、除号上下点完整；素材本身的断笔/杂线保留，识别效果待实拍验证。
- 原数字 49 张支持图的预处理输出与改动前逐像素一致；一组有效实时 ROI 连续预处理输出也与基线一致。
- 18 项缓存/分类切换/worker/冻结/预览测试通过；运行文件通过 Python 3.8 语法兼容检查。
- 使用电脑现有 Python 3.10.21、NumPy 1.26.4、OpenCV 4.10.0 环境。测试用替代 TCU 提取器验证软件流程，不代表板端识别准确率；未在电脑上生成部署用算符特征缓存。

## 1. 上传前，在串口停止旧程序并备份

若正在运行摄像头程序，先按 Ctrl+C，等退出回到 `xilinx@pynq:...$` 后操作。建库时不要同时运行 UI 或其他占用 TCU 的进程。

下面是一整段 Bash 命令，直接粘贴进串口终端；必须先备份，再通过文件资源管理器覆盖文件。

```bash
cd /home/xilinx/jupyter_notebooks/PEFSL_MIX_PLUS/pynq_app
stamp=$(date +%Y%m%d_%H%M%S)
backup_dir="backup_operator_s3/$stamp"
mkdir -p "$backup_dir"
for f in digit_preprocess.py digit_recognizer.py pefsl_touch_ui.py pynq_bbox_demo.py check_operator_support.py operator_support_manifest.json digit_support_r17.npz operator_support_r17.npz pefsl_learned.npz; do
    if [ -f "$f" ]; then cp -a "$f" "$backup_dir/"; fi
done
if [ -d operator_support ]; then cp -a operator_support "$backup_dir/"; fi
echo "Backup: $PWD/$backup_dir"
```

## 2. 用文件资源管理器复制以下项目

电脑源目录：`E:\FPGA-3up\PEFSL_MIX_PLUS\pynq_app\`。

板端目标目录：`/home/xilinx/jupyter_notebooks/PEFSL_MIX_PLUS/pynq_app/`。

| 电脑源目录下的项目 | 板端目标目录下的位置 | 操作 |
| --- | --- | --- |
| digit_preprocess.py | digit_preprocess.py | 覆盖 |
| digit_recognizer.py | digit_recognizer.py | 覆盖 |
| pefsl_touch_ui.py | pefsl_touch_ui.py | 覆盖 |
| pynq_bbox_demo.py | pynq_bbox_demo.py | 覆盖 |
| check_operator_support.py | check_operator_support.py | 新增/覆盖 |
| operator_support_manifest.json | operator_support_manifest.json | 新增/覆盖 |
| operator_support 文件夹 | operator_support/plus、sub、mul、div | 复制整个目录 |

若板端已有 `operator_support` 且内容不确定，先将其重命名为其他名称再复制本次目录，避免新旧图片混在同一类别。不要复制成 `operator_support/operator_support/plus`。诊断脚本会检查 manifest 与图片是否对应。

本轮沿用板端已经验证过的 `digit_support_r17.npz`、`pefsl_learned.npz`、`pefsl_scan.py`、bit/hwh、TCU 模型和触摸驱动。电脑没有新的真实 TCU 缓存可上传；算符缓存由下一步在板端生成。

可选上传 `test_operator_cache.py`、`test_operator_mode.py` 到同一目录，供板端运行软件测试；这两项不是启动必需文件。

## 3. 上传后检查版本、环境及素材

```bash
cd /home/xilinx/jupyter_notebooks/PEFSL_MIX_PLUS/pynq_app
sudo -E /usr/local/share/pynq-venv/bin/python3 -m py_compile digit_preprocess.py digit_recognizer.py pefsl_touch_ui.py pynq_bbox_demo.py check_operator_support.py
echo $?
sudo -E /usr/local/share/pynq-venv/bin/python3 - <<'PY'
import sys
import cv2
import numpy as np
from pefsl_touch_ui import APP_VERSION
from digit_preprocess import PREPROCESS_VERSION, preprocessing_backend
print('Python:', sys.executable)
print('NumPy:', np.__version__, 'OpenCV:', cv2.__version__)
print('UI:', APP_VERSION)
print('Preprocess:', PREPROCESS_VERSION, preprocessing_backend())
PY
sudo -E /usr/local/share/pynq-venv/bin/python3 check_operator_support.py --support-dir operator_support --output-dir operator_support_check
echo $?
```

预期：两个退出码均为 0；UI 为 `mix-plus-scan-20261030-r17-operator-s3`；预处理版本为 `r17-symbol-preprocess-v1`，后端 opencv；素材检查末尾显示：

```text
PASS: 4 classes / 20 images; manifest hashes and preprocessing valid
```

检查脚本会在板端生成：

```text
/home/xilinx/jupyter_notebooks/PEFSL_MIX_PLUS/pynq_app/operator_support_check/
    operator_preprocess_contact_sheet.png
    operator_preprocess_report.json
    plus/ sub/ mul/ div/ 下的预处理图
```

可用文件资源管理器将拼图取回查看。左右分别是源图和实际送入支持特征提取的预处理图；确认减号横线、除号两点仍在。若这一步报错，先处理对应文件/环境问题，不继续建库。

## 4. 先建四张冒烟缓存，再建二十张完整缓存

```bash
sudo -E /usr/local/share/pynq-venv/bin/python3 digit_recognizer.py build --class-set operators --support-dir operator_support --limit-per-class 1 --output operator_support_smoke.npz
echo $?
```

预期末尾：`4 classes / 4 images / 80D`，退出码 0。通过后执行：

```bash
sudo -E /usr/local/share/pynq-venv/bin/python3 digit_recognizer.py build --class-set operators --support-dir operator_support --output operator_support_r17.npz
echo $?
sudo -E /usr/local/share/pynq-venv/bin/python3 digit_recognizer.py inspect --class-set operators --cache operator_support_r17.npz
sudo -E /usr/local/share/pynq-venv/bin/python3 digit_recognizer.py inspect --class-set digits --cache digit_support_r17.npz
```

算符完整缓存预期：

```text
schema: operator_support_r17_v1
kind: operators
4 classes / 20 images / 80D
labels: ['plus', 'sub', 'mul', 'div']
counts: [5, 5, 5, 5]
finite features: True
```

数字缓存应为既有 v2、10 类/49 张/80D。如果数字缓存缺失或仍为 v1，应先按原数字建库流程处理；不能将算符缓存充当数字缓存。界面加载的是完整算符缓存，不是 smoke 缓存。

## 5. 保存自匹配和留一验证结果

```bash
set -o pipefail
sudo -E /usr/local/share/pynq-venv/bin/python3 digit_recognizer.py evaluate --class-set operators --cache operator_support_r17.npz --min-score 0.45 --min-margin 0.08 2>&1 | tee operator_eval.txt
echo $?
```

默认同时输出自匹配、留一验证、混淆矩阵以及每类正确接收/错误接收/拒绝数量。留一时会重新计算剩余训练样本的均值和原型。只有完整缓存适合留一验证，smoke 每类一张会明确报错。

0.45/0.08 是本轮提供的试验起点，并非已经标定的板端门限。命令退出码 0 仅代表评估执行成功，不代表识别达标；请保留报告用于判断类内区分情况。该评估也不能代替摄像头实拍测试。

## 6. 启动 D 模式实拍：两个 UI 版本任选其一

版本 A，保留右侧二值化预览：

```bash
sudo -E /usr/local/share/pynq-venv/bin/python3 pynq_bbox_demo.py --touch-poll-ms 100 --predict-ms 1000 --scan-ms 100 --digit-cache digit_support_r17.npz --operator-cache operator_support_r17.npz --operator-min-score 0.45 --operator-min-margin 0.08 --operator-history 5 --digit-preview on
```

按 Ctrl+C 退出后，可换版本 B，去掉预览：

```bash
sudo -E /usr/local/share/pynq-venv/bin/python3 pynq_bbox_demo.py --touch-poll-ms 100 --predict-ms 1000 --scan-ms 100 --digit-cache digit_support_r17.npz --operator-cache operator_support_r17.npz --operator-min-score 0.45 --operator-min-margin 0.08 --operator-history 5 --digit-preview off
```

启动输出应包含 UI 版本、Digits: 10 classes / 49 images / 80D、Operators: 4 classes / 20 images / 80D。

本阶段未加计算面板，底部仍为原 58 像素状态栏；预览版画框区 x0..639、y58..421，无预览版 x0..799、y58..421。第 6 步正式加入计算面板时再按完整计划调整底部布局。

## 7. 实拍验收顺序与需回传的内容

1. 进入 D，OP MODE 默认灰色（熄灭），只进行 0～9 数字分类。先验证原数字识别仍可用。
2. 点亮 OP MODE，绿色表示只进行四类算符分类。逐个将 +、-、×、÷ 放入镜头。
3. 每次 FREEZE 后框选一个算符并保留适当白边，观察右侧预处理及结果；LCD 用 +、-、*、/ 表示四算符。低置信结果为 ?。
4. 切换 OP MODE 时冻结画面和完成的 ROI 保留，旧候选清除，重新识别。再次点击后回数字模式；FREEZE/LIVE 不自动切换类型。
5. 测试空白画面和重画 ROI，确认旧结果清除。模式只限制分类范围，不能保证把错误类型的图像自动拒绝。
6. 切 A/B 检查原学习/识别/扫描；再进 D 应回到熄灯数字模式。
7. 可分别记录每种算符多次独立截图的正确、错误、? 数量和具体混淆，再决定是否调整算符门限或补支持图。

建议回传：`operator_eval.txt`、建库末尾输出、每种算符实际识别结果；如出现预处理问题，再取回 `operator_support_check/operator_preprocess_contact_sheet.png`。看到 ? 不应直接把门限降至零，应先核对算符笔画与支持图是否相近。
