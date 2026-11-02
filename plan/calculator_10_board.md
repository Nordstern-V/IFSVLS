# D 计算器扩容：10 个数字、9 个算符

> 本文为历史部署记录；当前 r21 的完整复现流程见 [软件复现说明](../docs/software_reproduction.md)。历史版本号与旧升级步骤不代表当前主入口。

日期：2026-10-31。版本：`mix-plus-bmode-20261031-c124-r18-calc10`。

本轮把容量扩为 X1～X10、OP1～OP9，最多 19 个 token。每个 X 仍为单个 0～9 数字，
按 X1、OP1、X2、…、OP9、X10 交替存入。表达式以数字结束即可 OUT，不必填满。
容量保护和按钮编号共用常量，两版预览布局均显示完整表达式和结果。
沿用原 FREEZE/LIVE、OP MODE、READY、UNDO、RESET 和计算规则，无需重新建库或生成 bitstream。

## 1. 停止程序，备份并检查当前基线

串口先 Ctrl+C 退出，再执行：

```bash
cd /home/xilinx/jupyter_notebooks/PEFSL_MIX_PLUS/pynq_app
grep '^APP_VERSION' pefsl_touch_ui.py
stamp=$(date +%Y%m%d_%H%M%S)
backup_dir="backup_calculator_10/$stamp"
mkdir -p "$backup_dir"
for f in digit_calculator.py pefsl_touch_ui.py pynq_bbox_demo.py pefsl_bmode.py pefsl_identity.py pefsl_proposals.py test_digit_calculator.py test_calculator_ui.py; do
    if [ -f "$f" ]; then cp -a "$f" "$backup_dir/"; fi
done
echo "Backup: $PWD/$backup_dir"
```

当前电脑代码已包含后续 r18 B 模式改进。本轮在这个基线上修改。
若板端版本已是 `mix-plus-bmode-20261031-c124-r18`，只需覆盖下面两个运行文件。
若板端仍是原 r17 计算器版本，须一并同步下面表中列出的 r18 基础文件，以满足当前 UI 的导入依赖。

## 2. 用文件资源管理器上传

电脑源目录：`E:\FPGA-3up\PEFSL_MIX_PLUS\pynq_app\`。

板端目标目录：`/home/xilinx/jupyter_notebooks/PEFSL_MIX_PLUS/pynq_app/`。

| 文件 | 上传条件 |
| --- | --- |
| digit_calculator.py | 必须覆盖：容量 10/9 |
| pefsl_touch_ui.py | 必须覆盖：X10、OP9 和 FULL 状态 |
| pynq_bbox_demo.py | 尚未部署 r18 时一起同步 |
| pefsl_bmode.py | 尚未部署 r18 时一起同步 |
| pefsl_identity.py | 尚未部署 r18 时一起同步 |
| pefsl_proposals.py | 尚未部署 r18 时一起同步 |
| test_digit_calculator.py、test_calculator_ui.py | 可选上传，用于软件回归；已有旧测试时要同步覆盖 |

关于 r18 B 模式本身的变化与旧学习文件兼容，见 [B_MODE_C124.md](../pynq_app/B_MODE_C124.md)。
本轮不覆盖板端 NPZ 缓存或学习文件；只复制明确列出的 Python 文件。

## 3. 上传后检查

```bash
cd /home/xilinx/jupyter_notebooks/PEFSL_MIX_PLUS/pynq_app
sudo -E /usr/local/share/pynq-venv/bin/python3 -m py_compile digit_calculator.py pefsl_touch_ui.py
echo $?
sudo -E /usr/local/share/pynq-venv/bin/python3 - <<'PY'
from digit_calculator import DigitCalculator, MAX_NUMBERS, MAX_OPERATORS
from pefsl_touch_ui import APP_VERSION
print('UI:', APP_VERSION)
print('Capacity:', MAX_NUMBERS, MAX_OPERATORS)
assert (MAX_NUMBERS, MAX_OPERATORS) == (10, 9)
c = DigitCalculator()
for i, number in enumerate((1, 2, 3, 4, 5, 6, 7, 8, 9, 0)):
    if i:
        c.append_operator('plus')
    c.append_number(number)
c.evaluate()
print(c.display_expression(), '=', c.result_text)
assert c.result_text == '45'
assert c.expecting is None
before = (c.tokens, c.result)
for append, value in ((c.append_number, 1), (c.append_operator, 'plus')):
    try:
        append(value)
    except ValueError as error:
        print('Expected rejection:', error)
    else:
        raise AssertionError('capacity limit was not enforced')
assert (c.tokens, c.result) == before
PY
echo $?
```

预期两个退出码均为 0，版本尾部为 `r18-calc10`，容量 `10 9`，完整求和结果 `45`。
若提示 `No module named pefsl_bmode/pefsl_identity/pefsl_proposals`，同步第 2 节的 r18 基础文件后重试。

如果上传了两个测试文件，可执行（替代 TCU，不访问板卡硬件）：

```bash
sudo -E /usr/local/share/pynq-venv/bin/python3 -m unittest -v test_digit_calculator test_calculator_ui
```

## 4. 启动两版界面并验收

```bash
sudo -E /usr/local/share/pynq-venv/bin/python3 pynq_bbox_demo.py \
    --touch-poll-ms 100 --predict-ms 1000 --scan-ms 100 \
    --digit-cache digit_support_r17.npz \
    --operator-cache operator_support_r17.npz \
    --digit-preview on
```

保留之前已验证的识别门限参数；默认算符门限仍为 0.45/0.08。
去掉预览版：Ctrl+C 退出，把最后一行改为 `--digit-preview off` 再启动。

1. 用原截图、画框和存入流程输入 `1+2+3+4+5+6+7+8+9+0`。
2. 存完第 5 个数字后应仍能继续；最后两项应显示 OP9、X10。
3. 按 OUT，结果为 `45`。已存满时显示 X FULL、OP FULL，继续点击不追加。
4. UNDO 删去最后的 0，保持当前类型，重新框选并存入 1，再 OUT，结果为 `46`。
5. RESET 后输入十个 9 连乘，结果应为 `3486784401`。核对完整表达式和结果均在屏内。
6. 两版布局分别验证上述流程，并抽查 `2+3*4=14`、除零提示、切换 OP MODE、LIVE/FREEZE、A/B/D。

容量扩展的 PC 软件验证与绘制检查使用替代 TCU，不代替新增槽位的板端触摸验收。

## PC 验证记录

Python 3.10.21、NumPy 1.26.4、OpenCV 4.10.0 环境下，当前项目 74 项测试全部通过，
其中计算模块 11 项、计算交互 13 项；本轮修改的 4 个运行/测试文件通过 Python 3.8 语法检查。
覆盖旧 5 数字上限可继续输入、10/9 满容量、超限拒绝、撤销补回、完整结果绘制及原识别模式回归。

完整测试命令：

```powershell
& 'D:\Python\Anaconda3\envs\pefsl\python.exe' -m unittest discover -s pynq_app -p 'test_*.py' -v
```

日志：`tmp/calculator_10_validation/test_results.txt`。以下为替代 TCU 的布局验证图：

历史验证截图未收录（10 数字：保留二值化预览）。

历史验证截图未收录（10 数字：隐藏二值化预览）。
