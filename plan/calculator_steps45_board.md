# 计算器第 4～5 步：部署与板端验收

> 本文为历史部署记录；当前 r21 的完整复现流程见 [软件复现说明](../docs/software_reproduction.md)。历史版本号与旧升级步骤不代表当前主入口。

后续更新：用户已确认本版功能在板端正常。当前容量已扩为 10 个数字、9 个算符，部署见 [calculator_10_board.md](calculator_10_board.md)。本文保留原 5 数字/4 算符版本的部署和验证记录。

更新日期：2026-10-31。UI 版本：`mix-plus-scan-20261030-r17-calculator-s45`。

适用前提：已完成算符计划第 1～3 步，板端已能识别数字和四个算符，且已有
`digit_support_r17.npz`、`operator_support_r17.npz`。本轮实现表达式存储、计算及两版计算面板，
并修复缓存 evaluate 的 ARM 32 位报错。PC 验证不能代替下面的板端触摸和实拍验收。

## 1. 停止旧 UI，先备份，再传文件

在串口按 Ctrl+C，等程序退出后执行。以下全部为 Bash 命令，可以直接粘贴到串口终端。

```bash
cd /home/xilinx/jupyter_notebooks/PEFSL_MIX_PLUS/pynq_app
stamp=$(date +%Y%m%d_%H%M%S)
backup_dir="backup_calculator_s45/$stamp"
mkdir -p "$backup_dir"
for f in digit_calculator.py pefsl_touch_ui.py digit_recognizer.py test_digit_calculator.py test_calculator_ui.py test_operator_cache.py test_operator_mode.py; do
    if [ -f "$f" ]; then cp -a "$f" "$backup_dir/"; fi
done
echo "Backup: $PWD/$backup_dir"
```

用文件资源管理器从电脑目录 `E:\FPGA-3up\PEFSL_MIX_PLUS\pynq_app\` 复制到板端目录：

```text
/home/xilinx/jupyter_notebooks/PEFSL_MIX_PLUS/pynq_app/
```

| 文件 | 操作 | 用途 |
| --- | --- | --- |
| digit_calculator.py | 新增 | 独立表达式和四则运算模块，启动必须存在 |
| pefsl_touch_ui.py | 覆盖 | 候选确认、存入/撤销/清空/输出、两版面板 |
| digit_recognizer.py | 覆盖 | ARM 32 位 evaluate 修复 |

三个文件都放在上述 `pynq_app` 目录，不能再套一层 `pynq_app`。
可选上传四个测试文件到同一目录：`test_digit_calculator.py`、`test_calculator_ui.py`、
`test_operator_cache.py`、`test_operator_mode.py`。若之前上传过测试文件，本轮要同步覆盖，避免旧布局断言误报。

沿用已部署的 `pynq_bbox_demo.py`、`digit_preprocess.py`、支持图片、两套 NPZ 缓存、
`pefsl_learned.npz`、bit/hwh、TCU 模型和触摸驱动。本轮不传电脑上的缓存覆盖板端缓存，不重新建库。
计算使用 ARM 软件，无须重新综合 FPGA 或生成 bitstream。

## 2. 检查版本和语法

```bash
cd /home/xilinx/jupyter_notebooks/PEFSL_MIX_PLUS/pynq_app
sudo -E /usr/local/share/pynq-venv/bin/python3 -m py_compile digit_calculator.py pefsl_touch_ui.py digit_recognizer.py
echo $?
sudo -E /usr/local/share/pynq-venv/bin/python3 - <<'PY'
from pefsl_touch_ui import APP_VERSION
from digit_calculator import DigitCalculator
print('UI:', APP_VERSION)
c = DigitCalculator()
c.append_number(2)
c.append_operator('plus')
c.append_number(3)
c.append_operator('mul')
c.append_number(4)
c.evaluate()
print(c.display_expression(), '=', c.result_text)
assert c.result_text == '14'
PY
echo $?
```

预期两个退出码均为 `0`，显示：

```text
UI: mix-plus-scan-20261030-r17-calculator-s45
2 + 3 * 4 = 14
```

如果上传了四个测试文件，可继续执行以下软件回归；使用替代 TCU，不启动 Overlay，不修改真实支持集或学习数据：

```bash
sudo -E /usr/local/share/pynq-venv/bin/python3 -m unittest -v test_digit_calculator test_calculator_ui test_operator_cache test_operator_mode
echo $?
```

本轮电脑端通过 40 项测试（计算 9 项、计算交互 11 项、缓存 11 项、识别模式 9 项）。板端预期末尾 `OK`，退出码 `0`。测试输出的 `Stored`、`DIV/0` 等包含正常的边界用例，不是板端识别结果。

## 3. 重跑上次失败的缓存评估

报错来自 `np.repeat`：缓存计数保存为 int64，而 ARMv7 的 NumPy 索引类型 `np.intp` 是 int32，
不允许按 safe 规则自动缩窄。现已先校验计数和总数范围，再转换成平台索引类型。
这个错误不表示算符图片或缓存特征损坏，不需要重新生成缓存。

```bash
set -o pipefail
sudo -E /usr/local/share/pynq-venv/bin/python3 digit_recognizer.py evaluate \
    --class-set operators \
    --cache operator_support_r17.npz \
    --min-score 0.45 \
    --min-margin 0.08 \
    2>&1 | tee operator_eval.txt
echo $?
```

预期完成自匹配和留一验证，打印各自的混淆矩阵，退出码 `0`，不再出现 int64/int32 Traceback。
准确率取决于缓存本身，不要求诊断值为 100%；这仍只是特征缓存诊断，不能代替实拍准确率。
若失败，保留完整 `operator_eval.txt` 便于定位。

## 4. 版本 A：保留二值化预览

```bash
sudo -E /usr/local/share/pynq-venv/bin/python3 pynq_bbox_demo.py \
    --touch-poll-ms 100 --predict-ms 1000 --scan-ms 100 \
    --digit-cache digit_support_r17.npz \
    --operator-cache operator_support_r17.npz \
    --operator-min-score 0.45 --operator-min-margin 0.08 \
    --digit-preview on
```

启动日志应显示本轮 UI 版本。进入 D 后：

- 顶部 `OP MODE` 熄灭，默认数字；点亮后只识别四个算符，再按回数字。
- 右侧保留 160×160 二值化预览；画框区 x0..639、y58..363。
- y364..421 显示表达式、结果、当前候选和提示；底部依次是 `Xn`、`OPn`、`UNDO`、`RESET`、`OUT`。
- 按钮灰暗表示尚不满足条件，存入按钮绿色表示可以确认。READY 仅表示候选稳定，仍须核对是否识别正确。

## 5. 先验证 2+3=5

1. 进入 D，保持 OP MODE 熄灭。摄像头对准数字 2，按 `FREEZE`，画面定格，右上按钮变为 `LIVE`。
2. 框选 2 并松手。等连续 3 次有效预测完成、显示 `NUM 2 READY` 且 `X1` 变绿后，按 `X1`。
   下方表达式应显示 `2`，串口出现 `[ui] calc: Stored X1=2; ...`。
3. 按 `LIVE` 恢复摄像头，对准 `+`；点亮 `OP MODE`，按 `FREEZE`，框选 `+`。
   显示 `OP + READY`、`OP1` 变绿后按 `OP1`，表达式变为 `2 +`。
4. 按 `LIVE`，对准数字 3；熄灭 `OP MODE`，按 `FREEZE`，框选 3。
   显示 `NUM 3 READY`、`X2` 变绿后按 `X2`，表达式变为 `2 + 3`。
5. 按 `OUT`，LCD 应显示 `2 + 3 = 5`，串口出现 `[ui] calc: 2 + 3 = 5`。

每次预测还需要 TCU 执行时间，连续 3 次预测不是固定等待 3 秒；以 READY 和存入按钮状态为准。
同一张冻结画面包含多个字符时，也可保持冻结，逐个重新画框并手动切换 OP MODE。
无需为了每个字符重复解冻；但只能一次框一个字符。

## 6. 撤销、清空和保护行为

| 操作/情况 | 应有行为 |
| --- | --- |
| `Xn` | 存入数字，按 X1、X2…X5 递增；不自动切换 OP MODE |
| `OPn` | 存入算符，按 OP1…OP4 递增；不自动切换 OP MODE |
| `UNDO` | 删除最后一项、清除旧结果和 ROI；保留冻结图与当前类型，重新画框后存入 |
| `RESET` | 清空整个计算表达式、结果和 ROI；保留冻结图与当前类型 |
| `OUT` | 表达式以数字结束即可计算；只录 X1 也可输出 |
| 输入 `8/0` 后 OUT | 显示 `DIV/0` 并保留输入；UNDO 后重框 2、存入、OUT 得到 4 |
| 实时状态、未完成画框、空白/无效 ROI、候选不稳定 | 不允许存入 |
| 数字模式按 OPn、算符模式按 Xn、槽顺序不符 | 不追加错误类型，不自动切换识别类型 |
| 连续双击存入 | 同一候选只存一次；后续需要新 ROI/截图或类型请求 |
| 预测中更换截图、ROI、类型或 A/B/D | 废弃过期候选；已存入表达式保留 |
| LIVE/FREEZE、切换 A/B/D | 保留表达式与结果；重新进入 D 默认数字、实时画面 |
| SAVE/LOAD/CLEAR | 继续管理原 A/B 学习数据，不清空计算表达式；清空计算用 RESET |
| 重启进程 | 计算表达式清空，不写入 `pefsl_learned.npz` |

确认 `2+3*4=14`、`8/2*3=12`、`2-5=-3`、`7/2=3.5`、`1/3=~0.333333`。
再输入 `9-8/2+3*4`，OUT 应得 `17`，达到 5 个数字/4 个算符后不能继续追加。
该版每个 X 保存一位数字，不支持一次存入多位数、括号或小数输入。

## 7. 版本 B：去掉二值化预览

先 Ctrl+C 停止版本 A，退出后执行：

```bash
sudo -E /usr/local/share/pynq-venv/bin/python3 pynq_bbox_demo.py \
    --touch-poll-ms 100 --predict-ms 1000 --scan-ms 100 \
    --digit-cache digit_support_r17.npz \
    --operator-cache operator_support_r17.npz \
    --operator-min-score 0.45 --operator-min-margin 0.08 \
    --digit-preview off
```

画框区扩展到 x0..799、y58..363，底部候选、表达式、按钮完全相同。
重复 `2+3=5`，并在原右侧预览区域画框，确认能识别和存入。
两版都要确认底部面板无法作为 ROI 起点、拖框越界会裁剪。

## 8. 本轮验收记录

按板端实际结果记录：版本打印、evaluate 退出码、两版 `2+3=5`、满容量结果 17、
DIV/0 与撤销纠正、无效/过期/重复候选保护、数字 0～9 回归、A 学习及保存/加载、
B 手动识别与扫描、触摸响应和 Ctrl+C 退出。
验证 CLEAR 时只使用可丢弃的临时学习数据，或先 SAVE，之后 LOAD 恢复。

PC 使用 Python 3.10.21、NumPy 1.26.4、OpenCV 4.10.0 和替代 TCU 验证算术、候选状态、线程切换及两版绘制；
40 项测试全部通过，9 个运行/测试文件通过 Python 3.8 语法兼容检查。ARM 32 位索引通过模拟窄索引测试覆盖，
真实板端 evaluate、LCD、触摸及 TCU 识别效果需以上实测确认。

电脑测试日志位于 `tmp/calculator_steps45_validation/test_results.txt`。两版绘制示意如下，
只用于核对布局，候选由测试替代 TCU 生成，不代表板端识别准确率：

历史验证截图未收录（版本 A：保留预览）。

历史验证截图未收录（版本 B：隐藏预览）。
