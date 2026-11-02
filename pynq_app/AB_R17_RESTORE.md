# A/B 恢复到指定 r17，保留键盘及数字计算（r21 / 2026-11-01）

当前 A/B 算法以用户指定目录 `E:\FPGA-3up\PEFSL_MIX_PLUS_digital_10.31\pynq_app` 为参照，恢复该版本的采样、手动分类和扫描流程。

## 当前行为

- A 学习：按旧版裁剪为 32×32，收集 80D TCU 特征和 HSV 统计；保留 r20 键盘名称输入与成功采集后提交名称的机制。
- B 手动框：恢复旧版 `PefslNcmClassifier.predict()`，对学习样本做全局均值居中、类别均值分类，显示名称与百分比分数。扫描余弦门限不影响手动框。
- B 自动扫描：恢复旧版多尺度轮换窗口、居中 Top1/Top5 余弦匹配、可选颜色门、中心/周围颜色差、尺寸限制、每类单框和连续扫描周期确认。`pefsl_scan.py` 与参考版本原本逐字节相同。
- C124 的注册参考图身份门、区域候选、局部回访、按观察确认及结果年龄限制不再参与 A/B 处理。

保留当前 D 模式数字/运算符识别、冻结、表达式存储、UNDO/RESET/OUT，以及 10 数字/9 算符计算容量。r20 键盘大小写字母、数字、空格、下划线、Enter/退格、热插拔和触摸进程恢复继续使用当前代码。

## 部署

在已部署 r20 的板端，仅覆盖 `pynq_app/pefsl_touch_ui.py` 和 `pynq_app/pynq_bbox_demo.py`。其他数字/计算器/键盘文件继续使用当前版本。不要从参考目录整体覆盖当前应用，否则会丢失后续功能。

```bash
cd /home/xilinx/jupyter_notebooks/PEFSL_MIX_PLUS/pynq_app
sudo -E /usr/local/share/pynq-venv/bin/python3 pynq_bbox_demo.py
```

启动版本：`mix-plus-ab-r17-20261101-r21-keyboard-calc10`。

沿用 r17 默认扫描参数：24 窗/周期、窗口间隔 100ms、余弦门限 0.56、可选直方图门限 0（关闭）、中心/周围差门限 0.14、连续 2 周期确认。旧的 `--b-max-result-age` 参数仍能被解析，供既有启动命令兼容；恢复后的 r17 A/B 不使用它。

## 学习文件与验收

代码部署不覆盖板端已有 `pefsl_learned.npz` 或数字/运算符缓存。旧版及新版本的基础特征、标签、框尺寸、颜色统计和键盘名称均可继续加载；C124 可选参考元数据被保留用于文件兼容，但不影响 r17 的 A/B 推理。

建议先 LOAD 已有数据检查 B，再按 A 重新学习同一组物体作效果对比。键盘可输入 `Apple 01_test`，Enter 确认后点类别按钮；SAVE、CLEAR、LOAD 后检查名称恢复。手动 B 的百分比是旧版 NCM 类间分数；scan 的百分比沿用旧版余弦分数显示，不表示经校准的正确率。

另验证 D 模式数字/算符、10 数字计算器、FREEZE/LIVE、UNDO/RESET/OUT 和物理键盘 Ctrl-C。主机回归使用替代摄像头/TCU，并与指定 r17 作为 oracle 比较；实际学习效果需在板端以同场景验证。

历史 C124 B 工作流测试存放在 `historical_tests/test_bmode_c124.py.txt`；当前活动的 A/B 行为测试为 `test_ab_r17_restore.py`，数据兼容测试仍保留在 `test_bmode.py`。
