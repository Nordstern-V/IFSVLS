# 精选复现包验证记录

整理日期：2026-11-02。验证在与原工程分离的精简目录中进行；没有修改原始工程文件。

## 主机验证

- `python -B -m unittest discover -s pynq_app -p 'test_*.py'`：113 项测试通过。覆盖当前计算器、键盘、算符缓存/模式及所选兼容模块。
- `python -B touch_deploy/test_touch_driver.py`：FT/GT 两类触摸协议模拟通过。该文件是直接执行的断言脚本，不能用 unittest 的“0 tests”代替验证。
- `python -B PEFSL_C124/verify_package.py`：可选 PC 包的 27 个文件 SHA256 与原包清单一致。
- 数字支持缓存：10 类、49 样本、80 维、全部有限值；缓存记录的 bitstream 与 tmodel SHA256 与本包一致。
- `skill/`：全部 16 文件纳入；PYNQ SKILL 的 5 个 Markdown 相对引用完整；PC 离线证据 CSV 哈希与说明一致。
- 文件清单逐文件校验 SHA256；Python 文件语法检查、Overlay 同名配对与 tmodel 本地文件引用检查通过。
- 发布文件通过本地链接检查及常见私钥/GitHub token/AWS key 格式扫描；不包含个人学习库、完整系统镜像和源目录中的历史缓存。

主机环境：Python 3.13（Anaconda）、NumPy 2.1.3、OpenCV 4.13.0。主机回归使用模拟硬件，板端仍需软件复现说明中的厂家 PYNQ 环境。

## 硬件与实际运行边界

Vivado 2025.2 已成功打开精选工程并验证原始 BD；3 个自定义 IP 的 14 处文件引用完整，固定 wrapper 的 41 个顶层端口与新生成结果一致。另一个干净短路径验证副本完整执行 `generate_target all`，输出 `HARDWARE_DEPENDENCY_CHECK_PASS`、退出码 0，并生成 IP/HWH。该额外验证副本仅调整了 BD/XCI 的相对生成目录名；发布的 BD/XCI 保持原始字节，构建脚本沿用原始相对输出目录。没有把临时验证副本和生成缓存加入仓库。

硬件使用方法见 [硬件复现说明](hardware_reproduction.md)。本次没有执行完整综合/实现，没有重新测试实板、训练模型或测量识别精度和实时性能。

配套 bitstream/HWH 来自原工程已有交付；重新构建后应检查时序并在目标板上验收。算符 TCU 缓存未在本机伪造或用 PC 特征替代，须按 [软件复现说明](software_reproduction.md) 在匹配板卡上生成。

`reproduction_manifest.json` 记录交付文件的大小和 SHA256，排除自身的哈希。`.gitattributes` 保持文件原始字节，避免 Git 自动转换行尾而导致校验变化。
