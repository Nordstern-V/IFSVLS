# MIX_PLUS PYNQ 触摸驱动

`touch_driver.py` 对应当前 `pefsl_mix_plus_touch.bit/.hwh`。它使用 HWH 中的三个 IP：

- `touch_iic`：AXI IIC，基地址 `0x41610000`
- `touch_int_gpio`：1 位输入，基地址 `0x41620000`
- `touch_rst_gpio`：1 位输出，基地址 `0x41630000`

驱动按 `sources/touch_reference/touch_dri.v` 实现 FT/GT 检测、FT 配置、状态轮询、坐标读取、800×480 映射和 `down/move/up` 消抖事件。参考 RTL 没有实际采样 `touch_int`，所以默认通过状态寄存器轮询；GPIO INT 会作为诊断值输出。

板上验证：

```bash
cd /home/xilinx/jupyter_notebooks/PEFSL_MIX_PLUS/touch_deploy
sudo -E /usr/local/share/pynq-venv/bin/python3 touch_iic_validate.py \
  --bitstream ../artifacts/pefsl_mix_plus_touch.bit \
  --seconds 30
```

触摸屏无操作时，验证脚本仍应完成 Overlay 下载、复位释放、芯片检测和状态轮询；触摸屏上按下、拖动、抬起时，应在终端看到对应事件，并在 `validation_result/touch_iic_report.json` 中记录。

主机端无需 PYNQ 即可运行协议测试：

```bash
python test_touch_driver.py
```


## 兼容性

旧版 PYNQ 的 `AxiGPIO.channel1.write` 需要 `(value, mask)` 两个参数，驱动会自动兼容这个签名。AXI IIC 的 IP 版本为 2.1，而部分 PYNQ 镜像只注册 2.0 驱动；验证脚本使用 `ignore_version=True` 加载兼容驱动。

当前版本 r10 使用 FT/GT 要求的重复起始读事务：FT 触摸状态和首点坐标从 `0x02` 连续读取，单次触摸采样只产生一个 AXI-IIC 读事务，减少长时间拖动时的总线压力；完成 `receive()` 后再等待总线释放。若 PYNQ 2.7 的 AXI-IIC 事务失败，驱动向 AXI-IIC `SOFTR`（偏移 `0x40`）写入 `0xA` 清除 busy 状态，并在同一个轮询周期内重试一次，不需要把触摸轮询周期调大。bbox 应用默认以 100 ms 轮询；INT 高电平稳定 250 ms 后用本地状态生成 `up`，避免下一次点击被误判为 `move`。更新 `touch_driver.py` 和 `pynq_bbox_demo.py` 后必须重启应用；若上一次进程异常退出导致总线仍 busy，重新下载 bitstream 或重启 PYNQ 后再运行验证脚本。





