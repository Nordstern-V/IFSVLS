__author__ = "CX"
__copyright__ = "Copyright 2020, alientek"
__email__ = "2568365021@qq.com"

from .ov5640_config import *
from .ov5640_af_config import OV5640_AF_Config
from time import sleep


class OV5640:
    """
    OV5640类，用于OV5640摄像头对象，提供使用OV5640相关的方法
    """
    def __init__(self, addr, iic):
        """
        初始化OV5640类需要提供两个参数：

        addr: OV5640器件地址
        iic : OV5640的iic实例
        """
        self.addr = addr
        self.iic = iic

    def wr_reg(self, reg, value, length=3):
        """
        OV5640写寄存器函数

        reg：  寄存器地址，
        value：寄存器值，
        length:寄存器地址和值的字节数
        """
        reg_hgh = reg >> 8
        reg_low = reg & 0xff
        self.iic.send(self.addr, bytes([reg_hgh, reg_low, value]), length)

    def init(self):
        i = 0

        for config in ov5640_init_cfg:
            if i == 1:
                sleep(0.1)

            self.wr_reg(config[0], config[1])
            i += 1

        self.flash_ctrl(1)
        self.flash_ctrl(0)

    def focus_init(self):
        """
        初始化自动对焦
        """
        addr = 0x8000
        # reset MCU
        self.wr_reg(0x3000, 0x20)

        # 下载固件
        for config in OV5640_AF_Config:
            self.wr_reg(addr, config)
            addr += 1

        self.wr_reg(0x3022, 0x00)
        self.wr_reg(0x3023, 0x00)
        self.wr_reg(0x3024, 0x00)
        self.wr_reg(0x3025, 0x00)
        self.wr_reg(0x3026, 0x00)
        self.wr_reg(0x3027, 0x00)
        self.wr_reg(0x3028, 0x00)
        self.wr_reg(0x3029, 0x7f)
        self.wr_reg(0x3000, 0x00)

    def focus_continue(self):
        """
        持续自动对焦,当失焦后,会自动继续对焦
        """
        self.wr_reg(0x3023, 0x01)
        self.wr_reg(0x3022, 0x08)  # 发送IDLE指令

        self.wr_reg(0x3023, 0x01)
        self.wr_reg(0x3022, 0x04)  # 发送持续对焦指令

    def focus_single(self):
        """
        执行一次自动对焦
        """
        self.wr_reg(0x3022, 0x03)  # 触发一次自动对焦

    def flash_ctrl(self, mode):
        """
        闪光灯控制
        mode:0,关闭
             1,打开
        """
        self.wr_reg(0x3016, 0x02)
        self.wr_reg(0x301c, 0x02)
        if mode:
            self.wr_reg(0x3019, 0x02)  # 打开闪光灯
        else:
            self.wr_reg(0x3019, 0x00)  # 关闭闪光灯

    def test_pattern(self, mode):
        """
        # 测试序列
        # mode:0,关闭
        #     1,彩条
        #     2,色块
        """
        if mode == 0:
            self.wr_reg(0X503D, 0X00)
        elif mode == 1:
            self.wr_reg(0X503D, 0X80)
        elif mode == 2:
            self.wr_reg(0X503D, 0X82)

    def exposure(self, exposure):
        """
        EV曝光补偿
        exposure:0~6,代表补偿-3~3.
        """
        self.wr_reg(0x3212, 0x03)  # start group 3
        self.wr_reg(0x3a0f, OV5640_EXPOSURE_TBL[exposure][0])
        self.wr_reg(0x3a10, OV5640_EXPOSURE_TBL[exposure][1])
        self.wr_reg(0x3a1b, OV5640_EXPOSURE_TBL[exposure][2])
        self.wr_reg(0x3a1e, OV5640_EXPOSURE_TBL[exposure][3])
        self.wr_reg(0x3a11, OV5640_EXPOSURE_TBL[exposure][4])
        self.wr_reg(0x3a1f, OV5640_EXPOSURE_TBL[exposure][5])
        self.wr_reg(0x3212, 0x13)  # end group 3
        self.wr_reg(0x3212, 0xa3)  # launch group 3

    def light_mode(self, mode):
        """
        # 白平衡设置
        # 0:自动
        # 1:日光sunny
        # 2,办公室office
        # 3,阴天cloudy
        # 4,家里home
        """
        self.wr_reg(0x3212, 0x03)  # start group 3
        for i in range(7):
            self.wr_reg(0x3400+i, OV5640_LIGHTMODE_TBL[mode][i])  # 设置饱和度
        self.wr_reg(0x3212, 0x13)  # end group 3
        self.wr_reg(0x3212, 0xa3)  # launch group 3

    def color_saturation(self, sat):
        """
        # 色度设置
        # sat:0~6,代表饱和度-3~3.
        """
        self.wr_reg(0x3212, 0x03)  # start group 3
        self.wr_reg(0x5381, 0x1c)
        self.wr_reg(0x5382, 0x5a)
        self.wr_reg(0x5383, 0x06)
        for i in range(6):
            self.wr_reg(0x5384+i, OV5640_SATURATION_TBL[sat][i])  # 设置饱和度
        self.wr_reg(0x538b, 0x98)
        self.wr_reg(0x538a, 0x01)
        self.wr_reg(0x3212, 0x13)  # end group 3
        self.wr_reg(0x3212, 0xa3)  # launch group 3

    def brightness(self, bright):
        """
        # 亮度设置
        # bright:0~8,代表亮度-4~4.
        """
        if bright < 4:
            brtval = 4-bright
        else:
            brtval = bright-4
        self.wr_reg(0x3212, 0x03)  # start group 3
        self.wr_reg(0x5587, brtval << 4)
        if bright < 4:
            self.wr_reg(0x5588, 0x09)
        else:
            self.wr_reg(0x5588, 0x01)
        self.wr_reg(0x3212, 0x13)  # end group 3
        self.wr_reg(0x3212, 0xa3)  # launch group 3

    def contrast(self, contrast):
        """
        # 对比度设置
        # contrast:0~6,代表亮度-3~3.
        """

        reg0val = 0X00  # contrast=3,默认对比度
        reg1val = 0X20
        if contrast == 0:  # -3
            reg1val = reg0val = 0X14
        elif contrast == 1:  # -2
            reg1val = reg0val = 0X18
        elif contrast == 2:  # -1
            reg1val = reg0val = 0X1C
        elif contrast == 4:  # 1
            reg0val = 0X10
            reg1val = 0X24
        elif contrast == 5:  # 2
            reg0val = 0X18
            reg1val = 0X28
        elif contrast == 6:  # 3
            reg0val = 0X1C
            reg1val = 0X2C

        self.wr_reg(0x3212, 0x03)  # start group 3
        self.wr_reg(0x5585, reg0val)
        self.wr_reg(0x5586, reg1val)
        self.wr_reg(0x3212, 0x13)  # end group 3
        self.wr_reg(0x3212, 0xa3)  # launch group 3

    def sharpness(self, sharp):
        """
        # 锐度设置
        # sharp:0~33,0,关闭33,auto其他值,锐度范围.
        """
        if sharp < 33:  # 设置锐度值
            self.wr_reg(0x5308, 0x65)
            self.wr_reg(0x5302, sharp)
        else:  # 自动锐度
            self.wr_reg(0x5308, 0x25)
            self.wr_reg(0x5300, 0x08)
            self.wr_reg(0x5301, 0x30)
            self.wr_reg(0x5302, 0x10)
            self.wr_reg(0x5303, 0x00)
            self.wr_reg(0x5309, 0x08)
            self.wr_reg(0x530a, 0x30)
            self.wr_reg(0x530b, 0x04)
            self.wr_reg(0x530c, 0x06)

    def special_effects(self, eft):
        """
         # 特效设置
         # 0:正常
         # 1,冷色
         # 2,暖色
         # 3,黑白
         # 4,偏黄
         # 5,反色
         # 6,偏绿
        """
        self.wr_reg(0x3212, 0x03)  # start group 3
        self.wr_reg(0x5580, OV5640_EFFECTS_TBL[eft][0])
        self.wr_reg(0x5583, OV5640_EFFECTS_TBL[eft][1])  # sat U
        self.wr_reg(0x5584, OV5640_EFFECTS_TBL[eft][2])  # sat V
        self.wr_reg(0x5003, 0x08)
        self.wr_reg(0x3212, 0x13)  # end group 3
        self.wr_reg(0x3212, 0xa3)  # launch group 3

    def set_outsize(self, h_pixel, v_pixel, total_h_pixel, total_v_pixel):
        """
         设置图像输出大小

        h_pixel: 输出水平像素
        v_pixel: 输出垂直像素
        total_h_pixel: 总的水平像素
        total_y_pixel: 总的垂直像素
        """

        self.wr_reg(0X3212, 0X03)  # 开始组3
        # 以下设置决定实际输出尺寸(带缩放)
        self.wr_reg(0x3808, h_pixel >> 8)    # 设置实际输出宽度高字节
        self.wr_reg(0x3809, h_pixel & 0xff)  # 设置实际输出宽度低字节
        self.wr_reg(0x380a, v_pixel >> 8)    # 设置实际输出高度高字节
        self.wr_reg(0x380b, v_pixel & 0xff)  # 设置实际输出高度低字节
        # 以下设置总的水平像素和垂直像素数
        self.wr_reg(0x380c, total_h_pixel >> 8)    # 设置total_h_pixel高字节
        self.wr_reg(0x380d, total_h_pixel & 0xff)  # 设置total_h_pixel低字节
        self.wr_reg(0x380e, total_v_pixel >> 8)    # 设置total_v_pixel高字节
        self.wr_reg(0x380f, total_v_pixel & 0xff)  # 设置total_v_pixel低字节

        self.wr_reg(0X3212, 0X13)  # 结束组3
        self.wr_reg(0X3212, 0Xa3)  # 启用组3设置

    def set_offset(self, offx=16, offy=4):
        """
         OV5640输出图像的大小的偏移,完全由该函数确定
         offx,offy,为输出图像在OV5640_ImageWin_Set设定窗口(假设长宽为xsize和ysize)上的偏移
         由于开启了scale功能,用于输出的图像窗口为:xsize-2*offx,ysize-2*offy
         width,height:实际输出图像的宽度和高度
         实际输出(width,height),是在xsize-2*offx,ysize-2*offy的基础上进行缩放处理.
         一般设置offx和offy的值为16和4,更小也是可以,不过默认是16和4
        """
        # 以下设置决定输出尺寸在ISP上面的取图范围
        # 范围:xsize-2*offx,ysize-2*offy
        self.wr_reg(0x3810, offx >> 8)    # 设置X offset高字节
        self.wr_reg(0x3811, offx & 0xff)  # 设置X offset低字节

        self.wr_reg(0x3812, offy >> 8)    # 设置Y offset高字节
        self.wr_reg(0x3813, offy & 0xff)  # 设置Y offset低字节
