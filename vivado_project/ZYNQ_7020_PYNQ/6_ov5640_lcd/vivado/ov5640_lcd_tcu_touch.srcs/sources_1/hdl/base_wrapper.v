//Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
//Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//--------------------------------------------------------------------------------
//Tool Version: Vivado v.2025.2 (win64) Build 6299465 Fri Nov 14 19:35:11 GMT 2025
//Date        : Thu Oct 29 21:40:01 2026
//Host        : LAPTOP-Y7000P running 64-bit major release  (build 9200)
//Command     : generate_target base_wrapper.bd
//Design      : base_wrapper
//Purpose     : IP block netlist
//--------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

module base_wrapper
   (DDR_addr,
    DDR_ba,
    DDR_cas_n,
    DDR_ck_n,
    DDR_ck_p,
    DDR_cke,
    DDR_cs_n,
    DDR_dm,
    DDR_dq,
    DDR_dqs_n,
    DDR_dqs_p,
    DDR_odt,
    DDR_ras_n,
    DDR_reset_n,
    DDR_we_n,
    FIXED_IO_ddr_vrn,
    FIXED_IO_ddr_vrp,
    FIXED_IO_mio,
    FIXED_IO_ps_clk,
    FIXED_IO_ps_porb,
    FIXED_IO_ps_srstb,
    IIC_1_scl_io,
    IIC_1_sda_io,
    cam_data,
    cam_href,
    cam_iic_scl_io,
    cam_iic_sda_io,
    cam_pclk,
    cam_pwdn,
    cam_rst_n,
    cam_vsync,
    lcd_bl,
    lcd_clk,
    lcd_de,
    lcd_hs,
    lcd_rgb_tri_io,
    lcd_vs,
    touch_iic_scl_io,
    touch_iic_sda_io,
    touch_int_tri_i,
    touch_rst_n_tri_o);
  inout [14:0]DDR_addr;
  inout [2:0]DDR_ba;
  inout DDR_cas_n;
  inout DDR_ck_n;
  inout DDR_ck_p;
  inout DDR_cke;
  inout DDR_cs_n;
  inout [3:0]DDR_dm;
  inout [31:0]DDR_dq;
  inout [3:0]DDR_dqs_n;
  inout [3:0]DDR_dqs_p;
  inout DDR_odt;
  inout DDR_ras_n;
  inout DDR_reset_n;
  inout DDR_we_n;
  inout FIXED_IO_ddr_vrn;
  inout FIXED_IO_ddr_vrp;
  inout [53:0]FIXED_IO_mio;
  inout FIXED_IO_ps_clk;
  inout FIXED_IO_ps_porb;
  inout FIXED_IO_ps_srstb;
  inout IIC_1_scl_io;
  inout IIC_1_sda_io;
  input [7:0]cam_data;
  input cam_href;
  inout cam_iic_scl_io;
  inout cam_iic_sda_io;
  input cam_pclk;
  output cam_pwdn;
  output cam_rst_n;
  input cam_vsync;
  output lcd_bl;
  output lcd_clk;
  output lcd_de;
  output lcd_hs;
  inout [23:0]lcd_rgb_tri_io;
  output lcd_vs;
  inout touch_iic_scl_io;
  inout touch_iic_sda_io;
  input [0:0]touch_int_tri_i;
  output [0:0]touch_rst_n_tri_o;

  wire [14:0]DDR_addr;
  wire [2:0]DDR_ba;
  wire DDR_cas_n;
  wire DDR_ck_n;
  wire DDR_ck_p;
  wire DDR_cke;
  wire DDR_cs_n;
  wire [3:0]DDR_dm;
  wire [31:0]DDR_dq;
  wire [3:0]DDR_dqs_n;
  wire [3:0]DDR_dqs_p;
  wire DDR_odt;
  wire DDR_ras_n;
  wire DDR_reset_n;
  wire DDR_we_n;
  wire FIXED_IO_ddr_vrn;
  wire FIXED_IO_ddr_vrp;
  wire [53:0]FIXED_IO_mio;
  wire FIXED_IO_ps_clk;
  wire FIXED_IO_ps_porb;
  wire FIXED_IO_ps_srstb;
  wire IIC_1_scl_i;
  wire IIC_1_scl_io;
  wire IIC_1_scl_o;
  wire IIC_1_scl_t;
  wire IIC_1_sda_i;
  wire IIC_1_sda_io;
  wire IIC_1_sda_o;
  wire IIC_1_sda_t;
  wire [7:0]cam_data;
  wire cam_href;
  wire cam_iic_scl_i;
  wire cam_iic_scl_io;
  wire cam_iic_scl_o;
  wire cam_iic_scl_t;
  wire cam_iic_sda_i;
  wire cam_iic_sda_io;
  wire cam_iic_sda_o;
  wire cam_iic_sda_t;
  wire cam_pclk;
  wire cam_pwdn;
  wire cam_rst_n;
  wire cam_vsync;
  wire lcd_bl;
  wire lcd_clk;
  wire lcd_de;
  wire lcd_hs;
  wire [0:0]lcd_rgb_tri_i_0;
  wire [1:1]lcd_rgb_tri_i_1;
  wire [10:10]lcd_rgb_tri_i_10;
  wire [11:11]lcd_rgb_tri_i_11;
  wire [12:12]lcd_rgb_tri_i_12;
  wire [13:13]lcd_rgb_tri_i_13;
  wire [14:14]lcd_rgb_tri_i_14;
  wire [15:15]lcd_rgb_tri_i_15;
  wire [16:16]lcd_rgb_tri_i_16;
  wire [17:17]lcd_rgb_tri_i_17;
  wire [18:18]lcd_rgb_tri_i_18;
  wire [19:19]lcd_rgb_tri_i_19;
  wire [2:2]lcd_rgb_tri_i_2;
  wire [20:20]lcd_rgb_tri_i_20;
  wire [21:21]lcd_rgb_tri_i_21;
  wire [22:22]lcd_rgb_tri_i_22;
  wire [23:23]lcd_rgb_tri_i_23;
  wire [3:3]lcd_rgb_tri_i_3;
  wire [4:4]lcd_rgb_tri_i_4;
  wire [5:5]lcd_rgb_tri_i_5;
  wire [6:6]lcd_rgb_tri_i_6;
  wire [7:7]lcd_rgb_tri_i_7;
  wire [8:8]lcd_rgb_tri_i_8;
  wire [9:9]lcd_rgb_tri_i_9;
  wire [0:0]lcd_rgb_tri_io_0;
  wire [1:1]lcd_rgb_tri_io_1;
  wire [10:10]lcd_rgb_tri_io_10;
  wire [11:11]lcd_rgb_tri_io_11;
  wire [12:12]lcd_rgb_tri_io_12;
  wire [13:13]lcd_rgb_tri_io_13;
  wire [14:14]lcd_rgb_tri_io_14;
  wire [15:15]lcd_rgb_tri_io_15;
  wire [16:16]lcd_rgb_tri_io_16;
  wire [17:17]lcd_rgb_tri_io_17;
  wire [18:18]lcd_rgb_tri_io_18;
  wire [19:19]lcd_rgb_tri_io_19;
  wire [2:2]lcd_rgb_tri_io_2;
  wire [20:20]lcd_rgb_tri_io_20;
  wire [21:21]lcd_rgb_tri_io_21;
  wire [22:22]lcd_rgb_tri_io_22;
  wire [23:23]lcd_rgb_tri_io_23;
  wire [3:3]lcd_rgb_tri_io_3;
  wire [4:4]lcd_rgb_tri_io_4;
  wire [5:5]lcd_rgb_tri_io_5;
  wire [6:6]lcd_rgb_tri_io_6;
  wire [7:7]lcd_rgb_tri_io_7;
  wire [8:8]lcd_rgb_tri_io_8;
  wire [9:9]lcd_rgb_tri_io_9;
  wire [0:0]lcd_rgb_tri_o_0;
  wire [1:1]lcd_rgb_tri_o_1;
  wire [10:10]lcd_rgb_tri_o_10;
  wire [11:11]lcd_rgb_tri_o_11;
  wire [12:12]lcd_rgb_tri_o_12;
  wire [13:13]lcd_rgb_tri_o_13;
  wire [14:14]lcd_rgb_tri_o_14;
  wire [15:15]lcd_rgb_tri_o_15;
  wire [16:16]lcd_rgb_tri_o_16;
  wire [17:17]lcd_rgb_tri_o_17;
  wire [18:18]lcd_rgb_tri_o_18;
  wire [19:19]lcd_rgb_tri_o_19;
  wire [2:2]lcd_rgb_tri_o_2;
  wire [20:20]lcd_rgb_tri_o_20;
  wire [21:21]lcd_rgb_tri_o_21;
  wire [22:22]lcd_rgb_tri_o_22;
  wire [23:23]lcd_rgb_tri_o_23;
  wire [3:3]lcd_rgb_tri_o_3;
  wire [4:4]lcd_rgb_tri_o_4;
  wire [5:5]lcd_rgb_tri_o_5;
  wire [6:6]lcd_rgb_tri_o_6;
  wire [7:7]lcd_rgb_tri_o_7;
  wire [8:8]lcd_rgb_tri_o_8;
  wire [9:9]lcd_rgb_tri_o_9;
  wire [0:0]lcd_rgb_tri_t_0;
  wire [1:1]lcd_rgb_tri_t_1;
  wire [10:10]lcd_rgb_tri_t_10;
  wire [11:11]lcd_rgb_tri_t_11;
  wire [12:12]lcd_rgb_tri_t_12;
  wire [13:13]lcd_rgb_tri_t_13;
  wire [14:14]lcd_rgb_tri_t_14;
  wire [15:15]lcd_rgb_tri_t_15;
  wire [16:16]lcd_rgb_tri_t_16;
  wire [17:17]lcd_rgb_tri_t_17;
  wire [18:18]lcd_rgb_tri_t_18;
  wire [19:19]lcd_rgb_tri_t_19;
  wire [2:2]lcd_rgb_tri_t_2;
  wire [20:20]lcd_rgb_tri_t_20;
  wire [21:21]lcd_rgb_tri_t_21;
  wire [22:22]lcd_rgb_tri_t_22;
  wire [23:23]lcd_rgb_tri_t_23;
  wire [3:3]lcd_rgb_tri_t_3;
  wire [4:4]lcd_rgb_tri_t_4;
  wire [5:5]lcd_rgb_tri_t_5;
  wire [6:6]lcd_rgb_tri_t_6;
  wire [7:7]lcd_rgb_tri_t_7;
  wire [8:8]lcd_rgb_tri_t_8;
  wire [9:9]lcd_rgb_tri_t_9;
  wire lcd_vs;
  wire touch_iic_scl_i;
  wire touch_iic_scl_io;
  wire touch_iic_scl_o;
  wire touch_iic_scl_t;
  wire touch_iic_sda_i;
  wire touch_iic_sda_io;
  wire touch_iic_sda_o;
  wire touch_iic_sda_t;
  wire [0:0]touch_int_tri_i;
  wire [0:0]touch_rst_n_tri_o;

  IOBUF IIC_1_scl_iobuf
       (.I(IIC_1_scl_o),
        .IO(IIC_1_scl_io),
        .O(IIC_1_scl_i),
        .T(IIC_1_scl_t));
  IOBUF IIC_1_sda_iobuf
       (.I(IIC_1_sda_o),
        .IO(IIC_1_sda_io),
        .O(IIC_1_sda_i),
        .T(IIC_1_sda_t));
  base base_i
       (.DDR_addr(DDR_addr),
        .DDR_ba(DDR_ba),
        .DDR_cas_n(DDR_cas_n),
        .DDR_ck_n(DDR_ck_n),
        .DDR_ck_p(DDR_ck_p),
        .DDR_cke(DDR_cke),
        .DDR_cs_n(DDR_cs_n),
        .DDR_dm(DDR_dm),
        .DDR_dq(DDR_dq),
        .DDR_dqs_n(DDR_dqs_n),
        .DDR_dqs_p(DDR_dqs_p),
        .DDR_odt(DDR_odt),
        .DDR_ras_n(DDR_ras_n),
        .DDR_reset_n(DDR_reset_n),
        .DDR_we_n(DDR_we_n),
        .FIXED_IO_ddr_vrn(FIXED_IO_ddr_vrn),
        .FIXED_IO_ddr_vrp(FIXED_IO_ddr_vrp),
        .FIXED_IO_mio(FIXED_IO_mio),
        .FIXED_IO_ps_clk(FIXED_IO_ps_clk),
        .FIXED_IO_ps_porb(FIXED_IO_ps_porb),
        .FIXED_IO_ps_srstb(FIXED_IO_ps_srstb),
        .IIC_1_scl_i(IIC_1_scl_i),
        .IIC_1_scl_o(IIC_1_scl_o),
        .IIC_1_scl_t(IIC_1_scl_t),
        .IIC_1_sda_i(IIC_1_sda_i),
        .IIC_1_sda_o(IIC_1_sda_o),
        .IIC_1_sda_t(IIC_1_sda_t),
        .cam_data(cam_data),
        .cam_href(cam_href),
        .cam_iic_scl_i(cam_iic_scl_i),
        .cam_iic_scl_o(cam_iic_scl_o),
        .cam_iic_scl_t(cam_iic_scl_t),
        .cam_iic_sda_i(cam_iic_sda_i),
        .cam_iic_sda_o(cam_iic_sda_o),
        .cam_iic_sda_t(cam_iic_sda_t),
        .cam_pclk(cam_pclk),
        .cam_pwdn(cam_pwdn),
        .cam_rst_n(cam_rst_n),
        .cam_vsync(cam_vsync),
        .lcd_bl(lcd_bl),
        .lcd_clk(lcd_clk),
        .lcd_de(lcd_de),
        .lcd_hs(lcd_hs),
        .lcd_rgb_tri_i({lcd_rgb_tri_i_23,lcd_rgb_tri_i_22,lcd_rgb_tri_i_21,lcd_rgb_tri_i_20,lcd_rgb_tri_i_19,lcd_rgb_tri_i_18,lcd_rgb_tri_i_17,lcd_rgb_tri_i_16,lcd_rgb_tri_i_15,lcd_rgb_tri_i_14,lcd_rgb_tri_i_13,lcd_rgb_tri_i_12,lcd_rgb_tri_i_11,lcd_rgb_tri_i_10,lcd_rgb_tri_i_9,lcd_rgb_tri_i_8,lcd_rgb_tri_i_7,lcd_rgb_tri_i_6,lcd_rgb_tri_i_5,lcd_rgb_tri_i_4,lcd_rgb_tri_i_3,lcd_rgb_tri_i_2,lcd_rgb_tri_i_1,lcd_rgb_tri_i_0}),
        .lcd_rgb_tri_o({lcd_rgb_tri_o_23,lcd_rgb_tri_o_22,lcd_rgb_tri_o_21,lcd_rgb_tri_o_20,lcd_rgb_tri_o_19,lcd_rgb_tri_o_18,lcd_rgb_tri_o_17,lcd_rgb_tri_o_16,lcd_rgb_tri_o_15,lcd_rgb_tri_o_14,lcd_rgb_tri_o_13,lcd_rgb_tri_o_12,lcd_rgb_tri_o_11,lcd_rgb_tri_o_10,lcd_rgb_tri_o_9,lcd_rgb_tri_o_8,lcd_rgb_tri_o_7,lcd_rgb_tri_o_6,lcd_rgb_tri_o_5,lcd_rgb_tri_o_4,lcd_rgb_tri_o_3,lcd_rgb_tri_o_2,lcd_rgb_tri_o_1,lcd_rgb_tri_o_0}),
        .lcd_rgb_tri_t({lcd_rgb_tri_t_23,lcd_rgb_tri_t_22,lcd_rgb_tri_t_21,lcd_rgb_tri_t_20,lcd_rgb_tri_t_19,lcd_rgb_tri_t_18,lcd_rgb_tri_t_17,lcd_rgb_tri_t_16,lcd_rgb_tri_t_15,lcd_rgb_tri_t_14,lcd_rgb_tri_t_13,lcd_rgb_tri_t_12,lcd_rgb_tri_t_11,lcd_rgb_tri_t_10,lcd_rgb_tri_t_9,lcd_rgb_tri_t_8,lcd_rgb_tri_t_7,lcd_rgb_tri_t_6,lcd_rgb_tri_t_5,lcd_rgb_tri_t_4,lcd_rgb_tri_t_3,lcd_rgb_tri_t_2,lcd_rgb_tri_t_1,lcd_rgb_tri_t_0}),
        .lcd_vs(lcd_vs),
        .touch_iic_scl_i(touch_iic_scl_i),
        .touch_iic_scl_o(touch_iic_scl_o),
        .touch_iic_scl_t(touch_iic_scl_t),
        .touch_iic_sda_i(touch_iic_sda_i),
        .touch_iic_sda_o(touch_iic_sda_o),
        .touch_iic_sda_t(touch_iic_sda_t),
        .touch_int_tri_i(touch_int_tri_i),
        .touch_rst_n_tri_o(touch_rst_n_tri_o));
  IOBUF cam_iic_scl_iobuf
       (.I(cam_iic_scl_o),
        .IO(cam_iic_scl_io),
        .O(cam_iic_scl_i),
        .T(cam_iic_scl_t));
  IOBUF cam_iic_sda_iobuf
       (.I(cam_iic_sda_o),
        .IO(cam_iic_sda_io),
        .O(cam_iic_sda_i),
        .T(cam_iic_sda_t));
  IOBUF lcd_rgb_tri_iobuf_0
       (.I(lcd_rgb_tri_o_0),
        .IO(lcd_rgb_tri_io[0]),
        .O(lcd_rgb_tri_i_0),
        .T(lcd_rgb_tri_t_0));
  IOBUF lcd_rgb_tri_iobuf_1
       (.I(lcd_rgb_tri_o_1),
        .IO(lcd_rgb_tri_io[1]),
        .O(lcd_rgb_tri_i_1),
        .T(lcd_rgb_tri_t_1));
  IOBUF lcd_rgb_tri_iobuf_10
       (.I(lcd_rgb_tri_o_10),
        .IO(lcd_rgb_tri_io[10]),
        .O(lcd_rgb_tri_i_10),
        .T(lcd_rgb_tri_t_10));
  IOBUF lcd_rgb_tri_iobuf_11
       (.I(lcd_rgb_tri_o_11),
        .IO(lcd_rgb_tri_io[11]),
        .O(lcd_rgb_tri_i_11),
        .T(lcd_rgb_tri_t_11));
  IOBUF lcd_rgb_tri_iobuf_12
       (.I(lcd_rgb_tri_o_12),
        .IO(lcd_rgb_tri_io[12]),
        .O(lcd_rgb_tri_i_12),
        .T(lcd_rgb_tri_t_12));
  IOBUF lcd_rgb_tri_iobuf_13
       (.I(lcd_rgb_tri_o_13),
        .IO(lcd_rgb_tri_io[13]),
        .O(lcd_rgb_tri_i_13),
        .T(lcd_rgb_tri_t_13));
  IOBUF lcd_rgb_tri_iobuf_14
       (.I(lcd_rgb_tri_o_14),
        .IO(lcd_rgb_tri_io[14]),
        .O(lcd_rgb_tri_i_14),
        .T(lcd_rgb_tri_t_14));
  IOBUF lcd_rgb_tri_iobuf_15
       (.I(lcd_rgb_tri_o_15),
        .IO(lcd_rgb_tri_io[15]),
        .O(lcd_rgb_tri_i_15),
        .T(lcd_rgb_tri_t_15));
  IOBUF lcd_rgb_tri_iobuf_16
       (.I(lcd_rgb_tri_o_16),
        .IO(lcd_rgb_tri_io[16]),
        .O(lcd_rgb_tri_i_16),
        .T(lcd_rgb_tri_t_16));
  IOBUF lcd_rgb_tri_iobuf_17
       (.I(lcd_rgb_tri_o_17),
        .IO(lcd_rgb_tri_io[17]),
        .O(lcd_rgb_tri_i_17),
        .T(lcd_rgb_tri_t_17));
  IOBUF lcd_rgb_tri_iobuf_18
       (.I(lcd_rgb_tri_o_18),
        .IO(lcd_rgb_tri_io[18]),
        .O(lcd_rgb_tri_i_18),
        .T(lcd_rgb_tri_t_18));
  IOBUF lcd_rgb_tri_iobuf_19
       (.I(lcd_rgb_tri_o_19),
        .IO(lcd_rgb_tri_io[19]),
        .O(lcd_rgb_tri_i_19),
        .T(lcd_rgb_tri_t_19));
  IOBUF lcd_rgb_tri_iobuf_2
       (.I(lcd_rgb_tri_o_2),
        .IO(lcd_rgb_tri_io[2]),
        .O(lcd_rgb_tri_i_2),
        .T(lcd_rgb_tri_t_2));
  IOBUF lcd_rgb_tri_iobuf_20
       (.I(lcd_rgb_tri_o_20),
        .IO(lcd_rgb_tri_io[20]),
        .O(lcd_rgb_tri_i_20),
        .T(lcd_rgb_tri_t_20));
  IOBUF lcd_rgb_tri_iobuf_21
       (.I(lcd_rgb_tri_o_21),
        .IO(lcd_rgb_tri_io[21]),
        .O(lcd_rgb_tri_i_21),
        .T(lcd_rgb_tri_t_21));
  IOBUF lcd_rgb_tri_iobuf_22
       (.I(lcd_rgb_tri_o_22),
        .IO(lcd_rgb_tri_io[22]),
        .O(lcd_rgb_tri_i_22),
        .T(lcd_rgb_tri_t_22));
  IOBUF lcd_rgb_tri_iobuf_23
       (.I(lcd_rgb_tri_o_23),
        .IO(lcd_rgb_tri_io[23]),
        .O(lcd_rgb_tri_i_23),
        .T(lcd_rgb_tri_t_23));
  IOBUF lcd_rgb_tri_iobuf_3
       (.I(lcd_rgb_tri_o_3),
        .IO(lcd_rgb_tri_io[3]),
        .O(lcd_rgb_tri_i_3),
        .T(lcd_rgb_tri_t_3));
  IOBUF lcd_rgb_tri_iobuf_4
       (.I(lcd_rgb_tri_o_4),
        .IO(lcd_rgb_tri_io[4]),
        .O(lcd_rgb_tri_i_4),
        .T(lcd_rgb_tri_t_4));
  IOBUF lcd_rgb_tri_iobuf_5
       (.I(lcd_rgb_tri_o_5),
        .IO(lcd_rgb_tri_io[5]),
        .O(lcd_rgb_tri_i_5),
        .T(lcd_rgb_tri_t_5));
  IOBUF lcd_rgb_tri_iobuf_6
       (.I(lcd_rgb_tri_o_6),
        .IO(lcd_rgb_tri_io[6]),
        .O(lcd_rgb_tri_i_6),
        .T(lcd_rgb_tri_t_6));
  IOBUF lcd_rgb_tri_iobuf_7
       (.I(lcd_rgb_tri_o_7),
        .IO(lcd_rgb_tri_io[7]),
        .O(lcd_rgb_tri_i_7),
        .T(lcd_rgb_tri_t_7));
  IOBUF lcd_rgb_tri_iobuf_8
       (.I(lcd_rgb_tri_o_8),
        .IO(lcd_rgb_tri_io[8]),
        .O(lcd_rgb_tri_i_8),
        .T(lcd_rgb_tri_t_8));
  IOBUF lcd_rgb_tri_iobuf_9
       (.I(lcd_rgb_tri_o_9),
        .IO(lcd_rgb_tri_io[9]),
        .O(lcd_rgb_tri_i_9),
        .T(lcd_rgb_tri_t_9));
  IOBUF touch_iic_scl_iobuf
       (.I(touch_iic_scl_o),
        .IO(touch_iic_scl_io),
        .O(touch_iic_scl_i),
        .T(touch_iic_scl_t));
  IOBUF touch_iic_sda_iobuf
       (.I(touch_iic_sda_o),
        .IO(touch_iic_sda_io),
        .O(touch_iic_sda_i),
        .T(touch_iic_sda_t));
endmodule
