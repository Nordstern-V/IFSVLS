# MIX_PLUS TCU bbox backend

This directory is the PYNQ-side software for plan step 4.

Files:

- tcu_pynq/: Tensil driver used by the Zynq TCU.
- resnet9_strided_16fmaps_onnx_custom_perf.tmodel, .tprog, .tdata: the model and program data.
- tcu_bbox_backend.py: preprocessing, 12-lane input packing, Driver.run wrapper, and 80D PEFSL feature adapter.
- tcu_feature_validate.py: deterministic board smoke test.

The bbox contract is:

1. Input is one 32x32 uint8 BGR frame.
2. Preprocess exactly as the desktop application: frame.astype(float32) / 255.0 - MEAN / STD.
3. Put the three channels into lanes 0..2 of a float32 array with shape (1024, 12); lanes 3..11 stay zero.
4. Call Driver.run({input_name: vec}).
5. The driver returns the two same-name Output segments concatenated as 84 values: 72 values followed by 12 values. The PEFSL feature is the first 72 plus the first 8 of the second segment, this assembled 80-value vector is the classifier input; it is not a physical contiguous DRAM slice. The last 4 values are retained as diagnostics.

## Board deployment

Copy the whole tcu_deploy directory, together with artifacts/pefsl_mix_plus_touch.bit and its .hwh, to the board. For example:

    /home/xilinx/jupyter_notebooks/PEFSL_MIX_PLUS/
      artifacts/pefsl_mix_plus_touch.bit
      tcu_deploy/
        tcu_pynq/
        resnet9_strided_16fmaps_onnx_custom_perf.tmodel
        resnet9_strided_16fmaps_onnx_custom_perf.tprog
        resnet9_strided_16fmaps_onnx_custom_perf.tdata
        tcu_bbox_backend.py
        tcu_feature_validate.py

Run:

    cd /home/xilinx/jupyter_notebooks/PEFSL_MIX_PLUS/tcu_deploy
    sudo -E /usr/local/share/pynq-venv/bin/python3 tcu_feature_validate.py --bitstream ../artifacts/pefsl_mix_plus_touch.bit --tmodel ./resnet9_strided_16fmaps_onnx_custom_perf.tmodel --repeats 2

Expected output includes raw84=84, feature80=80, tail4=..., and
TCU bbox feature validation PASS. The report is written to
validation_result/tcu_feature/tcu_feature_report.json.

This validates the TCU boundary only. Camera frame acquisition, touch state,
classification, and LCD drawing are the next integration layer.




