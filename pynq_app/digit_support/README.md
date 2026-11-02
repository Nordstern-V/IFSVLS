# r17 digit support set

This directory is the board-side copy of the digit support images from
`E:\FPGA-3up\PEFSL\PEFSL-main\support`.

Only classes `0` through `9` are included.  Operator classes (`plus`, `sub`,
`mul`, `div`) are excluded from the first digit mode.  The updated support set
uses formal `<digit>_<index>.png` names for some classes and
`debug_binary_*.png` names for others; `digit_recognizer.py` accepts both
known formats when building the board cache.

The images are used only as input data.  They must be passed through the r17
preprocessing and `TcuFeatureExtractor` on the board before a
`digit_support_r17.npz` cache is created.  A PC-side feature cache must not be
copied here because the PC and r17 TCU interfaces are different.

The current updated support sample counts are:

| Class | Samples |
| --- | ---: |
| 0–4 | 5 each |
| 5 | 4 |
| 6–9 | 5 each |

The resulting updated set contains 49 images.  The cache file is generated
on the board with the current r17 bitstream and TCU model.
