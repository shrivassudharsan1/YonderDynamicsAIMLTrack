Source weights `weights/best.pt` (6.3 MB), input 640, calibrated on 200 training images, scored on `data_grouped.yaml` with ONNX Runtime on CPU.

| model | file size MB | bottle AP50 | mallet AP50 | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| fp32 | 12.3 | 0.713 | 0.900 | 0.806 | 0.441 |
| int8 all | 3.4 | 0.000 | 0.000 | 0.000 | 0.000 |
| int8 backbone | 3.5 | 0.706 | 0.909 | 0.808 | 0.429 |
