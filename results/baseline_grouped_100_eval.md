Weights: `runs/baseline_grouped_100/weights/best.pt`  |  data: `data_grouped.yaml`  |  imgsz 640

Precision/recall are at the confidence that maximises F1 for that class (Ultralytics default).

| val images | class | boxes | precision | recall | AP50 | AP50-95 |
|---|---|---|---|---|---|---|
| original | bottle | 162 | 0.779 | 0.652 | 0.717 | 0.366 |
| original | mallet | 91 | 0.887 | 0.912 | 0.928 | 0.549 |
| original | **all** | | 0.833 | 0.782 | 0.822 | 0.457 |
| gray | bottle | 162 | 0.630 | 0.524 | 0.504 | 0.270 |
| gray | mallet | 91 | 0.880 | 0.011 | 0.072 | 0.027 |
| gray | **all** | | 0.755 | 0.268 | 0.288 | 0.149 |
| hue_shift | bottle | 162 | 0.544 | 0.407 | 0.435 | 0.234 |
| hue_shift | mallet | 91 | 1.000 | 0.000 | 0.020 | 0.008 |
| hue_shift | **all** | | 0.772 | 0.204 | 0.227 | 0.121 |
