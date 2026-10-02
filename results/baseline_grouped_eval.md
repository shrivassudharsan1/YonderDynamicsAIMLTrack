Weights: `runs/baseline_grouped/weights/best.pt`  |  data: `data_grouped.yaml`  |  imgsz 640

Precision/recall are at the confidence that maximises F1 for that class (Ultralytics default).

| val images | class | boxes | precision | recall | AP50 | AP50-95 |
|---|---|---|---|---|---|---|
| original | bottle | 162 | 0.734 | 0.679 | 0.709 | 0.341 |
| original | mallet | 91 | 0.900 | 0.888 | 0.938 | 0.533 |
| original | **all** | | 0.817 | 0.783 | 0.823 | 0.437 |
| gray | bottle | 162 | 0.398 | 0.574 | 0.506 | 0.253 |
| gray | mallet | 91 | 0.720 | 0.143 | 0.185 | 0.090 |
| gray | **all** | | 0.559 | 0.358 | 0.346 | 0.172 |
| hue_shift | bottle | 162 | 0.520 | 0.543 | 0.486 | 0.238 |
| hue_shift | mallet | 91 | 0.899 | 0.033 | 0.143 | 0.078 |
| hue_shift | **all** | | 0.710 | 0.288 | 0.315 | 0.158 |
