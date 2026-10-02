Weights: `runs/baseline_provided/weights/best.pt`  |  data: `data.yaml`  |  imgsz 640

Precision/recall are at the confidence that maximises F1 for that class (Ultralytics default).

| val images | class | boxes | precision | recall | AP50 | AP50-95 |
|---|---|---|---|---|---|---|
| original | bottle | 151 | 0.914 | 0.781 | 0.854 | 0.539 |
| original | mallet | 106 | 0.903 | 0.906 | 0.961 | 0.646 |
| original | **all** | | 0.909 | 0.844 | 0.908 | 0.593 |
| gray | bottle | 151 | 0.649 | 0.629 | 0.632 | 0.373 |
| gray | mallet | 106 | 0.773 | 0.033 | 0.098 | 0.035 |
| gray | **all** | | 0.711 | 0.331 | 0.365 | 0.204 |
| hue_shift | bottle | 151 | 0.794 | 0.629 | 0.679 | 0.434 |
| hue_shift | mallet | 106 | 1.000 | 0.015 | 0.206 | 0.078 |
| hue_shift | **all** | | 0.897 | 0.322 | 0.442 | 0.256 |
