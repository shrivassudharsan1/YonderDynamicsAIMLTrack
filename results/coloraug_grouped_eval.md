Weights: `runs/coloraug_grouped/weights/best.pt`  |  data: `data_grouped.yaml`  |  imgsz 640

Precision/recall are at the confidence that maximises F1 for that class (Ultralytics default).

| val images | class | boxes | precision | recall | AP50 | AP50-95 |
|---|---|---|---|---|---|---|
| original | bottle | 162 | 0.717 | 0.608 | 0.665 | 0.311 |
| original | mallet | 91 | 0.879 | 0.725 | 0.867 | 0.447 |
| original | **all** | | 0.798 | 0.667 | 0.766 | 0.379 |
| gray | bottle | 162 | 0.650 | 0.586 | 0.595 | 0.288 |
| gray | mallet | 91 | 0.808 | 0.451 | 0.583 | 0.300 |
| gray | **all** | | 0.729 | 0.518 | 0.589 | 0.294 |
| hue_shift | bottle | 162 | 0.684 | 0.660 | 0.667 | 0.300 |
| hue_shift | mallet | 91 | 0.803 | 0.703 | 0.817 | 0.429 |
| hue_shift | **all** | | 0.744 | 0.682 | 0.742 | 0.365 |
