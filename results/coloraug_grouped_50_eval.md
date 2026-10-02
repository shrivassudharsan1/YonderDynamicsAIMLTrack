Weights: `runs/coloraug_grouped_50/weights/best.pt`  |  data: `data_grouped.yaml`  |  imgsz 640

Precision/recall are at the confidence that maximises F1 for that class (Ultralytics default).

| val images | class | boxes | precision | recall | AP50 | AP50-95 |
|---|---|---|---|---|---|---|
| original | bottle | 162 | 0.768 | 0.648 | 0.707 | 0.373 |
| original | mallet | 91 | 0.960 | 0.813 | 0.922 | 0.528 |
| original | **all** | | 0.864 | 0.731 | 0.815 | 0.450 |
| gray | bottle | 162 | 0.706 | 0.593 | 0.595 | 0.324 |
| gray | mallet | 91 | 0.928 | 0.637 | 0.733 | 0.412 |
| gray | **all** | | 0.817 | 0.615 | 0.664 | 0.368 |
| hue_shift | bottle | 162 | 0.823 | 0.601 | 0.695 | 0.366 |
| hue_shift | mallet | 91 | 0.936 | 0.736 | 0.871 | 0.490 |
| hue_shift | **all** | | 0.879 | 0.669 | 0.783 | 0.428 |
