Weights `/Users/shrivassudharsan/YonderDynamicsAI/weights/best.pt`: 3,011,238 parameters, 8.2 GFLOPs at 640, 6.3 MB on disk (FP16 checkpoint)

Timing: mean wall-clock time per image for the full predict call (resize + forward pass + NMS), batch 1, 100 validation images, torch 2.14.1.

| input size | bottle AP50 | mallet AP50 | mAP50 | mAP50-95 | CPU ms/image | CPU FPS | GPU ms/image |
|---|---|---|---|---|---|---|---|
| 320 | 0.679 | 0.848 | 0.763 | 0.437 | 8.1 | 123 | 3.8 |
| 416 | 0.721 | 0.862 | 0.792 | 0.446 | 24.7 | 40 | 4.0 |
| 512 | 0.719 | 0.891 | 0.805 | 0.447 | 30.1 | 33 | 3.8 |
| 640 | 0.707 | 0.922 | 0.815 | 0.450 | 36.8 | 27 | 4.8 |
