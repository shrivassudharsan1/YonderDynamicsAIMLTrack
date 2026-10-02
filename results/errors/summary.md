Weights `weights/best.pt`, data `data_grouped.yaml`, 209 images, confidence >= 0.25, NMS IoU 0.5, match IoU >= 0.5

| class | labelled | TP | FN | FP | precision | recall |
|---|---|---|---|---|---|---|
| bottle | 162 | 112 | 50 | 40 | 0.737 | 0.691 |
| mallet | 91 | 76 | 15 | 10 | 0.884 | 0.835 |

False positives by cause:

| class | background | wrong class | loose box | duplicate |
|---|---|---|---|---|
| bottle | 16 | 2 | 20 | 2 |
| mallet | 3 | 1 | 5 | 1 |

Misses by object size (missed / labelled):

| class | tiny (<0.2% of image) | small (0.2-1%) | medium (1-5%) | large (>5%) |
|---|---|---|---|---|
| bottle | 6 / 14 | 14 / 47 | 13 / 60 | 17 / 41 |
| mallet | 2 / 6 | 0 / 19 | 3 / 44 | 10 / 22 |
