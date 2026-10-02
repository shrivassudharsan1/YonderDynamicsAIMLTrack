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

Misses that still had a same-class box on them (IoU 0.1-0.5), so they are loose boxes rather than unseen objects:

- bottle: 18 of 50 misses
- mallet: 3 of 15 misses

Scenes with the most misses (missed / labelled in that scene; scene ids from splits/grouped_split.csv):

- bottle: scene 12 15 / 26, scene 25 5 / 11, scene 42 5 / 36, scene 53 4 / 5, scene 48 3 / 5
- mallet: scene 78 5 / 5, scene 82 3 / 15, scene 3 2 / 3, scene 91 1 / 42, scene 49 1 / 6
