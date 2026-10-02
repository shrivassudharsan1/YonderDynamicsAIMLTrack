Weights `/Users/shrivassudharsan/YonderDynamicsAI/weights/best.pt`, 790 training images, 969 labelled objects.
29 objects (3.0%) would be missed at the deployment threshold (best matching confidence < 0.25), on images the model was trained on.

By class and object size (objects below the threshold / objects, mean confidence):

| class | tiny (<0.2% of image) | small (0.2-1%) | medium (1-5%) | large (>5%) |
|---|---|---|---|---|
| bottle | 14 / 47, 0.52 | 5 / 182, 0.79 | 5 / 149, 0.84 | 1 / 145, 0.87 |
| mallet | 3 / 28, 0.61 | 1 / 174, 0.76 | 0 / 199, 0.83 | 0 / 45, 0.77 |

Hardest scenes (at least 5 objects), lowest mean confidence first:

| scene | objects | mean confidence | below threshold |
|---|---|---|---|
| 13 | 22 | 0.51 | 7 |
| 47 | 23 | 0.51 | 8 |
| 52 | 21 | 0.64 | 1 |
| 26 | 7 | 0.68 | 0 |
| 50 | 87 | 0.71 | 4 |
| 67 | 42 | 0.72 | 0 |
| 64 | 17 | 0.73 | 1 |
| 63 | 46 | 0.75 | 4 |
| 80 | 5 | 0.75 | 0 |
| 54 | 8 | 0.76 | 0 |
