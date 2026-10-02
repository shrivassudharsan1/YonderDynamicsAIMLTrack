# Methodology

A YOLOv8n detector for `bottle` (0) and `mallet` (1). Three findings shaped the work:

1. The provided train/valid split has near-identical frames of the same scene on both sides, so its scores are optimistic. I re-split by scene.
2. A default-settings model finds mallets almost purely by their orange colour: rotate the hue and it finds 0 of 91. Colour augmentation fixes most of that for almost no cost on normal images.
3. The model's misses on unseen scenes are mostly large close-up objects, not small ones, and on my own video it never detects a claw hammer as a mallet.

Final model on the scene-grouped validation set (209 images), at the thresholds `predict.py` uses (confidence 0.25, NMS IoU 0.5):

| class | labelled | found | missed | false alarms | precision | recall | AP50 | AP50-95 |
|---|---|---|---|---|---|---|---|---|
| bottle | 162 | 112 | 50 | 40 | 0.737 | 0.691 | 0.707 | 0.373 |
| mallet | 91 | 76 | 15 | 10 | 0.884 | 0.835 | 0.922 | 0.528 |

## 1. How to run it

Tested on macOS 26 (Apple M4 Max), Python 3.13.0. Training used the Apple GPU (MPS); `train.py` picks CUDA, MPS or CPU automatically.

```
git clone https://github.com/shrivassudharsan1/YonderDynamicsAIMLTrack.git
cd YonderDynamicsAIMLTrack
bash setup.sh                          # creates .venv and installs requirements.txt
echo "ROBOFLOW_API_KEY=your_key" > .env   # or: export ROBOFLOW_API_KEY=your_key
bash run_all.sh                        # download, split, preprocess, evaluate, error analysis, inference
```

- The dataset download reads the key from the **`ROBOFLOW_API_KEY`** environment variable (or `.env`).
- `bash run_all.sh` uses the committed weights (`weights/best.pt`, 6 MB) and takes about a minute after the download. `bash run_all.sh --train` retrains the final model first (about 25 minutes on the M4 Max GPU).
- `setup.sh` picks the first of `python3.13`, `python3.12`, `python3` that is 3.12 or newer. To force one: `PYTHON=/path/to/python bash setup.sh`.

**Inference on a folder of images:**

```
.venv/bin/python predict.py <image_folder> <output_folder>
```

It writes one `<image_name>.txt` per image with one detection per line: `class_id x_center y_center width height confidence`, normalized 0-1, class ids as in the dataset. Images with no detections get an empty file. **Confidence threshold: 0.25** (`--conf`), NMS IoU 0.5 (`--iou`). Add `--save-images` to also save pictures with the boxes drawn.

Each step on its own:

| step | command | output |
|---|---|---|
| download | `python download_data.py` | `Sampled-YD-Object-Detection-2/` (never modified) |
| split | `python make_split.py` | `data/grouped/` |
| preprocess | `python preprocess.py` | `data/grouped_coloraug/` |
| train baseline | `python train.py --name baseline_grouped_100 --data data_grouped.yaml --epochs 100` | `runs/baseline_grouped_100/` |
| train final | `python train.py --name coloraug_grouped_50 --data data_grouped_coloraug.yaml --epochs 50` | `runs/coloraug_grouped_50/` |
| metrics | `python evaluate.py --weights weights/best.pt --data data_grouped.yaml` | per-class table, three colour conditions |
| error analysis | `python analyze_errors.py` | `results/errors/` |
| video test | `python video_test.py <video files>` | `results/video/` |
| efficiency | `python efficiency.py` | `results/efficiency.md` |
| hard examples | `python mine_hard_examples.py` | `results/hard_examples/` |

Every number below comes from a file in `results/`. A fresh clone plus `setup.sh` and `run_all.sh` reproduced the evaluation and error-analysis tables exactly. Retraining will not give identical weights (MPS training is not deterministic), so expect small differences if you use `--train`.

## 2. Thought process

### Looking at the data first

- All 999 images are 512x512, already resized by Roboflow with "stretch". Rotation (black corners) and brightness changes are already baked in.
- Objects are small: the median box covers about 1.3% of the image, and roughly a fifth are under 0.5%.
- Nearly every mallet is the same orange mallet. Bottles vary in colour and shape.
- Many images are consecutive video frames, with names like `t_bot_mal689` and `t_bot_mal693`.

### Data split

- No filename is in both provided folders, but neighbouring frames of the same scene are. Checked with ResNet-18 embeddings: for each validation image, the closest training image is usually the same scene shot moments apart.
- `make_split.py` clusters all 999 images into 92 scenes (average-linkage clustering of the embeddings, cosine distance cut at 0.25) and sends each whole scene to one side: 790 train / 209 valid, 0 scenes shared. The split list is committed (`splits/grouped_split.csv`) so it is the same on every machine.
- Same model, 40 epochs, each trained and scored on its own split:

  | split | bottle AP50 | mallet AP50 | mAP50 | mAP50-95 |
  |---|---|---|---|---|
  | provided | 0.854 | 0.961 | 0.908 | 0.593 |
  | scene-grouped | 0.709 | 0.938 | 0.823 | 0.437 |

- Bottles take almost all of the drop. All later numbers use the scene-grouped split.
- Caveat: the two validation sets contain different images, so not all of the drop is leakage.
- Caveat: the 0.25 distance cut was chosen by looking at cluster sizes at a few thresholds, not tuned. Clustering is on whole-image appearance, so two different scenes that look alike can be merged, and one scene shot from very different angles can be split.

### Preprocessing: colour augmentation

- The risk with a single orange mallet is that the model learns "orange blob" and not the shape. `evaluate.py` measures this by scoring the same validation images three ways: original, grayscale, and hue rotated by 90 degrees (orange becomes green). The shape is identical in all three.
- `preprocess.py` adds one extra copy of every training image, either grayscale (380 images) or with the hue rotated by a random 40-320 degrees (410 images). The originals stay, so 1,580 training images. Labels are copied unchanged. Validation images are not touched.
- Why this and not more rotation/brightness or contrast normalization: rotation and brightness are already in the dataset, and colour dependence was the weakness I had actually measured.
- Why written to a folder and not done on the fly: anyone can open `data/grouped_coloraug/` and see exactly what the model was trained on.
- Both models got the same number of training steps: baseline 100 epochs on 790 images, final 50 epochs on 1,580. Ultralytics' default augmentations (mosaic, flips, scale, small HSV jitter) were left on for both.

  | AP50 | validation images | baseline | colour-augmented |
  |---|---|---|---|
  | mallet | original | 0.928 | 0.922 |
  | mallet | grayscale | 0.072 | 0.733 |
  | mallet | hue-shifted | 0.020 | 0.871 |
  | bottle | original | 0.717 | 0.707 |
  | bottle | grayscale | 0.504 | 0.595 |
  | bottle | hue-shifted | 0.435 | 0.695 |
  | all (mAP50) | original | 0.822 | 0.815 |

- The baseline's mallet recall on hue-shifted images is 0.000 (0 of 91).
- The cost: on original images the colour-augmented model has lower mallet recall (0.813 vs 0.912) and higher precision (0.960 vs 0.887), at each class's best-F1 confidence. It misses more mallets and raises fewer false alarms.
- The real mallet is orange, so colour is a legitimate cue. If the rover only ever looks for that exact mallet in good light, the baseline's higher recall is a fair argument for it. I chose the augmented model because outdoor lighting and camera white balance change colour, and a model that fails completely when colour shifts is fragile.
- An earlier comparison at 40 vs 20 epochs showed a much larger cost (mallet AP50 0.938 vs 0.867). Both learning curves were still rising, so I reran both for longer; the gap mostly closed.

### Model

- YOLOv8n pretrained on COCO, 640 input, batch 16, Ultralytics defaults otherwise. Nano because the deployment target is an NPU and the dataset is small (790 images); a larger model was not tried.

### Evaluation

- Per class: precision, recall, AP50 and AP50-95. Accuracy is not used because detection has no meaningful count of true negatives.
- AP50 for comparing models, because it does not depend on a chosen threshold. Precision and recall at conf 0.25 for describing what `predict.py` actually outputs.
- There is no test set. The validation set was used to pick the best epoch and the NMS setting, so the reported numbers are somewhat optimistic.

### Error analysis

`analyze_errors.py`, final model, conf 0.25. Sheets: `results/errors/false_negatives.jpg`, `results/errors/false_positives.jpg` (green = label, red = prediction).

- **Large close-up objects are the main failure, not small ones.** Misses by share of the image the object fills:

  | class | tiny (<0.2%) | small (0.2-1%) | medium (1-5%) | large (>5%) |
  |---|---|---|---|---|
  | bottle | 6 / 14 | 14 / 47 | 13 / 60 | 17 / 41 |
  | mallet | 2 / 6 | 0 / 19 | 3 / 44 | 10 / 22 |

  These are close-ups and product photos on white backgrounds. A close-up of the mallet head was predicted as a bottle at 0.25. Likely cause: training is dominated by small distant objects, so image-filling ones are rare, and the product photos look nothing like the field photos.
- **Many errors are loose boxes.** 18 of the 50 missed bottles had a bottle prediction on them with IoU below 0.5. Each is counted as both a miss and a false alarm.
- **False alarms by cause** (bottle / mallet): background 16 / 3, loose box 20 / 5, duplicate 2 / 1, wrong class 2 / 1. The two classes are rarely confused.
- **Some "errors" are label problems.** Rotated images have inflated label boxes (a rotated box re-fitted as an upright rectangle gets bigger), and collage images have wrong labels. This is from looking at the sheets; I did not count them.
- **Cluttered scene.** Several mallet misses are one scene where the mallet lies across rust-coloured pipes, plus a motion-blurred frame.
- **Duplicates.** With Ultralytics' default NMS IoU (0.7) there were 17 duplicate boxes. At 0.5 there are 3, precision rises (bottle 0.679 to 0.737, mallet 0.809 to 0.884) and recall is unchanged, so `predict.py` uses 0.5. This was chosen on the validation set.

### Video test

Three phone videos (1080x1920, about 20 s each) in a dorm room on a concrete floor: a claw hammer with a black handle and steel head, a blue shaker bottle, and both together. Each starts across the room and moves in close. `video_test.py` samples 2 frames per second. The frames have no labels, so the counts are "frames with a box of that class"; whether the box is on the right object is from reading the sheets in `results/video/`.

| video | frames | with a mallet box | with a bottle box | with no box |
|---|---|---|---|---|
| hammer | 41 | 1 | 12 | 28 |
| bottle | 36 | 0 | 30 | 6 |
| both | 44 | 0 | 28 | 16 |

- **Bottle: works.** The box is on the bottle from across the room to close range; confidence rises from about 0.3 far away to about 0.9 mid-range. It is lost in the last frames when the bottle fills the frame and blurs, and when seen from directly above.
- **Hammer: never detected as a mallet.** The single mallet box is on a shoe rack. Not detected even when the hammer fills the width of the frame.
- **False alarms on furniture.** The 12 bottle boxes in the hammer video are on a mini-fridge and wardrobe drawers.
- **Baseline for comparison:** bottle boxes in 31 of 41 hammer-video frames (fridge, backpack, shoes, and the hammer handle labelled as a bottle). It does not detect the hammer as a mallet either.
- Squashing frames to a square like the training images made no real difference.
- What I take from this: colour augmentation made the model robust to the *same* mallet changing colour, not to a *different* mallet-like object. A claw hammer has a different shape, and the training set has one mallet design. Arguably the model is right that a claw hammer is not a mallet, but it means the model would likely miss a differently shaped mallet too. Augmentation cannot replace variety in the objects.

### Stretch: model efficiency

`efficiency.py`. YOLOv8n: 3.01 M parameters, 8.2 GFLOPs at 640, 6.3 MB.

| input size | bottle AP50 | mallet AP50 | mAP50 | CPU ms / image |
|---|---|---|---|---|
| 320 | 0.679 | 0.848 | 0.763 | 8 |
| 416 | 0.721 | 0.862 | 0.792 | 19-25 |
| 512 | 0.719 | 0.891 | 0.805 | 26-30 |
| 640 | 0.707 | 0.922 | 0.815 | 35-37 |

- Timings are wall-clock for the whole predict call on an M4 Max CPU, batch 1. They are **not** NPU numbers; only the relative trend is meaningful. The ranges are the spread over three runs.
- The jump from 320 to 416 is bigger than the pixel count explains. It repeated, and I do not know why.
- Trade-off: input size is the cheapest knob. 320 is about four times faster and costs 0.05 mAP50, mostly on mallets. For a rover that approaches the object, a lower resolution may be acceptable because the object gets bigger as it gets closer, but far-range detection suffers first.
- For the RKNN NPU, the next steps would be ONNX export and INT8 quantization with a calibration set drawn from real field frames, then re-running `evaluate.py` on the quantized model, since quantization can hurt small-object accuracy. **Not attempted here.**

### Stretch: hard example mining

`mine_hard_examples.py` scores every labelled object in the *training* set by the confidence of the best matching prediction. Anything still missed after training on it is genuinely hard, rare, or mislabelled.

- 29 of 969 training objects (3.0%) would be missed at conf 0.25. 17 of the 29 are tiny (<0.2% of the image): 14 of 47 tiny bottles and 3 of 28 tiny mallets.
- Large objects are almost never missed in training (1 of 190) but are the main misses in validation. So tiny objects are hard even when memorized (too few pixels at 512x512), while close-ups are learnable but the training set has too few distinct ones to generalize.
- Several of the hardest images are label problems: boxes cut off at the image edge after rotation, and collage images (scene 13).
- What to collect, in priority order, instead of more of the same:
  1. **More mallet designs and colours.** One design is the root of both the colour dependence and the hammer result.
  2. **Close-range views** of both objects, in the field, at the distances the rover's arm camera would see.
  3. **Higher-resolution captures of distant objects** (or tiled inference), since more 512x512 images of 20-pixel objects add little.
  4. **Negative images**: indoor scenes, furniture, tools and bags with no bottle or mallet, to cut the false alarms seen in the video.
  5. **Fix labels** in the worst scenes before adding data; `results/hard_examples/objects.csv` ranks them.
- As an ongoing loop: run the model on new rover footage, send the frames with confidence between about 0.1 and 0.5 (and frames where detections flicker between consecutive frames) for labelling first.

## 3. Known limitations

- **No held-out test set.** Best epoch and NMS IoU were picked on the validation set.
- **One training run per configuration.** No repeated seeds, so differences of about 0.01 mAP (such as 0.822 vs 0.815) are within noise.
- **The scene clustering is automatic** and unverified beyond spot checks; some leakage may remain.
- **Does not detect the hammer** in the video test, and gives false bottle boxes on furniture indoors.
- **Weak on close-ups and tiny objects**, and bottle recall is only 0.69.
- **Video test has no labels**; correctness there is judged by eye.
- **Label noise was observed, not fixed.** Relabelled data would go in a new folder.
- **Efficiency numbers are from a laptop**, and no quantization or NPU export was done.
- **Only YOLOv8n at default hyperparameters** was tried.

What I would do next: collect the data listed above, fix the labels, hold out a test set of whole scenes, and repeat the comparison over several seeds.
