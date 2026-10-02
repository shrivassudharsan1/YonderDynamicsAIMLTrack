"""Build a colour-augmented training set in data/grouped_coloraug/ (the download is left untouched).

Why: almost every mallet in the dataset is the same orange mallet, and the baseline model
finds far fewer of them once colour is removed (see evaluate.py). To push the model
towards shape, every training image gets one extra copy with its colour changed:

  gray       colour removed (3-channel grayscale)
  hue shift  hue rotated by a random 40-320 degrees, so orange becomes some other colour

Boxes are unchanged, so the label file is copied as is. Validation images are NOT touched:
data_grouped_coloraug.yaml validates on the same data/grouped/valid as the baseline.

    python make_split.py   # first, builds data/grouped/
    python preprocess.py
"""

import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
SOURCE = ROOT / "data" / "grouped" / "train"
OUT = ROOT / "data" / "grouped_coloraug" / "train"
SEED = 0


def to_gray(image):
    return cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def hue_shift(image, degrees):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = (hsv[..., 0].astype(np.int32) + degrees // 2) % 180  # OpenCV hue is 0-179
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def main():
    if not SOURCE.exists():
        raise SystemExit("data/grouped/ not found; run make_split.py first.")
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)
    (OUT / "labels").mkdir(parents=True)

    rng = random.Random(SEED)
    counts = Counter()
    for image_path in sorted((SOURCE / "images").iterdir()):
        label_path = SOURCE / "labels" / f"{image_path.stem}.txt"
        shutil.copy2(image_path, OUT / "images" / image_path.name)
        shutil.copy2(label_path, OUT / "labels" / label_path.name)
        counts["original"] += 1

        image = cv2.imread(str(image_path))
        if rng.random() < 0.5:
            tag, augmented = "gray", to_gray(image)
        else:
            tag, augmented = "hue", hue_shift(image, rng.randint(40, 320))
        cv2.imwrite(str(OUT / "images" / f"{image_path.stem}_{tag}.jpg"), augmented)
        shutil.copy2(label_path, OUT / "labels" / f"{image_path.stem}_{tag}.txt")
        counts[tag] += 1

    print(f"{OUT.relative_to(ROOT)}: {dict(counts)}, {sum(counts.values())} images total")


if __name__ == "__main__":
    main()
