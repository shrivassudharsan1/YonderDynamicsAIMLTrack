"""Hard example mining: which TRAINING objects is the model still unsure about or wrong on?

    python mine_hard_examples.py --weights weights/best.pt

The model has already seen these images, so anything it still misses or scores low is
either genuinely hard, rare in the data, or mislabelled. For every labelled object in
data/grouped/train the script records the confidence of the best matching prediction
(same class, IoU >= 0.5; 0 if there is none), then groups the results by object size
and by scene to show where more data would help most.

Writes results/hard_examples/: summary.md, objects.csv (every object, hardest first)
and hardest.jpg (the lowest-scoring images; green = label, red = prediction).
"""

import argparse
import os
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from analyze_errors import IOU_MATCH, SIZE_BINS, draw, iou, load_labels, save_sheet, size_bin, xywh_to_xyxy

ROOT = Path(__file__).parent
TRAIN = ROOT / "data" / "grouped" / "train" / "images"
MANIFEST = ROOT / "splits" / "grouped_split.csv"
OUT = ROOT / "results" / "hard_examples"
NAMES = {0: "bottle", 1: "mallet"}
DEPLOY_CONF = 0.25


def summarize(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row["confidence"])
    out = []
    for name, confs in groups.items():
        confs = np.array(confs)
        out.append((name, len(confs), float(confs.mean()), int((confs < DEPLOY_CONF).sum())))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights" / "best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--top", type=int, default=24, help="images on the sheet")
    args = parser.parse_args()

    with MANIFEST.open() as f:
        scene_of = {row["image"]: row["scene"] for row in csv.DictReader(f)}
    model = YOLO(str(args.weights))
    OUT.mkdir(parents=True, exist_ok=True)

    rows, per_image = [], {}
    for image_path in sorted(TRAIN.iterdir()):
        labels = load_labels(image_path)
        # Very low threshold so every object gets its best available confidence.
        result = model.predict(str(image_path), conf=0.01, iou=0.5, imgsz=args.imgsz, verbose=False)[0]
        preds = [(int(c), xywh_to_xyxy(b), float(s)) for c, b, s in zip(result.boxes.cls.tolist(), result.boxes.xywhn.tolist(), result.boxes.conf.tolist())]
        worst = 1.0
        for cls, box in labels:
            confidence = max((s for c, b, s in preds if c == cls and iou(box, b) >= IOU_MATCH), default=0.0)
            worst = min(worst, confidence)
            rows.append({"image": image_path.name, "scene": scene_of[image_path.name], "class": NAMES[cls],
                         "size": size_bin(box), "confidence": confidence})
        per_image[image_path] = (worst, labels, [p for p in preds if p[2] >= DEPLOY_CONF])

    rows.sort(key=lambda r: r["confidence"])
    with (OUT / "objects.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "scene", "class", "size", "confidence"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "confidence": f"{row['confidence']:.3f}"})

    missed = sum(r["confidence"] < DEPLOY_CONF for r in rows)
    lines = [
        f"Weights `{os.path.relpath(args.weights, ROOT)}`, {len(per_image)} training images, {len(rows)} labelled objects.",
        f"{missed} objects ({missed / len(rows):.1%}) would be missed at the deployment threshold "
        f"(best matching confidence < {DEPLOY_CONF}), on images the model was trained on.",
        "",
        "By class and object size (objects below the threshold / objects, mean confidence):",
        "",
        "| class | " + " | ".join(name for *_, name in SIZE_BINS) + " |",
        "|---|" + "---|" * len(SIZE_BINS),
    ]
    for cls in NAMES.values():
        cells = []
        for *_, size in SIZE_BINS:
            confs = np.array([r["confidence"] for r in rows if r["class"] == cls and r["size"] == size])
            cells.append(f"{int((confs < DEPLOY_CONF).sum())} / {len(confs)}, {confs.mean():.2f}" if len(confs) else "-")
        lines.append(f"| {cls} | " + " | ".join(cells) + " |")

    scenes = sorted(summarize(rows, "scene"), key=lambda s: s[2])
    lines += ["", "Hardest scenes (at least 5 objects), lowest mean confidence first:", "",
              "| scene | objects | mean confidence | below threshold |", "|---|---|---|---|"]
    for name, n, mean, low in [s for s in scenes if s[1] >= 5][:10]:
        lines.append(f"| {name} | {n} | {mean:.2f} | {low} |")

    summary = "\n".join(lines)
    print(summary)
    (OUT / "summary.md").write_text(summary + "\n")

    hardest = sorted(per_image.items(), key=lambda item: item[1][0])[: args.top]
    tiles = [draw(path, labels, preds, NAMES, f"scene {scene_of[path.name]}  worst conf {worst:.2f}")
             for path, (worst, labels, preds) in hardest]
    save_sheet(tiles, OUT / "hardest.jpg", limit=args.top)
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
