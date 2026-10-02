"""Error analysis at the deployment confidence threshold (the one predict.py uses).

    python analyze_errors.py --weights weights/best.pt --data data_grouped.yaml

Every prediction and every labelled box on the validation set gets one outcome:

  TP              prediction overlaps a same-class label with IoU >= 0.5
  FP wrong class  overlaps a label of the OTHER class with IoU >= 0.5
  FP loose box    overlaps a same-class label, but only with IoU 0.1-0.5
  FP duplicate    a second box on an object that is already matched
  FP background   overlaps nothing labelled
  FN              a labelled box no prediction matched

Writes results/errors/summary.md plus image sheets of the misses and false alarms
(green = label, red = prediction).
"""

import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "errors"
IOU_MATCH = 0.5
IOU_LOOSE = 0.1
SIZE_BINS = [(0, 0.002, "tiny (<0.2% of image)"), (0.002, 0.01, "small (0.2-1%)"), (0.01, 0.05, "medium (1-5%)"), (0.05, 2, "large (>5%)")]


def xywh_to_xyxy(b):
    x, y, w, h = b
    return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def size_bin(box):
    area = (box[2] - box[0]) * (box[3] - box[1])
    return next(name for lo, hi, name in SIZE_BINS if lo <= area < hi)


def load_labels(image_path):
    label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    labels = []
    for line in label_path.read_text().splitlines():
        if line.strip():
            c, *box = line.split()
            labels.append((int(c), xywh_to_xyxy([float(v) for v in box])))
    return labels


def classify(labels, preds):
    """Return (prediction outcomes, matched flags for labels). preds sorted by confidence."""
    matched = [False] * len(labels)
    outcomes = []
    for cls, box, conf in preds:
        ious = [iou(box, lb) for _, lb in labels]
        same = [(v, i) for i, v in enumerate(ious) if labels[i][0] == cls]
        free = [(v, i) for v, i in same if not matched[i] and v >= IOU_MATCH]
        if free:
            matched[max(free)[1]] = True
            outcomes.append("TP")
        elif any(v >= IOU_MATCH for v, _ in same):
            outcomes.append("FP duplicate")
        elif any(v >= IOU_MATCH for i, v in enumerate(ious) if labels[i][0] != cls):
            outcomes.append("FP wrong class")
        elif any(v >= IOU_LOOSE for v, _ in same):
            outcomes.append("FP loose box")
        else:
            outcomes.append("FP background")
    return outcomes, matched


def draw(image_path, labels, preds, names, title):
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    for cls, box in labels:
        p1, p2 = (int(box[0] * w), int(box[1] * h)), (int(box[2] * w), int(box[3] * h))
        cv2.rectangle(image, p1, p2, (0, 220, 0), 2)
        cv2.putText(image, names[cls], (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)
    for cls, box, conf in preds:
        p1, p2 = (int(box[0] * w), int(box[1] * h)), (int(box[2] * w), int(box[3] * h))
        cv2.rectangle(image, p1, p2, (0, 0, 255), 2)
        cv2.putText(image, f"{names[cls]} {conf:.2f}", (p1[0], min(h - 4, p2[1] + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    image = cv2.resize(image, (384, 384))
    cv2.rectangle(image, (0, 0), (384, 20), (0, 0, 0), -1)
    cv2.putText(image, title, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return image


def save_sheet(tiles, path, columns=4, limit=24):
    tiles = tiles[:limit]
    if not tiles:
        return
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + columns]) for i in range(0, len(tiles), columns)]
    cv2.imwrite(str(path), np.vstack(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights" / "best.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "data_grouped.yaml")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.5, help="NMS IoU threshold, same default as predict.py")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.data.read_text())
    names = cfg["names"]
    images = sorted((args.data.resolve().parent / cfg["val"]).iterdir())
    model = YOLO(str(args.weights))
    args.out.mkdir(parents=True, exist_ok=True)

    outcome_counts = defaultdict(Counter)  # class -> outcome -> n
    fn_by_size = defaultdict(Counter)      # class -> size bin -> [missed]
    gt_by_size = defaultdict(Counter)
    sheets = defaultdict(list)
    rows = []

    for image_path in images:
        labels = load_labels(image_path)
        result = model.predict(str(image_path), conf=args.conf, iou=args.nms_iou, imgsz=args.imgsz, verbose=False)[0]
        preds = sorted(
            ((int(c), xywh_to_xyxy(b), float(s)) for c, b, s in zip(result.boxes.cls.tolist(), result.boxes.xywhn.tolist(), result.boxes.conf.tolist())),
            key=lambda p: -p[2],
        )
        outcomes, matched = classify(labels, preds)

        for (cls, _, conf), outcome in zip(preds, outcomes):
            outcome_counts[cls][outcome] += 1
            if outcome != "TP":
                rows.append((image_path.name, names[cls], outcome, f"{conf:.2f}"))
        for (cls, box), hit in zip(labels, matched):
            gt_by_size[cls][size_bin(box)] += 1
            if not hit:
                outcome_counts[cls]["FN"] += 1
                fn_by_size[cls][size_bin(box)] += 1
                rows.append((image_path.name, names[cls], "FN", ""))

        missed = sorted({names[c] for (c, _), hit in zip(labels, matched) if not hit})
        false_alarms = sorted({o for o in outcomes if o != "TP"})
        if missed:
            sheets["false_negatives"].append(draw(image_path, labels, preds, names, "missed: " + ", ".join(missed)))
        if false_alarms:
            sheets["false_positives"].append(draw(image_path, labels, preds, names, ", ".join(false_alarms)))

    lines = [f"Weights `{os.path.relpath(args.weights, ROOT)}`, data `{args.data.name}`, {len(images)} images, confidence >= {args.conf}, NMS IoU {args.nms_iou}, match IoU >= {IOU_MATCH}", ""]
    lines += ["| class | labelled | TP | FN | FP | precision | recall |", "|---|---|---|---|---|---|---|"]
    for cls, name in names.items():
        c = outcome_counts[cls]
        tp, fn = c["TP"], c["FN"]
        fp = sum(v for k, v in c.items() if k.startswith("FP"))
        lines.append(f"| {name} | {tp + fn} | {tp} | {fn} | {fp} | {tp / max(tp + fp, 1):.3f} | {tp / max(tp + fn, 1):.3f} |")
    lines += ["", "False positives by cause:", "", "| class | background | wrong class | loose box | duplicate |", "|---|---|---|---|---|"]
    for cls, name in names.items():
        c = outcome_counts[cls]
        lines.append(f"| {name} | {c['FP background']} | {c['FP wrong class']} | {c['FP loose box']} | {c['FP duplicate']} |")
    lines += ["", "Misses by object size (missed / labelled):", "", "| class | " + " | ".join(n for *_, n in SIZE_BINS) + " |", "|---|" + "---|" * len(SIZE_BINS)]
    for cls, name in names.items():
        lines.append(f"| {name} | " + " | ".join(f"{fn_by_size[cls][n]} / {gt_by_size[cls][n]}" for *_, n in SIZE_BINS) + " |")

    summary = "\n".join(lines)
    print(summary)
    (args.out / "summary.md").write_text(summary + "\n")
    (args.out / "errors.csv").write_text("image,class,outcome,confidence\n" + "\n".join(",".join(r) for r in rows) + "\n")
    for name, tiles in sheets.items():
        save_sheet(tiles, args.out / f"{name}.jpg")
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
