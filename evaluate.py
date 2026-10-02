"""Evaluate trained weights per class, on the validation set and on colour-stressed copies of it.

    python evaluate.py --weights runs/baseline_grouped/weights/best.pt --data data_grouped.yaml

Three versions of the same validation images are scored:
  original   the images as they are
  gray       colour removed (3-channel grayscale), so only shape/texture is left
  hue_shift  hue rotated by 90 degrees (orange becomes green), shape unchanged

A big drop on gray/hue_shift means the model is recognising the object by its colour.
The stressed copies are written to data/stress/ and the table to runs/<run>/eval.md.
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).parent
STRESS = ROOT / "data" / "stress"


def to_gray(image):
    return cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def hue_shift(image, degrees=90):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = (hsv[..., 0].astype(np.int32) + degrees // 2) % 180  # OpenCV hue is 0-179
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


VARIANTS = {"original": None, "gray": to_gray, "hue_shift": hue_shift}


def build_variant(name, transform, data_cfg, data_yaml):
    """Return a data yaml whose val set is the (transformed) validation images."""
    if transform is None:
        return data_yaml
    val_images = (data_yaml.parent / data_cfg["val"]).resolve()
    out = STRESS / data_yaml.stem / name
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    shutil.copytree(val_images.parent / "labels", out / "labels")
    for image_path in sorted(val_images.iterdir()):
        cv2.imwrite(str(out / "images" / image_path.name), transform(cv2.imread(str(image_path))))
    variant_yaml = out / "data.yaml"
    variant_yaml.write_text(
        yaml.safe_dump({"train": str(out / "images"), "val": str(out / "images"), "nc": data_cfg["nc"], "names": data_cfg["names"]})
    )
    return variant_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights" / "best.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "data_grouped.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    data_yaml = args.data.resolve()
    data_cfg = yaml.safe_load(data_yaml.read_text())
    model = YOLO(str(args.weights))

    lines = [
        f"Weights: `{args.weights}`  |  data: `{args.data.name}`  |  imgsz {args.imgsz}",
        "",
        "Precision/recall are at the confidence that maximises F1 for that class (Ultralytics default).",
        "",
        "| val images | class | boxes | precision | recall | AP50 | AP50-95 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, transform in VARIANTS.items():
        variant_yaml = build_variant(name, transform, data_cfg, data_yaml)
        m = model.val(data=str(variant_yaml), imgsz=args.imgsz, plots=False, verbose=False,
                      project=str(ROOT / "runs" / "eval"), name=name, exist_ok=True)
        for i, c in enumerate(m.box.ap_class_index):
            lines.append(
                f"| {name} | {m.names[int(c)]} | {int(m.nt_per_class[c])} | {m.box.p[i]:.3f} | {m.box.r[i]:.3f} | {m.box.ap50[i]:.3f} | {m.box.ap[i]:.3f} |"
            )
        lines.append(f"| {name} | **all** | | {m.box.mp:.3f} | {m.box.mr:.3f} | {m.box.map50:.3f} | {m.box.map:.3f} |")

    table = "\n".join(lines)
    print("\n" + table)
    run_dir = args.weights.resolve().parent.parent
    if run_dir.parent.name == "runs":
        (run_dir / "eval.md").write_text(table + "\n")
        print(f"\nsaved to {run_dir / 'eval.md'}")


if __name__ == "__main__":
    main()
