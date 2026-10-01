"""Run the trained detector on a folder of images.

    python predict.py <image_folder> <output_folder> [--weights weights/best.pt] [--conf 0.25]

For every image this writes <output_folder>/<image_name>.txt with one detection per line:

    class_id x_center y_center width height confidence

Coordinates are normalized to 0-1, class ids match the dataset (0 = bottle, 1 = mallet).
An image with no detections still gets an (empty) .txt file.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_WEIGHTS = Path(__file__).parent / "weights" / "best.pt"
DEFAULT_CONF = 0.25


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image_folder", type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--save-images", action="store_true", help="also save images with boxes drawn")
    return parser.parse_args()


def main():
    args = parse_args()
    images = sorted(p for p in args.image_folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit(f"No images found in {args.image_folder}")
    args.output_folder.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    total = 0
    for image_path in images:
        result = model.predict(str(image_path), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        lines = []
        for cls, (x, y, w, h), conf in zip(
            result.boxes.cls.tolist(), result.boxes.xywhn.tolist(), result.boxes.conf.tolist()
        ):
            lines.append(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.4f}")
        (args.output_folder / f"{image_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        if args.save_images:
            result.save(filename=str(args.output_folder / f"{image_path.stem}_pred.jpg"))
        total += len(lines)

    print(f"{len(images)} images, {total} detections (conf >= {args.conf}) -> {args.output_folder}")


if __name__ == "__main__":
    main()
