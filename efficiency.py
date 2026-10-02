"""Model size, speed and the accuracy/speed trade-off of input resolution.

    python efficiency.py --weights weights/best.pt --data data_grouped.yaml

Reports parameter count, GFLOPs and file size, then for several input sizes the
validation mAP and the per-image inference time on CPU (and on the GPU if there is one).
Timings are for this machine, not the rover's NPU: read them as relative, not absolute.
Writes results/efficiency.md.
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

ROOT = Path(__file__).parent
SIZES = [320, 416, 512, 640]


def latency_ms(model, images, imgsz, device, warmup=10):
    """Mean wall-clock time per image (batch 1) for the whole predict call:
    resize, forward pass and NMS. Wall clock is used because the Apple GPU runs
    asynchronously, so Ultralytics' own forward-pass timer under-reports it."""
    frames = [cv2.imread(str(image)) for image in images]
    for frame in frames[:warmup]:
        model.predict(frame, imgsz=imgsz, device=device, verbose=False)
    start = time.perf_counter()
    for frame in frames:
        model.predict(frame, imgsz=imgsz, device=device, verbose=False)
    return (time.perf_counter() - start) * 1000 / len(frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights" / "best.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "data_grouped.yaml")
    parser.add_argument("--n-images", type=int, default=100, help="images used for timing")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.data.read_text())
    images = sorted((args.data.resolve().parent / cfg["val"]).iterdir())[: args.n_images]
    gpu = "0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None

    model = YOLO(str(args.weights))
    params = sum(p.numel() for p in model.model.parameters())
    gflops = get_flops(model.model, 640)
    lines = [
        f"Weights `{os.path.relpath(args.weights, ROOT)}`: {params:,} parameters, {gflops:.1f} GFLOPs at 640, "
        f"{args.weights.stat().st_size / 1e6:.1f} MB on disk (FP16 checkpoint)",
        "",
        f"Timing: mean wall-clock time per image for the full predict call (resize + forward pass + NMS), batch 1, {len(images)} validation images, torch {torch.__version__}.",
        "",
        "| input size | bottle AP50 | mallet AP50 | mAP50 | mAP50-95 | CPU ms/image | CPU FPS |" + (" GPU ms/image |" if gpu else ""),
        "|---|---|---|---|---|---|---|" + ("---|" if gpu else ""),
    ]
    for imgsz in SIZES:
        metrics = YOLO(str(args.weights)).val(data=str(args.data.resolve()), imgsz=imgsz, plots=False, verbose=False,
                                             project=str(ROOT / "runs" / "eval"), name=f"size{imgsz}", exist_ok=True)
        cpu = latency_ms(YOLO(str(args.weights)), images, imgsz, "cpu")
        row = (f"| {imgsz} | {metrics.box.ap50[0]:.3f} | {metrics.box.ap50[1]:.3f} | {metrics.box.map50:.3f} | "
               f"{metrics.box.map:.3f} | {cpu:.1f} | {1000 / cpu:.0f} |")
        if gpu:
            row += f" {latency_ms(YOLO(str(args.weights)), images, imgsz, gpu):.1f} |"
        lines.append(row)

    table = "\n".join(lines)
    print(table)
    (ROOT / "results" / "efficiency.md").write_text(table + "\n")


if __name__ == "__main__":
    main()
