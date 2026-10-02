"""Train a YOLO detector on the mallet/bottle dataset.

    python train.py --name baseline --epochs 40

Results (weights, curves, per-epoch metrics) go to runs/<name>/.
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).parent


def pick_device():
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="baseline", help="run name, results go to runs/<name>")
    parser.add_argument("--model", default="yolov8n.pt", help="pretrained checkpoint to start from")
    parser.add_argument("--data", default=str(ROOT / "data.yaml"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=pick_device())
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
