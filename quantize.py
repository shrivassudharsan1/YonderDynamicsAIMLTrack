"""Export the model to ONNX and quantize it to 8-bit integers, then measure what that costs.

    python quantize.py --weights weights/best.pt

The rover's NPU runs 8-bit integer models, so this checks how much accuracy survives the
conversion. It does NOT produce an RKNN file and nothing here runs on the NPU.

Three models are scored on the same validation set with the same code:
  fp32          ONNX export, 32-bit floats (should match the PyTorch model)
  int8 all      every layer quantized, including the box-decoding maths at the end
  int8 backbone convolutions quantized, box decoding in the detection head left in float

Calibration (choosing the 8-bit ranges) uses training images only.
Models go to runs/quantized/ (git-ignored), the table to results/quantization.md.
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from ultralytics import YOLO

ROOT = Path(__file__).parent
OUT = ROOT / "runs" / "quantized"
CALIBRATION_IMAGES = ROOT / "data" / "grouped" / "train" / "images"
HEAD = "/model.22/"  # YOLOv8 detection head


class Calibration(CalibrationDataReader):
    def __init__(self, images, imgsz):
        self.batches = iter(images)
        self.imgsz = imgsz

    def get_next(self):
        path = next(self.batches, None)
        if path is None:
            return None
        image = cv2.cvtColor(cv2.resize(cv2.imread(str(path)), (self.imgsz, self.imgsz)), cv2.COLOR_BGR2RGB)
        return {"images": image.transpose(2, 0, 1)[None].astype(np.float32) / 255.0}


def box_decoding_nodes(model_path):
    """Head nodes that turn raw outputs into pixel boxes: everything in the head except
    its ordinary convolutions. Box coordinates (0-640) and class scores (0-1) share one
    output tensor, which a single 8-bit scale cannot represent well."""
    graph = onnx.load(str(model_path)).graph
    return [n.name for n in graph.node if n.name.startswith(HEAD) and (n.op_type != "Conv" or "dfl" in n.name)]


def score(model_path, data, imgsz):
    metrics = YOLO(str(model_path), task="detect").val(
        data=str(data), imgsz=imgsz, batch=1, device="cpu", plots=False, verbose=False,
        project=str(ROOT / "runs" / "eval"), name=f"quant_{model_path.stem}", exist_ok=True)
    return metrics.box.ap50[0], metrics.box.ap50[1], metrics.box.map50, metrics.box.map


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights" / "best.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "data_grouped.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--n-calibration", type=int, default=200)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    exported = Path(YOLO(str(args.weights)).export(format="onnx", imgsz=args.imgsz, opset=17, simplify=True))
    fp32 = OUT / "fp32.onnx"
    exported.replace(fp32)
    prepared = OUT / "fp32_prepared.onnx"
    quant_pre_process(str(fp32), str(prepared))

    images = sorted(CALIBRATION_IMAGES.iterdir())
    random.Random(0).shuffle(images)
    images = images[: args.n_calibration]

    variants = {"int8 all": [], "int8 backbone": box_decoding_nodes(prepared)}
    models = {"fp32": fp32}
    for name, excluded in variants.items():
        path = OUT / f"{name.replace(' ', '_')}.onnx"
        quantize_static(
            str(prepared), str(path), Calibration(images, args.imgsz),
            quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
            per_channel=True, nodes_to_exclude=excluded,
        )
        models[name] = path

    data = args.data.resolve()
    lines = [
        f"Source weights `{os.path.relpath(args.weights, ROOT)}` ({args.weights.stat().st_size / 1e6:.1f} MB), "
        f"input {args.imgsz}, calibrated on {len(images)} training images, scored on `{args.data.name}` with ONNX Runtime on CPU.",
        "",
        "| model | file size MB | bottle AP50 | mallet AP50 | mAP50 | mAP50-95 |",
        "|---|---|---|---|---|---|",
    ]
    for name, path in models.items():
        bottle, mallet, map50, map5095 = score(path, data, args.imgsz)
        lines.append(f"| {name} | {path.stat().st_size / 1e6:.1f} | {bottle:.3f} | {mallet:.3f} | {map50:.3f} | {map5095:.3f} |")

    table = "\n".join(lines)
    print(table)
    (ROOT / "results" / "quantization.md").write_text(table + "\n")


if __name__ == "__main__":
    main()
