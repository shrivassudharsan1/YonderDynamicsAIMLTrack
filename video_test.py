"""Real-world test: run the detector on frames sampled from phone videos.

    python video_test.py IMG_8112.MOV IMG_8113.MOV IMG_8114.MOV --weights weights/best.pt --tag final

Two frames per second are extracted to video_frames/<video>/ (git-ignored), the model is
run with the same settings as predict.py, and for each video this reports how many frames
had at least one mallet / bottle detection. There are no labels for these frames, so
whether a box is on the right object has to be checked by eye on the saved sheets in
results/video/.

--stretch squashes each frame to a 512x512 square first, the way Roboflow resized the
training images ("Resize: Stretch"), instead of keeping the phone's aspect ratio.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).parent
FRAMES = ROOT / "video_frames"
OUT = ROOT / "results" / "video"
NAMES = {0: "bottle", 1: "mallet"}
COLORS = {0: (255, 120, 0), 1: (0, 0, 255)}


def extract_frames(video, per_second):
    out = FRAMES / video.stem
    out.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))  # OpenCV applies the phone's rotation metadata
    step = max(1, round(capture.get(cv2.CAP_PROP_FPS) / per_second))
    frames, index = [], 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            path = out / f"{video.stem}_{index:05d}.jpg"
            cv2.imwrite(str(path), frame)
            frames.append(path)
        index += 1
    return frames


def tile(frame, boxes, label):
    h, w = frame.shape[:2]
    for cls, (x1, y1, x2, y2), conf in boxes:
        cv2.rectangle(frame, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), COLORS[cls], max(3, w // 150))
        cv2.putText(frame, f"{NAMES[cls]} {conf:.2f}", (int(x1 * w), max(30, int(y1 * h) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, w / 500, COLORS[cls], max(2, w // 300))
    frame = cv2.resize(frame, (216, 384))
    cv2.putText(frame, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights" / "best.pt")
    parser.add_argument("--tag", default="final", help="name used in the output files")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--per-second", type=float, default=2.0)
    parser.add_argument("--stretch", action="store_true", help="squash frames to 512x512 like the training images")
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Weights `{args.weights}`, conf >= {args.conf}, NMS IoU {args.iou}, "
        f"{'frames stretched to 512x512' if args.stretch else 'frames at phone aspect ratio'}",
        "",
        "| video | frames | with a mallet box | with a bottle box | with no box | mean top mallet conf | mean top bottle conf |",
        "|---|---|---|---|---|---|---|",
    ]
    for video in args.videos:
        frames = extract_frames(video, args.per_second)
        if not frames:
            raise SystemExit(f"Could not read any frames from {video}")
        tiles, hits, top = [], {0: 0, 1: 0}, {0: [], 1: []}
        empty = 0
        for i, path in enumerate(frames):
            frame = cv2.imread(str(path))
            source = cv2.resize(frame, (512, 512)) if args.stretch else frame
            result = model.predict(source, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)[0]
            boxes = [(int(c), b, float(s)) for c, b, s in zip(result.boxes.cls.tolist(), result.boxes.xyxyn.tolist(), result.boxes.conf.tolist())]
            for cls in NAMES:
                confs = [s for c, _, s in boxes if c == cls]
                if confs:
                    hits[cls] += 1
                    top[cls].append(max(confs))
            empty += not boxes
            tiles.append(tile(frame, boxes, f"{i / args.per_second:.1f}s"))

        mean = lambda v: f"{np.mean(v):.2f}" if v else "-"
        lines.append(f"| {video.name} | {len(frames)} | {hits[1]} | {hits[0]} | {empty} | {mean(top[1])} | {mean(top[0])} |")
        while len(tiles) % 8:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i : i + 8]) for i in range(0, len(tiles), 8)])
        cv2.imwrite(str(OUT / f"{video.stem}_{args.tag}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])

    summary = "\n".join(lines)
    print(summary)
    (OUT / f"summary_{args.tag}.md").write_text(summary + "\n")


if __name__ == "__main__":
    main()
