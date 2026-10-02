#!/usr/bin/env bash
# Run the whole pipeline: download, split, preprocess, (train), evaluate, analyse.
#
#   bash run_all.sh              # evaluate the committed weights (weights/best.pt), no training
#   bash run_all.sh --train      # also retrain the final model first (about 25 min on an M4 Max GPU)
#
# Needs ROBOFLOW_API_KEY in the environment or in .env (see .env.example).
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
[ -x "$PY" ] || PY=python
WEIGHTS=weights/best.pt

echo "== 1/7 download dataset";            $PY download_data.py
echo "== 2/7 build scene-grouped split";   $PY make_split.py
echo "== 3/7 build colour-augmented set";  $PY preprocess.py

if [ "${1:-}" = "--train" ]; then
  echo "== 4/7 train (50 epochs)"
  $PY train.py --name coloraug_grouped_50 --data data_grouped_coloraug.yaml --epochs 50
  WEIGHTS=runs/coloraug_grouped_50/weights/best.pt
else
  echo "== 4/7 training skipped, using $WEIGHTS"
fi

echo "== 5/7 per-class metrics + colour stress test";  $PY evaluate.py --weights "$WEIGHTS" --data data_grouped.yaml
echo "== 6/7 error analysis";                          $PY analyze_errors.py --weights "$WEIGHTS" --data data_grouped.yaml
echo "== 7/7 inference on the validation images";      $PY predict.py data/grouped/valid/images predictions --weights "$WEIGHTS"
echo "Done. Detections are in predictions/, tables and figures in results/."
