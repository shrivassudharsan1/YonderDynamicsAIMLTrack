"""Download the take-home dataset from Roboflow into Sampled-YD-Object-Detection-2/.

Reads the API key from the ROBOFLOW_API_KEY environment variable (or a git-ignored
.env file). The key is never written into this file.

    python download_data.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

WORKSPACE = "malletbottle2"
PROJECT = "sampled-yd-object-detection"
VERSION = 2
DATASET_DIR = Path(__file__).parent / "Sampled-YD-Object-Detection-2"


def main():
    if (DATASET_DIR / "data.yaml").exists():
        print(f"{DATASET_DIR.name}/ already exists, skipping download.")
        return

    load_dotenv()
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit("Set ROBOFLOW_API_KEY in your environment or in a .env file (see .env.example).")

    rf = Roboflow(api_key=api_key)
    version = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
    version.download("yolov8", location=str(DATASET_DIR))


if __name__ == "__main__":
    main()
