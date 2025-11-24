import os
from pathlib import Path

# Project root (resolved automatically)
ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT/"data"
IMAGES_DIR = os.path.join(ROOT, "images")
LABELS_DIR = os.path.join(ROOT, "labels")

# path to dataset YAML used by Ultralytics
DATASET_YAML = DATA_DIR / "dataset.yaml"

# where to save weights & logs
OUTPUT_DIR = ROOT / "runs" / "yolo"

# sample uploaded dataset cover (you provided this file)
DATASET_COVER = Path("/mnt/data/dataset-cover.png")