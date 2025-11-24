import os
from pathlib import Path

# Project root (resolved automatically)
ROOT = Path(__file__).resolve().parent
print(ROOT)
DATA_DIR = ROOT/"data"

# Base project directory (absolute path)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to dataset
DATASET_DIR = os.path.join(BASE_DIR, "data", "3", "YOLO_dataset")

# YOLO-specific paths
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

TRAIN_IMAGES = os.path.join(IMAGES_DIR, "train")
VAL_IMAGES = os.path.join(IMAGES_DIR, "val")

TRAIN_LABELS = os.path.join(LABELS_DIR, "train")
VAL_LABELS = os.path.join(LABELS_DIR, "val")

# Path to data.yaml for YOLO
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")


# Model config
MODEL_CONFIG = {
    "img_size": 640,
    "batch_size": 16,
    "epochs": 1,
    "model_name": "yolov8n",  # can be yolov8s / yolov8m etc.
}

print("Loaded config:")
for key, value in MODEL_CONFIG.items():
    print(f"{key}: {value}")