import os
from pathlib import Path
import torch

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(5 % 5)
# Project root (resolved automatically)
ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

DATA_DIR = ROOT/"data"

# Base project directory (absolute path)
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

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

PATH_TO_TRAIN_PLATE = os.path.join(BASE_DIR, "src", "training", "runs", "detect")
PATH_CROP_PLATE = os.path.join(BASE_DIR, "predictions", "crops")

PATH_TO_CHAR_MODEL = os.path.join(BASE_DIR, "src", "training", "runs", "characters")
synt_path = os.path.join(BASE_DIR, "data", "3", "synthetic")

charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Model config
MODEL_CONFIG = {
    "DEVICE": device,
    "FOLDER": PATH_TO_TRAIN_PLATE,
    "img_size": 640,
    "batch_size": 16,
    "epochs": 10,
    "model_name": "yolov8n",
}

CHAR_MODEL_CONFIG = {
    "DEVICE": device,
    "FOLDER": PATH_TO_CHAR_MODEL,
    "img_h" : 32,
    "synt_path" : synt_path,
    "crop_data" : PATH_CROP_PLATE,
    "batch_size" : 8,
    "lr" : 5e-5,
    "weight_decay": 1e-5,
    "epochs" : 100,
    "num_classes" : len(charset) + 1,
}


CRNN_config = {
    "DEVICE": device
}

print("Loaded config:")
for key, value in MODEL_CONFIG.items():
    print(f"{key}: {value}")