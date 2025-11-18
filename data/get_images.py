import kagglehub
import shutil
import os

import sys
from config import data_path

target_dir = data_path  # your path

# If directory exists AND contains files → abort
if os.path.exists(target_dir) and os.listdir(target_dir):
    print(f"Error: Target directory '{target_dir}' already exists and is not empty.")
    sys.exit(1)

# Download
src_path = kagglehub.dataset_download(
    "harshitsingh09/license-plate-detection-dataset-anpr-yolo-format"
)

# Otherwise, ensure directory exists (empty or new)
os.makedirs(target_dir, exist_ok=True)

# Move the dataset
shutil.move(src_path, target_dir)

print("Moved to:", target_dir)


def load_images_dataset(path):
    """Downloads and returns the Heart Failure dataset as a Pandas DataFrame."""

    # Download latest version
    path = kagglehub.dataset_download("harshitsingh09/license-plate-detection-dataset-anpr-yolo-format")

    print("Path to dataset files:", path)

    return df, X, y