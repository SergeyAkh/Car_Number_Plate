import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def get_latest_run(base_dir=None):
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"{base_dir} does not exist")

    runs = [os.path.join(base_dir, d) for d in os.listdir(base_dir)]
    runs = [d for d in runs if os.path.isdir(d)]

    if not runs:
        raise FileNotFoundError("No YOLO runs found")

    latest = max(runs, key=os.path.getmtime)
    best_model_path = os.path.join(latest, "weights", "best.pt")

    return best_model_path

def get_latest_model_new(base_dir):
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"{base_dir} does not exist")

    candidates = []

    for root, dirs, files in os.walk(base_dir):
        # YOLO best.pt
        if "weights" in root and "best.pt" in files:
            candidates.append(os.path.join(root, "best.pt"))

        # Standard PyTorch models
        for f in files:
            if f.endswith(".pth"):
                candidates.append(os.path.join(root, f))

    if not candidates:
        raise FileNotFoundError("No model (.pt or .pth) found")

    latest = max(candidates, key=os.path.getmtime)
    return latest

def plot_two_images_with_text(img1, img2, text,
                              text_pos=(0.02, 0.98)):
    """
    img1, img2: numpy arrays (H×W×C or H×W)
    text: string
    text_pos: (x, y) in axes coords (0–1)
    """
    # Convert PIL images to numpy
    if not isinstance(img1, np.ndarray):
        img1 = np.array(img1)
    if not isinstance(img2, np.ndarray):
        img2 = np.array(img2)

    fig, axes = plt.subplots(2, 1, figsize=(6, 8))

    # First image
    axes[0].imshow(img1)
    axes[0].axis('off')

    # Second image
    axes[1].imshow(img2, cmap='gray' if img2.ndim == 2 else None)
    axes[1].axis('off')

    fig.text(
        0.5, 0.01,
        f"Plate number is: {text}",
        ha='right',
        va='bottom',
        fontsize=13,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )

    plt.tight_layout()
    plt.show()

def predict_and_crop(model, source, crop_dir = None, train_val_single = None, save_crop = None):
    """
    Runs prediction and crops all detections.
    Works for: single image OR any folder of images.
    """
    # Run prediction
    results = model.predict(source, save=False, save_crop=False, verbose=False)

    if train_val_single is not None:
        crop_dir = os.path.join(crop_dir, train_val_single, "images")
        os.makedirs(crop_dir, exist_ok=True)

    for r in results:
        img = Image.fromarray(r.orig_img)
        img_name = os.path.splitext(os.path.basename(r.path))[0]

        for i, box in enumerate(r.boxes[0]):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            crop = img.crop((x1, y1, x2, y2))
            if save_crop:
                crop_path = os.path.join(crop_dir, f"{img_name}_crop_{i}.jpg")
                crop.save(crop_path)
                print(f"Crops saved to: {crop_dir}")
    if train_val_single is None:
        return crop
