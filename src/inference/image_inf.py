from ultralytics import YOLO
from config import MODEL_CONFIG, BASE_DIR, ROOT, TRAIN_IMAGES, VAL_IMAGES, PATH_CROP_PLATE
from src.utils.utilites import get_latest_run, predict_and_crop
from src.models.character_model.char_model import TinyOCR
from src.utils.char_utils import *
import matplotlib.pyplot as plt
import torch
import os

import matplotlib
matplotlib.use("TkAgg")

charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
blank_idx = len(charset)

img_h=32
num_classes = len(charset) + 1  # blank# create model

box_model = YOLO(get_latest_run(MODEL_CONFIG["FOLDER"]))
model = TinyOCR(num_classes)
model.load_state_dict(torch.load("/Users/sergeiakhmadulin/Car_Number_Plate/src/training/runs/characters/finetuned_full_78.pth"))

img = "N49.jpeg"
original_image = Image.open(os.path.join(VAL_IMAGES, img))
plate_box = predict_and_crop(box_model, os.path.join(VAL_IMAGES, img))

img_1 = read_img(plate_box, img_h)

logits = model(img_1)

pred = logits.argmax(dim=2).squeeze(1).detach().cpu()

decoded = ctc_decode(pred, blank_idx, charset)


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
    axes[0].imshow(img1, cmap='gray' if img1.ndim == 2 else None)
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

plot_two_images_with_text(
    original_image,
    plate_box,
    text=decoded
)