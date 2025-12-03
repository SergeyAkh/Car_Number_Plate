import os
import cv2
import torch
from torch.utils.data import Dataset

from src.models.model_utils import preprocess_image, encode_text, decode_text


class LicensePlateOCRDataset(Dataset):
    def __init__(self, root_dir, img_h=32):
        self.image_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")
        self.img_h = img_h

        self.samples = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_name)

        label_path = os.path.join(
            self.label_dir, img_name.replace(".jpg", ".txt")
        )

        # load image
        img = preprocess_image(img_path, img_h=self.img_h)

        # load label
        with open(label_path, "r") as f:
            text = f.read().strip()

        label_enc = encode_text(text)

        return img, label_enc, text

def ocr_collate(batch):
    images = []
    labels = []

    for item in batch:
        # Ensure shape [C, H, W]
        img = item[0]
        label = item[1]
        if img.ndim == 2:
            img = img.unsqueeze(0)         # [1, H, W]
        elif img.ndim == 3 and img.shape[0] != 1:
            img = img.permute(2, 0, 1)     # [1, H, W]
        images.append(img)
        labels.append(label)

    # pad to max width
    max_w = max(img.shape[-1] for img in images)

    padded = []
    for img in images:
        pad_w = max_w - img.shape[-1]
        padded_img = torch.nn.functional.pad(img, (0, pad_w, 0, 0))     # pad width only
        padded.append(padded_img)

    images = torch.stack(padded)   # [B, C, H, W]

    # labels
    targets = torch.cat(labels)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)

    return images, targets, target_lengths