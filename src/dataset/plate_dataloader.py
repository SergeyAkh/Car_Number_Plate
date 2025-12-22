import os
import cv2
import torch
from torch.utils.data import Dataset

from src.models.model_utils import preprocess_image, encode_text, decode_text

from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])


class LicensePlateOCRDataset(Dataset):
    def __init__(self, root_dir, img_h=32, min_text_length=1):
        self.image_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")
        self.img_h = img_h
        self.min_text_length = min_text_length

        self.samples = []
        all_files = sorted(os.listdir(self.image_dir))

        for img_name in all_files:
            label_path = os.path.join(
                self.label_dir, img_name.replace(".jpg", ".txt")
            )
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    text = f.read().strip()

                if len(text) >= min_text_length:
                    self.samples.append(img_name)
                else:
                    print(f"Warning: Skipping empty/short label: {img_name}")
            else:
                print(f"Warning: Missing label for {img_name}")

        print(f"Loaded {len(self.samples)} samples (filtered)")

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


# Your collate function doesn't compute input lengths properly
def ocr_collate(batch):
    images = []
    labels = []
    text_list = []

    for item in batch:
        img = item[0]
        label = item[1]
        text = item[2]

        if img.ndim == 2:
            img = img.unsqueeze(0)
        elif img.ndim == 3 and img.shape[0] != 1:
            img = img.permute(2, 0, 1)

        if torch.rand(1) > 0.5:
                img = train_transform(img)

        images.append(img)
        labels.append(label)
        text_list.append(text)

    # Find max width
    max_w = max(img.shape[-1] for img in images)

    padded = []
    for img in images:
        pad_w = max_w - img.shape[-1]
        padded_img = torch.nn.functional.pad(img, (0, pad_w, 0, 0))
        padded.append(padded_img)

    images = torch.stack(padded)  # [B, 1, H, W]

    # labels
    targets = torch.cat(labels)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)

    return images, targets, target_lengths, text_list

# Filter training data to only short plates initially
class FilteredDataset(Dataset):
    def __init__(self, original_dataset, max_length=6):
        self.original = original_dataset
        self.indices = []

        for i in range(len(original_dataset)):
            _, _, text = original_dataset[i]
            if len(text) <= max_length:
                self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.original[self.indices[idx]]