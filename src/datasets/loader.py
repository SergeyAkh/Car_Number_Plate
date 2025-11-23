import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class LicensePlateDataset(Dataset):
    def __init__(self, img_dir, label_dir, transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        # Load image
        img = Image.open(img_path).convert('RGB')

        boxes = []
        labels = []

        # Load labels if exists
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    cls, xc, yc, w, h = map(float, line.split())
                    labels.append(int(cls))
                    boxes.append([xc, yc, w, h])  # Still normalized for now

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # Convert YOLO normalized boxes → absolute (x1, y1, x2, y2)
        if len(boxes) > 0:
            width, height = img.size
            abs_boxes = []
            for box in boxes:
                xc, yc, w, h = box
                x1 = (xc - w/2) * width
                y1 = (yc - h/2) * height
                x2 = (xc + w/2) * width
                y2 = (yc + h/2) * height
                abs_boxes.append([x1, y1, x2, y2])
            boxes = torch.tensor(abs_boxes, dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target


from torch.utils.data import DataLoader
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = LicensePlateDataset(
    img_dir="dataset/images",
    label_dir="dataset/labels",
    transforms=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=lambda batch: tuple(zip(*batch))  # important for detection models
)