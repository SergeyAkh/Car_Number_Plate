from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset.license_plate_dataset import LicensePlateDataset

from config import IMAGES_DIR, LABELS_DIR

def create_dataloader():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = LicensePlateDataset(
        img_dir=IMAGES_DIR,
        label_dir=LABELS_DIR,
        transforms=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )

    return train_loader