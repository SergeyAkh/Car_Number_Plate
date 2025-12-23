import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from config import charset

class OCRFileDataset(Dataset):
    def __init__(self, images_dir, label_dir, transform=None, img_h=32, min_text_length = 1, charset = charset):
        self.images_dir = images_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_h = img_h
        self.charset = charset

        img_files = [f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")]

        self.samples = []

        for img_name in img_files:
            label_path = os.path.join(
                self.label_dir, img_name.replace(".jpg", ".txt")
            )
            if os.path.exists(label_path):
                try:
                    with open(label_path, "r") as f:
                        text = f.read().strip()
                except Exception as e:
                    print(e)
                    print(label_path)

                if len(text) >= min_text_length:
                    self.samples.append(img_name)
                else:
                    pass
                    # print(f"Warning: Skipping empty/short label: {img_name}")
            else:
                print(f"Warning: Missing label for {img_name}")

        print(f"Loaded {len(self.samples)} samples (filtered)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name = self.samples[idx]
        txt_name = os.path.splitext(img_name)[0] + ".txt"

        img_path = os.path.join(self.images_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img = img.convert("L")  # grayscale

        txt_path = os.path.join(self.label_dir, txt_name)
        with open(txt_path, "r") as f:
            text = f.read().strip().upper()

        w, h = img.size
        new_w = int(w * (self.img_h / h))
        img = img.resize((new_w, self.img_h), Image.BILINEAR)

        if self.transform is not None:
            img = self.transform(img)

        try:
            target = torch.tensor([charset.index(c) for c in text], dtype=torch.long)
        except Exception as e:
            print(e)
            print(img_name, text)

        return img, target, text

def collate_fn(batch):
    imgs, targets, texts = zip(*batch)
    widths = [img.shape[2] for img in imgs]
    H = imgs[0].shape[1]
    B = len(imgs)
    W = max(widths)  # batch width = max width

    out = torch.ones(B, 1, H, W)
    for i, img in enumerate(imgs):
        out[i, :, :, :img.shape[2]] = img

    # targets
    target_lens = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets = torch.cat(targets)


    input_lens = torch.tensor(widths, dtype=torch.long)

    return out, targets, input_lens, target_lens, texts
