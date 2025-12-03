import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset.plate_dataloader import LicensePlateOCRDataset, ocr_collate
import cv2
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from src.models.crnn import CRNN
from config import device, CHARSET, PATH_CROP_PLATE

from src.models.model_utils import preprocess_image, encode_text, decode_text

train_dir = os.path.join(PATH_CROP_PLATE,"train")
val_dir = os.path.join(PATH_CROP_PLATE,"val")

train_ds = LicensePlateOCRDataset(train_dir)
val_ds = LicensePlateOCRDataset(val_dir)

# batch_size=1
#
# train_dl = DataLoader(
#     train_ds, batch_size=batch_size,
#     shuffle=False, collate_fn=ocr_collate
# )
#
# images, targets, target_lengths = next(iter(train_dl))
# images = images.to(device)
# model = CRNN().to(device)
# criterion = nn.CTCLoss(blank=len(CHARSET))
#
# model.train()
#
#
# logits = model(images)         # [B, T, C]
# logits = logits.permute(1, 0, 2)
# log_probs = nn.functional.log_softmax(logits, dim=2)
#
# T, B, _ = log_probs.shape
# input_lengths = torch.full((B,), fill_value=T, dtype=torch.long)
# targets = targets.to("cpu")
# target_lengths = target_lengths.to("cpu")
# input_lengths = input_lengths.to("cpu")
# log_probs = log_probs.to("cpu")
# loss = criterion(log_probs, targets, input_lengths, target_lengths)


def train_crnn(
    train_dir,
    val_dir,
    epochs=7,
    batch_size=16,
    lr=1e-4,
    device=device
):
    train_ds = LicensePlateOCRDataset(train_dir)
    val_ds = LicensePlateOCRDataset(val_dir)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=False, collate_fn=ocr_collate
    )

    val_dl = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, collate_fn=ocr_collate
    )

    model = CRNN().to(device)
    criterion = nn.CTCLoss(blank=len(CHARSET))
    opt = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, targets, target_lens in tqdm(train_dl, leave=False):
            images = images.to(device)  # MPS device
            # move targets and lengths to CPU for CTCLoss
            targets_cpu = targets.to("cpu")

            target_lens_cpu = target_lens.to("cpu")

            opt.zero_grad()

            out = model(images)

            out = out.permute(1, 0, 2)
            out = nn.functional.log_softmax(out, dim=2)

            out_cpu = out.to("cpu")

            T, B, _ = out_cpu.shape
            input_lengths = torch.full((B,), fill_value=T, dtype=torch.long)

            input_lens_cpu = input_lengths.to("cpu")
            # print("T =", T)
            # print("target_lens:", target_lens)
            loss = criterion(out_cpu, targets_cpu, input_lens_cpu, target_lens_cpu)
            # print("loss:", loss)
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "crnn_best.pth")
    print("Model saved to crnn_best.pth")

    return model

# trained_model = train_crnn(train_dir, val_dir)

def ctc_greedy_decode(logits):
    """
    logits: Tensor (T, C) after model output for a single sample
    returns: list of character indices (decoded)
    """
    # pick class with highest probability at each timestep
    print("argmax per timestep:", logits.argmax(-1)[:50])
    pred_indices = logits.argmax(dim=-1).cpu().numpy().tolist()

    decoded = []
    prev = -1
    blank = len(CHARSET)  # last character is CTC blank

    for p in pred_indices:
        if p != prev and p != blank:
            decoded.append(p)
        prev = p

    return decoded

def predict_plate(model, img_path, device="cpu"):
    model.eval()

    with torch.no_grad():
        img = preprocess_image(img_path, img_h = 32)
        img = img.unsqueeze(0).to(device)
        # forward pass
        logits = model(img)  # (1, T, C)
        print(logits.shape)
        logits = logits[0]   # remove batch → (T, C)
        # decode
        char_indices = ctc_greedy_decode(logits)
        text = decode_text(char_indices)

        return text

model = CRNN()              # create model with same architecture
model.load_state_dict(torch.load("crnn_best.pth", map_location="cpu"))
model.eval()

text = predict_plate(model, "/Users/sergeiakhmadulin/Car_Number_Plate/predictions/crops/train/images/Cars0_crop_0.jpg")
print("text: ",text)
# model = CRNN().to(device)
# model.load_state_dict(torch.load("crnn_best.pth", map_location=device))
#
# plate = predict_plate(model, "test_images/car1.jpg", device)
# print("Predicted:", plate)