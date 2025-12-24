import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random

from torch.utils.data import DataLoader, ConcatDataset, Subset
from src.dataset import OCRFileDataset, collate_fn, augment_image
from src.utils import evaluate_accuracy
from src.models import TinyOCR
from config import charset, CHAR_MODEL_CONFIG, synt_path

blank_idx = len(charset)

device = CHAR_MODEL_CONFIG["DEVICE"]
train = os.path.join(CHAR_MODEL_CONFIG["crop_data"], "train")
val = os.path.join(CHAR_MODEL_CONFIG["crop_data"], "val")

train_transform = T.Compose([
    T.Lambda(lambda img: augment_image(img)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
    T.Lambda(lambda x: 1.0 - x)
])

val_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
    T.Lambda(lambda x: 1.0 - x)
])


synt_real = "real"

if synt_real == "synt":
    dataset_train_full = OCRFileDataset(
        os.path.join(synt_path, "images"),
        os.path.join(synt_path, "labels"),
        transform=train_transform,
        charset=charset
    )

    dataset_val_full = OCRFileDataset(
        os.path.join(synt_path, "images"),
        os.path.join(synt_path, "labels"),
        transform=val_transform,
        charset=charset
    )

    indices = list(range(len(dataset_train_full)))
    random.shuffle(indices)

    split = int(0.9 * len(indices))
    train_idx = indices[:split]
    val_idx   = indices[split:]

    dataset_train = Subset(dataset_train_full, train_idx)
    dataset_val   = Subset(dataset_val_full, val_idx)


elif synt_real == "real":

    dataset_train = OCRFileDataset(os.path.join(train, "images"), os.path.join(train, "labels"),
                                   transform=train_transform)
    dataset_train_full = OCRFileDataset(
        os.path.join(synt_path, "images"),
        os.path.join(synt_path, "labels"),
        transform=train_transform
    )
    indices = list(range(len(dataset_train_full)))
    train_idx = indices[:int(len(dataset_train)*0.2)]

    dataset_train = ConcatDataset([dataset_train])
    print(f"Train dataset size: {len(dataset_train)}")
    dataset_val = OCRFileDataset(os.path.join(val, "images"), os.path.join(val, "labels"), transform=val_transform)

batch_size=4

loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


num_classes = len(charset) + 1  # blank
model = TinyOCR(num_classes)

ctc_loss = nn.CTCLoss(blank=len(charset), zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def compute_time_downscale(model, input_height=32):

    with torch.no_grad():
        dummy = torch.zeros(1, 1, input_height, 400)  # 400 — любое большое значение
        out = model.conv(dummy)  # B×C×H×W'
        _, _, _, w_out = out.shape
        downscale = 400 // w_out
    return downscale

def finetune_model(model, data_loader_train, data_loader_val, epochs, device, optimizer, mod_name):

    model.load_state_dict(torch.load(f"finetuned_full_71.pth"))

    for p in model.conv.parameters():
        p.requires_grad = False


    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5, weight_decay = 1e-5
    )
    time_downscale = compute_time_downscale(model)

    char_acc_ind = 0

    model, char_acc_ind = train_batches(time_downscale, data_loader_train, data_loader_val, epochs, device, optimizer, mod_name, char_acc_ind)

    for p in model.conv.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)

    model, char_acc_ind = train_batches(time_downscale, data_loader_train, data_loader_val, epochs, device, optimizer, mod_name, char_acc_ind)

    return model

def train_batches(time_downscale, data_loader_train, data_loader_val, epochs, device, optimizer, mod_name, char_acc_ind):
    model.train()
    for epoch in range(epochs):
        for imgs, targets, input_lens, target_lens, texts in tqdm(data_loader_train, leave=False):
            imgs = imgs.to(device)  # B × 1 × H × W

            logits = model(imgs)  # T × B × num_classes
            log_probs = F.log_softmax(logits, dim=2)

            log_probs = log_probs.to("cpu")
            targets = targets.to("cpu")
            target_lens = target_lens.to("cpu")
            true_input_lens = (input_lens // time_downscale).clamp(min=1).to("cpu")
            # T × B × C
            loss = ctc_loss(log_probs, targets, true_input_lens, target_lens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            # ===========================
            #   CTC decode predictions
            # ===========================

            exact_train, char_acc_train, all_preds_tr, all_gts_tr = evaluate_accuracy(model, data_loader_train, blank_idx,
                                                                                      charset, device)
            exact_val, char_acc_val, all_preds, all_gts = evaluate_accuracy(model, data_loader_val, blank_idx, charset,
                                                                            device)

            print(f"[Accuracy @ epoch {epoch}] "
                  f"Exact_tr={exact_train * 100:.2f}% | CharAcc_tr={char_acc_train * 100:.2f}% | "
                  f"Exact_vl={exact_val * 100:.2f}% | CharAcc_vl={char_acc_val * 100:.2f}%")
            if char_acc_val > char_acc_ind and epoch != 0:
                char_acc_ind = char_acc_val
                torch.save(model.state_dict(), f"{mod_name}.pth")
                print(f"new accuracy model saved, with val accuracy: {char_acc_ind * 100:.2f}%")

    return model, char_acc_ind


finetune_model(model, loader_train, loader_val, epochs=100, device = device, optimizer=optimizer,mod_name = "finetuned_full_2")