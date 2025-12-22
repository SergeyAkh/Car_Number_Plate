import torch, os, sys
from torchvision import transforms as T
from torch.utils.data import DataLoader
from src.training.synthetic import SyntheticTextDataset, collate_fn, IMG_HEIGHT, MAX_IMG_WIDTH
from PIL import Image
import numpy as np

# === Adjust these to match your project ===
PROJECT_SCRIPT = "simple_crnn_ctc_fixed.py"   # just for context, not imported
MODEL_PATH = "crnn_ctc_fixed.pth"            # if you saved model; if not, we'll use freshly init model below
USE_SAVED_MODEL = False                      # set True if you want to load saved state dict
BATCH_SIZE = 8

# === Import or reimplement small helpers (if your file structure differs, adapt imports) ===
# I'll re-create the minimal CHARSET and model class that matches your script so this snippet runs standalone.
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

import torch.nn as nn

class DummyCRNN(nn.Module):
    # Simple init that matches shapes used in your training script
    def __init__(self, img_h=32, n_channels=1, n_classes=len(CHARSET)+1, hidden_size=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(64,128,3,1,1),
            nn.ReLU(True),
            nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(128,256,3,1,1),
            nn.ReLU(True),
        )
        feat_h = max(1, img_h // 4)
        rnn_in = 256 * feat_h
        self.rnn = nn.LSTM(rnn_in, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size*2, n_classes)
    def forward(self, x):
        b = x.size(0)
        x = self.cnn(x)
        c,h,w = x.shape[1], x.shape[2], x.shape[3]
        x = x.permute(0,3,1,2).contiguous().view(b, w, c*h)
        x, _ = self.rnn(x)
        x = self.linear(x)
        x = x.permute(1,0,2)   # T, B, C
        return x
#
# # === Try to import your dataset class if available in same folder ===
# try:
#     # if your script exposes SyntheticTextDataset, import it
#     from src.training.synthetic import SyntheticTextDataset, collate_fn, IMG_HEIGHT, MAX_IMG_WIDTH
#     print("Imported SyntheticTextDataset from simple_crnn_ctc_fixed.py")
#     DS_OK = True
# except Exception as e:
#     print("Could not import SyntheticTextDataset from simple_crnn_ctc_fixed.py:", e)
#     DS_OK = False
#
# # If import succeeded, make a dataloader; otherwise recreate minimal dataset by rendering text via PIL
# if DS_OK:

#     transform = T.Compose([T.ToTensor()])
#     ds = SyntheticTextDataset(n_samples=BATCH_SIZE, transform=transform, font_path=None)
#     dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
#     imgs, targets, widths, target_lens, texts = next(iter(dl))
# else:
#     # create toy images: white background with black digit strings using PIL
#     from PIL import ImageDraw, ImageFont
#     transform = T.Compose([T.ToTensor()])
#     imgs_list = []
#     targets_list = []
#     texts = []
#     font = ImageFont.load_default()
#     for i in range(BATCH_SIZE):
#         txt = "".join(np.random.choice(list(CHARSET), size=np.random.randint(3,8)))
#         texts.append(txt)
#         im = Image.new('L', (150, 32), 255)
#         draw = ImageDraw.Draw(im)
#         draw.text((2,6), txt, fill=0, font=font)
#         t = transform(im)
#         imgs_list.append(t)
#         targets_list.append(torch.tensor([CHARSET.index(c) for c in txt], dtype=torch.long))
#     # pad into batch
#     W = max(t.shape[2] for t in imgs_list)
#     B = len(imgs_list)
#     imgs = torch.ones(B,1,imgs_list[0].shape[1], W)
#     for i,t in enumerate(imgs_list):
#         imgs[i,:,:,:t.shape[2]] = t
#     targets = torch.cat(targets_list)
#     widths = torch.tensor([t.shape[2] for t in imgs_list], dtype=torch.long)
#     target_lens = torch.tensor([len(t) for t in targets_list], dtype=torch.long)
#
# # Print basic image batch info and save the first sample image
# print("\\n--- IMAGE / DATASET CHECK ---")
# print("Batch imgs shape:", imgs.shape)            # B, C, H, W
# print("imgs dtype:", imgs.dtype, "range:", float(imgs.min()), float(imgs.max()), "mean:", float(imgs.mean()))
# print("Widths tensor:", widths)
# print("Target lens:", target_lens)
# print("Concatenated targets length:", targets.shape if 'targets' in locals() else 'n/a')
# print("Sample texts:", texts[:8])
#
# # Save first 4 images to disk for inspection
# os.makedirs("debug_images", exist_ok=True)
# for i in range(min(4, imgs.shape[0])):
#     arr = (imgs[i].cpu().numpy().squeeze() * 255).astype('uint8')
#     Image.fromarray(arr).save(f"debug_images/sample_{i}.png")
#     print(f"Saved debug_images/sample_{i}.png (text='{texts[i]}')")
#
# # === Model & forward pass ===
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("\\n--- MODEL CHECK ---")
# n_classes = len(CHARSET) + 1
# model = DummyCRNN(img_h=32, n_channels=1, n_classes=n_classes).to(device)
# if USE_SAVED_MODEL and os.path.exists(MODEL_PATH):
#     st = torch.load(MODEL_PATH, map_location=device)
#     if 'model_state' in st:
#         model.load_state_dict(st['model_state'])
#         print("Loaded model_state from", MODEL_PATH)
#     else:
#         try:
#             model.load_state_dict(st)
#             print("Loaded state dict from", MODEL_PATH)
#         except Exception as e:
#             print("Failed to load model state:", e)
# else:
#     print("Using freshly initialized model")
#
# model.eval()
# imgs_dev = imgs.to(device)
# with torch.no_grad():
#     outputs = model(imgs_dev)   # T, B, C
# print("Model output shape (T, B, C):", outputs.shape)
# print("Logits stats: min", float(outputs.min()), "max", float(outputs.max()), "mean", float(outputs.mean()))
# probs = outputs.softmax(2)
# print("Softmax per-class min/max (sample):", float(probs.min()), float(probs.max()))
# # check blank class probability over timesteps for first sample
# blank_idx = len(CHARSET)
# first_probs = probs[:,0,:]   # T, C
# blank_probs = first_probs[:, blank_idx].cpu().numpy()
# print("Blank probs (first sample) head 20:", blank_probs[:20])
#
# # Argmax indices per timestep (first sample)
# argmax_inds = probs.argmax(dim=2).cpu().numpy()  # T, B
# print("Argmax (first sample) first 60 timesteps:", argmax_inds[:60,0].tolist())
#
# # Show greedy collapse for the first sample step-by-step
# def greedy_from_inds(ind_seq, charset):
#     out = []
#     prev = -1
#     for idx in ind_seq:
#         if idx != prev and idx != len(charset):
#             out.append(charset[idx])
#         prev = idx
#     return ''.join(out)
#
# first_argmax = argmax_inds[:,0].tolist()
# decoded = greedy_from_inds(first_argmax, CHARSET)
# print("Greedy decoded (first sample):", repr(decoded))
# print("GT text (first sample):", texts[0])
#
# # Compute a single-batch CTCLoss (sanity)
# ctc = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
# Tt = outputs.size(0)
# B = outputs.size(1)
# input_lens = torch.full((B,), Tt, dtype=torch.long).to(device)
# # ensure targets and target_lens are on device
# if 'targets' in locals():
#     try:
#         loss = ctc(outputs.log_softmax(2), targets.to(device), input_lens, target_lens.to(device))
#         print("Single-batch CTCLoss:", float(loss.detach().cpu().numpy()))
#     except Exception as e:
#         print("CTC loss computation failed:", e)
# else:
#     print("No targets available to compute loss")





transform = T.Compose([T.ToTensor(), T.Lambda(lambda x: 1.0 - x)])
ds = SyntheticTextDataset(n_samples=8, transform=transform, font_path=None)
dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
imgs, targets, widths, target_lens, texts = next(iter(dl))

print("imgs.shape:", imgs.shape)         # expect B,1,H,W where W == max(widths)
print("widths:", widths)
print("target_lens:", target_lens)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DummyCRNN(IMG_HEIGHT, 1, n_classes=len(CHARSET)+1).to(device)
with torch.no_grad():
    out = model(imgs.to(device))
print("model outputs shape (T, B, C):", out.shape)
print("out.size(0) == widths.max()? ", out.size(0), widths.max().item())