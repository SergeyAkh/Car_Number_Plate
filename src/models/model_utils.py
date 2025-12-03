import cv2
import torch
from config import CHARSET

def preprocess_image(img_path, img_h):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    # resize height to 32, maintain aspect ratio
    new_w = int(w * (img_h / h))
    # enforce minimum width
    new_w = max(new_w, 128)
    img = cv2.resize(img, (new_w, img_h))

    # img = img / 255.0
    img = img.astype("float32") / 255.0
    img = (img - 0.5) / 0.5
    img = torch.tensor(img).unsqueeze(0).float()

    return img

def encode_text(text, charset=CHARSET, char2idx=None):
    if char2idx is None:
        char2idx = {ch: i for i, ch in enumerate(charset)}

    idxs = []
    for ch in text:
        if ch in char2idx:
            idxs.append(char2idx[ch])
        # else: можно логировать неизвестные символы

    return torch.tensor(idxs, dtype=torch.long)

def decode_text(indices, charset=CHARSET):
    """
    Decode a sequence of class indices (after CTC greedy) to a string.
    - indices: list/iterable of ints or 1D torch tensor (already collapsed: no identical neighbours)
               It may still contain blanks (we will ignore them).
    - charset: string of characters (index 0 -> charset[0], etc.)
    Returns: decoded string
    """
    # ensure a plain Python list of ints
    if isinstance(indices, torch.Tensor):
        indices = indices.cpu().tolist()
    else:
        indices = list(indices)

    blank_idx = len(charset)  # convention used in training: blank is last index

    out_chars = []
    for idx in indices:
        if idx == blank_idx:
            continue
        if 0 <= idx < len(charset):
            out_chars.append(charset[idx])
        else:
            # ignore unknown indices (safety)
            continue

    return "".join(out_chars)