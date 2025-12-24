import torch
from PIL import Image
import torchvision.transforms as T

def ctc_decode(pred, blank_idx, charset):
    """
    pred: (T,) numpy or tensor of class indices
    """
    pred = pred.numpy()
    decoded = []
    prev = blank_idx

    for p in pred:
        if p != prev and p != blank_idx:
            decoded.append(charset[p])
        prev = p
    return "".join(decoded)

def compute_accuracy_single(decoded_texts, gt_texts):
    """
    decoded_texts: list[str]
    gt_texts: list[str]
    """
    exact = 0
    total_chars = 0
    correct_chars = 0

    for pred, gt in zip(decoded_texts, gt_texts):
        if pred == gt:
            exact += 1

        # посимвольная точность
        L = max(len(pred), len(gt))
        for i in range(L):
            if i < len(pred) and i < len(gt) and pred[i] == gt[i]:
                correct_chars += 1
        total_chars += L

    exact_match = exact / len(gt_texts)
    char_acc = correct_chars / total_chars if total_chars > 0 else 0
    return exact_match, char_acc

def evaluate_accuracy(model, data_loader, blank_idx, charset, device):
    model.eval()

    all_preds = []
    all_gts = []

    with torch.no_grad():
        for imgs, targets, input_lens, target_lens, texts in data_loader:

            imgs = imgs.to(device)

            logits = model(imgs)              # T × B × C
            pred_idxs = logits.argmax(dim=2).transpose(0, 1)  # B × T

            for i in range(pred_idxs.size(0)):
                p = pred_idxs[i].cpu().numpy()
                # CTC decode
                decoded_idx = []
                prev = blank_idx
                for x in p:
                    if x != blank_idx and x != prev:
                        decoded_idx.append(x)
                    prev = x
                pred_text = "".join([charset[c] for c in decoded_idx])

                all_preds.append(pred_text)
                all_gts.append(texts[i])

    # ---- compute accuracy ----
    exact = sum([p == g for p, g in zip(all_preds, all_gts)]) / len(all_gts)

    correct_chars = 0
    total_chars = 0
    for p, g in zip(all_preds, all_gts):
        L = max(len(p), len(g))
        for i in range(L):
            if i < len(p) and i < len(g) and p[i] == g[i]:
                correct_chars += 1
        total_chars += L
    char_acc = correct_chars / total_chars if total_chars else 0

    return exact, char_acc, all_preds, all_gts

def read_img(name, img_h):
    img_pil = name.convert("RGB").convert("L")

    w, h = img_pil.size
    new_w = int(w * (img_h / h))
    img = img_pil.resize((new_w, img_h), Image.BILINEAR)

    transform = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: 1.0 - x)
    ])
    img = transform(img).unsqueeze(0)  # C×H×W

    return img

def ctc_beam_search(log_probs, beam_width=10, blank=0):
    """
    log_probs: (T, C) torch tensor (log softmax)
    return: list of char indices (no blanks, no repeats)
    """
    T, C = log_probs.shape
    beams = {(): 0.0}

    for t in range(T):
        new_beams = {}

        for seq, score in beams.items():
            for c in range(C):
                p = log_probs[t, c].item()

                if c == blank:
                    new_seq = seq
                else:
                    if len(seq) == 0 or seq[-1] != c:
                        new_seq = seq + (c,)
                    else:
                        new_seq = seq

                new_score = score + p
                if new_seq in new_beams:
                    new_beams[new_seq] = max(new_beams[new_seq], new_score)
                else:
                    new_beams[new_seq] = new_score

        beams = dict(
            sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width]
        )

    best_seq = max(beams.items(), key=lambda x: x[1])[0]
    return best_seq
