from config import VAL_IMAGES, charset, CHAR_MODEL_CONFIG
from src.utils.utilites import predict_and_crop,get_latest_model_new, plot_two_images_with_text
from src.models import TinyOCR, box_model
from src.utils.char_utils import *

import torch
import os

import matplotlib
matplotlib.use("TkAgg")

blank_idx = len(charset)

best_char_model = get_latest_model_new(CHAR_MODEL_CONFIG["FOLDER"])
model = TinyOCR(CHAR_MODEL_CONFIG["num_classes"])
model.load_state_dict(torch.load(best_char_model))

img = "N92.jpeg"
original_image = Image.open(os.path.join(VAL_IMAGES, img))
plate_box = predict_and_crop(box_model, os.path.join(VAL_IMAGES, img))

img_1 = read_img(plate_box, CHAR_MODEL_CONFIG["img_h"])

logits = model(img_1)

pred = logits.argmax(dim=2).squeeze(1).detach().cpu()

decoded = ctc_decode(pred, blank_idx, charset)

plot_two_images_with_text(
    original_image,
    plate_box,
    text=decoded
)