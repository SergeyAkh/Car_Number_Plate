from ultralytics import YOLO
from config import MODEL_CONFIG, BASE_DIR, ROOT, TRAIN_IMAGES, VAL_IMAGES, PATH_CROP_PLATE
from src.utils.utilites import get_latest_run, predict_and_crop
import os

import matplotlib
matplotlib.use("TkAgg")  # stable in PyCharm
import matplotlib.pyplot as plt

print(os.path.join(TRAIN_IMAGES))
box_model = YOLO(get_latest_run(MODEL_CONFIG["FOLDER"]))


val_to_predict = os.path.join(VAL_IMAGES)
train_to_predict = os.path.join(TRAIN_IMAGES)
predict_and_crop(box_model, train_to_predict, crop_dir=PATH_CROP_PLATE, train_val_single = "train")

# img = "Cars119.png"
# results = box_model(os.path.join(VAL_IMAGES, img), device = MODEL_CONFIG["DEVICE"])
#
# plt.imshow(results[0].plot())  # YOLO draws boxes here
# plt.axis("off")
# plt.show()

