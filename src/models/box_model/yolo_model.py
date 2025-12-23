from ultralytics import YOLO
from config import MODEL_CONFIG, BASE_DIR, ROOT, TRAIN_IMAGES, VAL_IMAGES, PATH_CROP_PLATE
from src.utils.utilites import get_latest_run, predict_and_crop
import os

box_model = YOLO(get_latest_run(MODEL_CONFIG["FOLDER"]))

def create_predictions(box_model, VAL_IMAGES, TRAIN_IMAGES):
    val_to_predict = os.path.join(VAL_IMAGES)
    train_to_predict = os.path.join(TRAIN_IMAGES)
    predict_and_crop(box_model, train_to_predict, crop_dir=PATH_CROP_PLATE, train_val_single = "train")





