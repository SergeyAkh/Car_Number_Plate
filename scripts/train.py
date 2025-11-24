from config import DATA_YAML, MODEL_CONFIG
from ultralytics import YOLO

model = YOLO(MODEL_CONFIG["model_name"])

model.train(
    data=DATA_YAML,
    epochs=MODEL_CONFIG["epochs"],
    imgsz=MODEL_CONFIG["img_size"],
    batch=MODEL_CONFIG["batch_size"],
)