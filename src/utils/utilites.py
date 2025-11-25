import os
from PIL import Image

def get_latest_run(base_dir=None):
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"{base_dir} does not exist")

    runs = [os.path.join(base_dir, d) for d in os.listdir(base_dir)]
    runs = [d for d in runs if os.path.isdir(d)]

    if not runs:
        raise FileNotFoundError("No YOLO runs found")

    latest = max(runs, key=os.path.getmtime)
    best_model_path = os.path.join(latest, "weights", "best.pt")

    return best_model_path

def predict_and_crop(model, source, crop_dir, train_val_single = None):
    """
    Runs prediction and crops all detections.
    Works for: single image OR any folder of images.
    """
    # Run prediction
    results = model.predict(source, save=False, save_crop=False)

    if train_val_single is not None:
        crop_dir = os.path.join(crop_dir, train_val_single, "images")

    os.makedirs(crop_dir, exist_ok=True)

    for r in results:
        img = Image.fromarray(r.orig_img)
        img_name = os.path.splitext(os.path.basename(r.path))[0]

        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            crop = img.crop((x1, y1, x2, y2))

            crop_path = os.path.join(crop_dir, f"{img_name}_crop_{i}.jpg")
            crop.save(crop_path)

    print(f"Crops saved to: {crop_dir}")