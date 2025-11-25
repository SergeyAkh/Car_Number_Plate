import pytesseract
from PIL import Image
import os

img_dir = "/Users/sergeiakhmadulin/Car_Number_Plate/predictions/crops/val/images"
label_dir = "/Users/sergeiakhmadulin/Car_Number_Plate/predictions/crops/val/labels"

os.makedirs(label_dir, exist_ok=True)

for img_name in os.listdir(img_dir):
    if not img_name.lower().endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(img_dir, img_name)

    img = Image.open(img_path)

    config = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    text = pytesseract.image_to_string(img, config=config)
    text = text.strip().replace(" ", "").replace("\n", "")
    # Extract text


    # Save label
    label_path = os.path.join(label_dir, img_name.rsplit(".", 1)[0] + ".txt")
    with open(label_path, "w") as f:
        f.write(text)

    print(img_name, "→", text)
