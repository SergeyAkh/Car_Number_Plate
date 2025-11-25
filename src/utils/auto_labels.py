from paddleocr import PaddleOCR
import os

img_dir = "/Users/sergeiakhmadulin/Car_Number_Plate/predictions/crops"
label_dir = "ModelB_dataset/train/labels"

os.makedirs(label_dir, exist_ok=True)

ocr = PaddleOCR(use_angle_cls=True, lang="en")

for img_name in os.listdir(img_dir):
    if not img_name.lower().endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(img_dir, img_name)
    result = ocr.ocr(img_path, cls=True)

    # Extract text
    raw_text = "".join([line[1][0] for line in result[0]])

    # Normalize text: remove spaces, dashes, dots
    clean_text = (
        raw_text.replace(" ", "")
                .replace("-", "")
                .replace("_", "")
    )

    # Save label
    # label_path = os.path.join(label_dir, img_name.rsplit(".", 1)[0] + ".txt")
    # with open(label_path, "w") as f:
    #     f.write(clean_text)

    print(img_name, "→", clean_text)
