import random
from PIL import Image
import torchvision.transforms.functional as F
import numpy as np

def augment_image(img):
    """
    img: PIL.Image (grayscale)
    returns: PIL.Image
    """
    # 1. Small rotation (rare)
    if random.random() < 0.3:
        angle = random.uniform(-3, 3)
        img = F.rotate(img, angle, fill=255)

    # 2. Mild perspective (very rare)
    if random.random() < 0.3:
        dx = 2
        dy = 2
        img = F.perspective(
            img,
            startpoints=[
                (0, 0),
                (img.width, 0),
                (img.width, img.height),
                (0, img.height),
            ],
            endpoints=[
                (random.randint(-dx, dx), random.randint(-dy, dy)),
                (img.width + random.randint(-dx, dx), random.randint(-dy, dy)),
                (img.width + random.randint(-dx, dx), img.height + random.randint(-dy, dy)),
                (random.randint(-dx, dx), img.height + random.randint(-dy, dy)),
            ],
            fill=255
        )

    # 3. Contrast (common but mild)
    if random.random() < 0.3:
        factor = random.uniform(0.8, 1.2)
        img = F.adjust_contrast(img, factor)

    # 4. Brightness (common but mild)
    if random.random() < 0.3:
        factor = random.uniform(0.85, 1.15)
        img = F.adjust_brightness(img, factor)

    # 5. Blur (rare and weak)
    if random.random() < 0.3:
        img = F.gaussian_blur(img, kernel_size=3)

    # 6. Gaussian noise (rare and weak)
    if random.random() < 0.3:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 4, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # 7. Horizontal shift (very small)
    if random.random() < 0.4:
        shift = random.randint(-2, 2)
        arr = np.array(img)
        arr = np.roll(arr, shift, axis=1)
        img = Image.fromarray(arr)

    return img