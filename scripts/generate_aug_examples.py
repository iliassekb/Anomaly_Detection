"""
Generate side-by-side augmentation example images for the report.
Each image shows: Original (left) | Augmented (right), same style as rotation_peach_example.jpg
Output: aug_exemple/ folder
"""

import os
import cv2
import numpy as np
import albumentations as A
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = "aug_exemple"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SOURCE_IMAGE = "data/yolo_multiclass_dataset/images/train/peach_cut_003.jpg"

img_bgr = cv2.imread(SOURCE_IMAGE)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Resize to a consistent square for clean display
SIZE = 700
img_rgb = cv2.resize(img_rgb, (SIZE, SIZE))

LABEL_COLOR_ORIG = (40, 40, 40)       # dark grey for "Original"
LABEL_COLOR_AUG  = (200, 100, 0)      # orange for augmented label (matches example)
BG_COLOR = (245, 245, 245)
TITLE_H = 45
GAP = 8

def make_comparison(original: np.ndarray, augmented: np.ndarray,
                    aug_label: str, filename: str):
    h, w = original.shape[:2]
    canvas_w = w * 2 + GAP * 3
    canvas_h = h + TITLE_H + GAP

    canvas = np.full((canvas_h, canvas_w, 3), BG_COLOR, dtype=np.uint8)

    # Paste images
    x0 = GAP
    x1 = GAP * 2 + w
    canvas[TITLE_H:TITLE_H + h, x0:x0 + w] = original
    canvas[TITLE_H:TITLE_H + h, x1:x1 + w] = augmented

    img_pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 28)
    except Exception:
        font = ImageFont.load_default()

    # "Original" label centered above left image
    orig_text = "Original"
    bbox = draw.textbbox((0, 0), orig_text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((x0 + (w - tw) // 2, 8), orig_text, fill=LABEL_COLOR_ORIG, font=font)

    # Augmentation label centered above right image
    bbox2 = draw.textbbox((0, 0), aug_label, font=font)
    tw2 = bbox2[2] - bbox2[0]
    draw.text((x1 + (w - tw2) // 2, 8), aug_label, fill=LABEL_COLOR_AUG, font=font)

    out_path = os.path.join(OUTPUT_DIR, filename)
    img_pil.save(out_path, quality=95)
    print(f"  Saved: {out_path}")


# ── Deterministic seed for reproducibility ───────────────────────────────────
SEED = 42

augmentations = [
    (
        "flip_horizontal",
        "Flip Horizontal",
        A.HorizontalFlip(p=1.0),
    ),
    (
        "flip_vertical",
        "Flip Vertical",
        A.VerticalFlip(p=1.0),
    ),
    (
        "rotation_25deg",
        "Rotation  25°",
        A.Rotate(limit=(25, 25), border_mode=cv2.BORDER_REFLECT_101, p=1.0),
    ),
    (
        "zoom",
        "Zoom  ×1.20",
        A.RandomResizedCrop(
            size=(SIZE, SIZE),
            scale=(0.69, 0.69),   # zoom ~1.20×
            ratio=(1.0, 1.0),
            p=1.0,
        ),
    ),
    (
        "gaussian_noise",
        "Bruit Gaussien",
        A.GaussNoise(std_range=(0.08, 0.08), p=1.0),
    ),
    (
        "gaussian_blur",
        "Flou Gaussien",
        A.GaussianBlur(blur_limit=(11, 11), p=1.0),
    ),
    (
        "elastic_distortion",
        "Distorsion Elastique",
        A.ElasticTransform(alpha=120, sigma=8,
                           border_mode=cv2.BORDER_REFLECT_101, p=1.0),
    ),
    (
        "brightness",
        "Luminosite  +40%",
        A.RandomBrightnessContrast(
            brightness_limit=(0.4, 0.4),
            contrast_limit=(0.0, 0.0),
            p=1.0,
        ),
    ),
]

print(f"Generating {len(augmentations)} augmentation examples -> {OUTPUT_DIR}/\n")
for fname, label, transform in augmentations:
    np.random.seed(SEED)
    result = transform(image=img_rgb)
    aug_img = result["image"]
    make_comparison(img_rgb, aug_img, label, f"{fname}.jpg")

print("\nDone.")
