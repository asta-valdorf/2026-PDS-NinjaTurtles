import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path

# pen detection and removal


# full dataset
df = pd.read_csv('md.csv')

IMG_DIR = Path("imgs")
MASK_DIR = Path("masks")

img_id = df["img_id"].iloc[0]
img_path = IMG_DIR / img_id
mask_path = MASK_DIR / img_id.replace(".png", "_mask.png")


# Pen detection parameters for quick single-image tuning. (HSV)
# catch dark/light blue with different saturation and black/lighter
BLUE_LOW_1 = (90, 40, 80)
BLUE_HIGH_1 = (140, 255, 255)

BLUE_LOW_2 = (85, 30, 60)
BLUE_HIGH_2 = (140, 200, 255)

BLACK_LOW = (0, 0, 0)
BLACK_HIGH = (180, 50, 80)

FILTER_MIN_AREA = 8
FILTER_MIN_ASPECT_RATIO = 1.5
MIN_PEN_PIXELS = 100
INPAINT_DILATION_ITERATIONS = 3
INPAINT_RADIUS = 5


def filter_pen_components(mask, min_area=FILTER_MIN_AREA, min_aspect_ratio=FILTER_MIN_ASPECT_RATIO):
    """
    Keep only connected components that look stroke-like rather than blob-like.
    This helps reject round dark lesion regions that can resemble black pen.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean_mask = np.zeros_like(mask)

    for label_idx in range(1, num_labels):
        x, y, w, h, area = stats[label_idx]

        if area < min_area:
            continue

        aspect_ratio = max(w, h) / max(1, min(w, h))
        if aspect_ratio < min_aspect_ratio:
            continue

        clean_mask[labels == label_idx] = 255

    return clean_mask


def count_pen_components(mask, min_area=FILTER_MIN_AREA, min_aspect_ratio=FILTER_MIN_ASPECT_RATIO):
    """
    Count stroke-like connected components in a mask.
    """
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    count = 0

    for label_idx in range(1, num_labels):
        x, y, w, h, area = stats[label_idx]
        if area < min_area:
            continue

        aspect_ratio = max(w, h) / max(1, min(w, h))
        if aspect_ratio < min_aspect_ratio:
            continue

        count += 1

    return count


def detect_pen(img_bgr):
    """
    Detect likely blue and black pen marks in a BGR uint8 image.
    Blue pen is detected by hue; black pen is detected conservatively and
    filtered by component shape to avoid removing dark lesion structures.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # different blue saturation
    lower_blue_1 = np.array(BLUE_LOW_1, dtype=np.uint8)
    upper_blue_1 = np.array(BLUE_HIGH_1, dtype=np.uint8)
    lower_blue_2 = np.array(BLUE_LOW_2, dtype=np.uint8)
    upper_blue_2 = np.array(BLUE_HIGH_2, dtype=np.uint8)

    # combined into one with bitwise OR
    blue_mask_1 = cv2.inRange(hsv, lower_blue_1, upper_blue_1)
    blue_mask_2 = cv2.inRange(hsv, lower_blue_2, upper_blue_2) 
    blue_mask = cv2.bitwise_or(blue_mask_1, blue_mask_2)

    lower_black = np.array(BLACK_LOW, dtype=np.uint8)
    upper_black = np.array(BLACK_HIGH, dtype=np.uint8)
    
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    black_mask = cv2.medianBlur(black_mask, 3)
    black_mask = filter_pen_components(black_mask)          # filter to reduce to pen strokes

    pen_mask = cv2.bitwise_or(blue_mask, black_mask)
    print(img_bgr.dtype)
    print(img_bgr.min(), img_bgr.max())
    return pen_mask, blue_mask, black_mask


def has_pen_mark(pen_mask, min_pixels=MIN_PEN_PIXELS):
    """
    Decide whether an image contains enough pen-like pixels to justify inpainting.
    """
    return np.count_nonzero(pen_mask) >= min_pixels


def remove_pen(img_bgr, pen_mask):
    """
    Inpaint over detected pen marks.
    Pen strokes are often thicker than the raw mask, so the mask is dilated first.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_dilated = cv2.dilate(pen_mask, kernel, iterations=INPAINT_DILATION_ITERATIONS)
    clean_bgr = cv2.inpaint(img_bgr, mask_dilated, inpaintRadius=INPAINT_RADIUS, flags=cv2.INPAINT_TELEA)

    return mask_dilated, clean_bgr


def show_pen_result(image_path):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"Warning: Could not read image {image_path}")
        return

    pen_mask, blue_mask, black_mask = detect_pen(img_bgr)
    has_pen = has_pen_mark(pen_mask)
    print(f"Blue pixels: {np.count_nonzero(blue_mask)}")
    print(f"Black pixels: {np.count_nonzero(black_mask)}")

    if has_pen:
        mask_dilated, clean_bgr = remove_pen(img_bgr, pen_mask)
    else:
        mask_dilated = pen_mask.copy()
        clean_bgr = img_bgr.copy()

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    print("HSV stats:")
    print("H:", hsv[:,:,0].min(), hsv[:,:,0].max())
    print("S:", hsv[:,:,1].min(), hsv[:,:,1].max())
    print("V:", hsv[:,:,2].min(), hsv[:,:,2].max())
    
    pen_pixels = int(np.count_nonzero(pen_mask))

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(
        f"{Path(image_path).name} | pen_pixels={pen_pixels} | "
        f"{'pen detected' if has_pen else 'skipped'}"
    )

    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(blue_mask, cmap="gray")
    axes[1].set_title("Blue pen mask")
    axes[1].axis("off")

    axes[2].imshow(black_mask, cmap="gray")
    axes[2].set_title("Black pen mask")
    axes[2].axis("off")

    axes[3].imshow(mask_dilated, cmap="gray")
    axes[3].set_title("Final pen mask")
    axes[3].axis("off")

    axes[4].imshow(cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB))
    axes[4].set_title("Pen removed")
    axes[4].axis("off")
    
    plt.tight_layout()
    plt.show()

def show_pen_examples(df, n=5):
    if len(df) == 0:
        print("No images available for pen examples")
        return

    samples = df.sample(min(n, len(df)), random_state=None)

    for _, row in samples.iterrows():
        show_pen_result(row["image_path"])



df = pd.DataFrame(data)
show_pen_examples(df)


