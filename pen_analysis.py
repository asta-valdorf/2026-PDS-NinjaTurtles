import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path


"""
Pen removal file structure:

- filter_pen_components: shape-based analysis
- detect_pen: color-based pen detection
- has_pen_mark: decision logic
- remove_pen: inpainting
- pen_removal: full preprocessing pipeline
"""

# Pen detection parameters (HSV)
# catch dark/light blue with different saturation and hue, and black pen marks

BLUE_LOW_1 = (90, 40, 80)
BLUE_HIGH_1 = (140, 255, 255)

BLUE_LOW_2 = (85, 30, 60)
BLUE_HIGH_2 = (140, 200, 255)

BLUE_LOW_3  = (85, 15, 50)
BLUE_HIGH_3 = (145, 120, 200)


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

    # label connected components in the binary mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean_mask = np.zeros_like(mask)

    for label_idx in range(1, num_labels):
        x, y, w, h, area = stats[label_idx]

        # allow large regions regardless of aspect ratio to preserve thick markers
        if area > 300:
            clean_mask[labels == label_idx] = 255
            continue

        # reject isolated noise
        if area < min_area:
            continue

        # distuingish strokes from blobs
        aspect_ratio = max(w, h) / max(1, min(w, h))
        if aspect_ratio < min_aspect_ratio:
            continue

        # keep only those passing all filters
        clean_mask[labels == label_idx] = 255

    return clean_mask


def detect_pen(img_bgr):
    """
    Detect blue and black pen marks.
    HSV color space. Blue pen is detected by saturation, black pen is detected conservatively and
    filtered by component shape to avoid removing dark lesion structures.
    """
    # convert from BGR to HSV
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


    blue_mask_3 = cv2.inRange(hsv,
                            np.array(BLUE_LOW_3, np.uint8),
                            np.array(BLUE_HIGH_3, np.uint8))

    # combine all blue
    blue_mask = cv2.bitwise_or(blue_mask, blue_mask_3)


    lower_black = np.array(BLACK_LOW, dtype=np.uint8)
    upper_black = np.array(BLACK_HIGH, dtype=np.uint8)
    
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    black_mask = cv2.medianBlur(black_mask, 3)
    black_mask = filter_pen_components(black_mask)      # only elongated pen marks

    # final pen mask
    pen_mask = cv2.bitwise_or(blue_mask, black_mask)

    return pen_mask, blue_mask, black_mask


def has_pen_mark(pen_mask, min_pixels=MIN_PEN_PIXELS) -> bool:
    """
    Determine whether an image contains enough pen-like pixels to justify removal.
    """
    return np.count_nonzero(pen_mask) >= min_pixels


def remove_pen(img_bgr, pen_mask):
    """
    Removes detected pen marks with image inpainting.
    Pen strokes are often thicker than the raw mask, so the mask is dilated first.
    """
    # ellipse matches better
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # expand mask with dilate
    mask_dilated = cv2.dilate(pen_mask, kernel, iterations=INPAINT_DILATION_ITERATIONS)
    clean_bgr = cv2.inpaint(img_bgr, mask_dilated, inpaintRadius=INPAINT_RADIUS, flags=cv2.INPAINT_TELEA)

    return mask_dilated, clean_bgr


def pen_removal(img_bgr):
    """
    Full pen-mark removal.
    Detects pen marks and inpaints if sufficient pen pixels are found.
    Returns cleaned image.
    """
    pen_mask, _, _ = detect_pen(img_bgr)

    if has_pen_mark(pen_mask):
        _, clean_bgr = remove_pen(img_bgr, pen_mask)
        return clean_bgr
    else:
        return img_bgr
