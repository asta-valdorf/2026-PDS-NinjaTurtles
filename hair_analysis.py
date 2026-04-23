import cv2
import numpy as np
from pathlib import Path

# hair detection and removal for light and dark hair
# group into three groups based on amount of coverage

def hair_coverage(img_gray, kernel_size=9, threshold=10) -> float:
    """
    Estimate the fraction of the image covered by hair.
    Uses combined blackhat+tophat for complete dark/light hair detection.
    """

    # structuring element shaped as a cross to emphasize thin hair-like structures
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, 3))
    
    # only get dark hair to not get light spots with tophat and overdetect coverage
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)

    # segment the pixels where the difference between closing and original image intensities is bigger than 10 
    _, mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)

    hair_mask = cv2.medianBlur(mask, 3)        # reduce small noise
    
    # compute coverage ratio
    total_area = img_gray.shape[0] * img_gray.shape[1]
    hair_area = np.count_nonzero(hair_mask)
    
    return hair_area / total_area


def removeHair_auto(img_org, img_gray, lesion_mask=None):
    """
    Selects removal parameters based on coverage level
    Chooses between light-hair and dark-hair detection
    Inpaints detected hair regions
    """
    
    coverage = hair_coverage(img_gray)

    if coverage < 0.05:
        return "skip", None, None, img_org.copy(), coverage, 0, 0

    if coverage < 0.2:
        kernel_size = 9
        black_threshold = 11
        top_threshold = 20
        radius = 3
    else:
        kernel_size = 11
        black_threshold = 15
        top_threshold = 25
        radius = 5


    # structuring element for hair enhancement
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, 3))

    # compute morphological responses
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
    tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)

    # threshold responses to create binary masks
    _, black_mask = cv2.threshold(blackhat, black_threshold, 255, cv2.THRESH_BINARY)
    _, top_mask = cv2.threshold(tophat, top_threshold, 255, cv2.THRESH_BINARY)


    black_score = np.count_nonzero(black_mask)
    top_score = np.count_nonzero(top_mask)

    # select the dominant hair color, only choose light mode if it clearly wins
    if top_score > 1.2 * black_score:
        mode = "light"
        response = tophat
        mask = top_mask
    else:
        mode = "dark"
        response = blackhat
        mask = black_mask

    # inpaint detected hair regions
    img_out = cv2.inpaint(img_org, mask, radius, cv2.INPAINT_NS)

    return mode, response, mask, img_out, coverage, black_score, top_score


def hair_removal(img_bgr):
    """
    Complete hair removal pipeline.

    - Converts image to grayscale
    - Estimates hair coverage
    - Applies adaptive parameters based on coverage
    - Skips processing when hair presence is negligible
    - Returns cleaned image.
    """

    # convert to grayscale for morphological processing
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # apply adaptive hair removal
    mode, response, mask, img_out, coverage, black_score, top_score = \
        removeHair_auto(img_bgr, img_gray)

    # if skipped, return original image
    if mode == "skip":
        return img_bgr

    return img_out



