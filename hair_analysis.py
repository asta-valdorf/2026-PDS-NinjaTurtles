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
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
    tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
    
    # Combine dark- and light-hair responses
    combined = cv2.add(blackhat, tophat)
    
    # threshold for binary hair mask
    _, hair_mask = cv2.threshold(combined, threshold, 255, cv2.THRESH_BINARY)
    hair_mask = cv2.medianBlur(hair_mask, 3)        # reduce small noise
    
    # compute coverage ratio
    total_area = img_gray.shape[0] * img_gray.shape[1]
    hair_area = np.count_nonzero(hair_mask)
    
    return hair_area / total_area


def removeHair_auto(img_org, img_gray, lesion_mask=None):
    """
    - Selects removal parameters based on coverage level
    - Chooses between light-hair and dark-hair detection
    - Inpaints detected hair regions


    Args:
        img_org (_type_): _description_
        img_gray (_type_): _description_
        lesion_mask (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    coverage = hair_coverage(img_gray)

    if coverage < 0.01:
        return "skip", None, None, img_org.copy(), coverage, 0, 0

    if coverage < 0.08:
        kernel_size = 9
        black_threshold = 12
        top_threshold = 18
        radius = 3
    else:
        kernel_size = 11
        black_threshold = 15
        top_threshold = 22
        radius = 5


    # structuring element for hair enhancement
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))


    # compute morphological responses
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
    tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)


    # threshold responses to create binary masks
    _, black_mask = cv2.threshold(blackhat, black_threshold, 255, cv2.THRESH_BINARY)
    _, top_mask = cv2.threshold(tophat, top_threshold, 255, cv2.THRESH_BINARY)


    black_score = np.count_nonzero(black_mask)
    top_score = np.count_nonzero(top_mask)

    # select the dominant hair color, only choose light mode if it clearly wins
    if top_score > 1.3 * black_score:
        mode = "light"
        response = tophat
        mask = top_mask
    else:
        mode = "dark"
        response = blackhat
        mask = black_mask

    # inpaint detected hair regions
    img_out = cv2.inpaint(img_org, mask, radius, cv2.INPAINT_TELEA)

    return mode, response, mask, img_out, coverage, black_score, top_score


# based on histogram
def label_coverage(coverage) -> str:
    """
    Categorize hair coverage level for analysis.
    Used during exploratory data analysis to group images
    into low, medium, and high hair coverage regimes.
    """

    if coverage < 0.01:
        return "low"
    elif coverage < 0.08:
        return "medium"
    else:
        return "high"



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