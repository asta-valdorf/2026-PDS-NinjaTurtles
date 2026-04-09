# Code taken from exercise session 05_Feature_Extraction

import numpy as np
from math import floor, ceil
from skimage.transform import rotate

def midpoint(mask):
    '''
    Finds midpoint of mask array.

    Args:
        mask (np.ndarray): input mask

    Returns:
        row_mid (float): row index of midpoint
        col_mid (float): column index of midpoint 

    Note:
        We find midpoint of whole picture and not midpoint of the lesion,
        because in the code we use the output from cut_mask to find the midpoint.
    '''
    row_mid = mask.shape[0] / 2
    col_mid = mask.shape[1] / 2

    return row_mid, col_mid

def asymmetry(mask):
    '''
    Compute asymmetry score between 0 and 1 from the vertical and the horizontal axis
    on a binary mask, with 0 being complete symmetry, 1 being complete asymmetry,
    i.e. no pixels overlapping when folding mask on x- and y-axis

    Args:
        mask (np.ndarray): input mask

    Returns:
        asymmetry_score (float): Float between 0 and 1 indicating level of asymmetry.
        np.nan: If mask is empty
    '''

    row_mid, col_mid = midpoint(mask)

    # Split mask into halves horizontally and vertically
    upper_half = mask[:ceil(row_mid), :]
    lower_half = mask[floor(row_mid):, :]
    left_half = mask[:, :ceil(col_mid)]
    right_half = mask[:, floor(col_mid):]

    # Flip one half for each axis
    flipped_lower = np.flip(lower_half, axis=0)
    flipped_right = np.flip(right_half, axis=1)

    # Use logical xor to find pixels where only one half is present
    hori_xor_area = np.logical_xor(upper_half, flipped_lower)
    vert_xor_area = np.logical_xor(left_half, flipped_right)

    total_pxls = np.sum(mask)

    # If mask is empty
    if total_pxls == 0:
        return np.nan

    hori_asymmetry_pxls = np.sum(hori_xor_area)
    vert_asymmetry_pxls = np.sum(vert_xor_area)

    # Calculate asymmetry score
    asymmetry_score = (hori_asymmetry_pxls + vert_asymmetry_pxls) / (total_pxls * 2)

    return round(asymmetry_score, 4)

def cut_mask(mask):
    '''
    Cut empty space from mask array such that it has smallest possible dimensions.

    Removes rows and columns that contain only zero values and 
    returns the smallest sub-array that contains non-zero values.

    Args:
        mask (np.ndarray): mask to cut

    Returns:
        cut_mask_ (np.ndarray): cut mask
        None: if given mask is empty 
    '''
    col_sums = np.sum(mask, axis=0)
    row_sums = np.sum(mask, axis=1)

    # Stores columns with non-zero values
    active_cols = []
    for index, col_sum in enumerate(col_sums):
        if col_sum != 0:
            active_cols.append(index)

    # Stores rows with non-zero values
    active_rows = []
    for index, row_sum in enumerate(row_sums):
        if row_sum != 0:
            active_rows.append(index)

    # return None if mask is empty after rotation
    if len(active_cols) == 0 or len(active_rows) == 0:
        return None

    col_min = active_cols[0]
    col_max = active_cols[-1]
    row_min = active_rows[0]
    row_max = active_rows[-1]

    cut_mask_ = mask[row_min:row_max+1, col_min:col_max+1]

    return cut_mask_

def rotation_asymmetry(mask, n: int):
    '''
    Rotate mask n times and calculate asymmetry score for each iteration.
    Rotates n times between 0 and 90 degrees, as 90 degree rotations do not change the
    asymmetry score, i.e., a 30 degree rotation is the same as a 120 degree rotation.

    Args:
        mask (np.ndarray): input mask
        n (int): amount of rotations

    Returns:
        asymmetry_scores (dict): dict of asymmetry scores calculated from each rotation.
    '''
    asymmetry_scores = {}

    for i in range(n):

        degrees = 90 * i / n

        rotated_mask = rotate(mask, degrees)
        cutted_mask = cut_mask(rotated_mask)

        # If mask is empty
        if cutted_mask is None:
            continue

        asymmetry_scores[degrees] = asymmetry(cutted_mask)

    return asymmetry_scores

def mean_asymmetry(mask, rotations = 30):
    '''
    Return mean asymmetry score from mask.

    Args:
        mask (np.ndarray): mask to compute asymmetry score for
        rotations (int, optional): amount of rotations (default 30)

    Returns:
        mean_score (float): mean asymmetry score.
        np.nan: If length of assymetry_scores are zero
    '''
    # rotation_asymmetry introduces interpolation, so mask may contain values between 0 and 1, instead of strictly 0 and 1
    asymmetry_scores = rotation_asymmetry(mask, rotations)

    # If mask is empty
    if len(asymmetry_scores) == 0:
        return np.nan
    
    mean_score = sum(asymmetry_scores.values()) / len(asymmetry_scores)

    return mean_score