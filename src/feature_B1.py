# Code taken from exercise session 05_Feature_Extraction

import numpy as np
from scipy.spatial import ConvexHull

def convexity_score(mask):
    '''
    Calculate convexity score of binary mask,
    with 1 indicating a smoother border and 0 a more crooked border.

    Convexity is defined as the ratio of the mask area to the area of its convex hull (smallest boundary consisting of all 1 values).

    Args:
        mask (np.ndarray): input masked image

    Returns:
        convexity_score (float): Float between 0 and 1 indicating convexity.
        np.nan: If the mask is empty
    '''

    # Get coordinates of all pixels in the lesion mask
    coords = np.transpose(np.nonzero(mask))

    # If mask is empty
    if len(coords) < 3:  # need at least 3 points for ConvexHull
        return np.nan

    # Compute convex hull of lesion pixels
    hull = ConvexHull(coords)

    # If convex hull has zero area
    if hull.volume == 0:
        return np.nan

    # Compute area of lesion mask
    lesion_area = np.count_nonzero(mask)

    # Compute convexity as ratio of lesion area to hull volume
    convexity = lesion_area / hull.volume

    return convexity