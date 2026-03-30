import numpy as np
from math import nan
from scipy.spatial import ConvexHull

def convexity_score(mask):
    '''Calculate convexity score between 0 and 1,
    with 0 indicating a smoother border and 1 a more crooked border.

    Args:
        image (numpy.ndarray): input masked image

    Returns:
        convexity_score (float): Float between 0 and 1 indicating convexity.
    '''

    # Get coordinates of all pixels in the lesion mask
    coords = np.transpose(np.nonzero(mask))

    if len(coords) < 3:  # need at least 3 points for ConvexHull
        return nan

    # Compute convex hull of lesion pixels
    hull = ConvexHull(coords)

    if hull.volume == 0:
        return nan

    # Compute area of lesion mask
    lesion_area = np.count_nonzero(mask)

    # Compute convexity as ratio of lesion area to hull volume
    convexity = lesion_area / hull.volume

    return convexity #round(1-convexity, 3)