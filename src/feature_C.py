# Code taken from exercise session 05_Feature_Extraction

import cv2
import numpy as np
from skimage.transform import resize
from sklearn.cluster import KMeans

def get_com_col(cluster, centroids):
    """
    Extracts most common colors from a KMeans clustering result 
    and visualizes them as a color bar.

    Args:
        cluster (KMeans): Fitted KMeans object containing cluster labels.
        centroids (np.ndarray): Array of cluster centroid colors.

    Returns:
        com_col_list (list): List of most common colors (RGB) 
                             with a presence of > 8% of the picture.
    """
    com_col_list = []

    # Histogram of cluster labels
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins=labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    # Blank rectangle for color bar visualization
    rect = np.zeros((50, 300, 3), dtype=np.uint8)

    # Sort colors by percentage
    colors = sorted(
        [(percent, color) for (percent, color) in zip(hist, centroids)], key= lambda x:x[0]
        )

    start = 0
    for percent, color in colors:
        if percent > 0.08: # Consider only colors with more than 8% presence in picture
            com_col_list.append(color)
        end = start + (percent * 300)
        cv2.rectangle(
            rect,
            (int(start), 0),
            (int(end), 50),
            color.astype("uint8").tolist(),
            -1,
        )
        start = end
    return com_col_list

def get_multicolor_rate2(im, mask, n=5):
    """
    Computes a measure of color diversity using KMeans clustering.
    Returns the maximum Euclidean distance between the common colors, with
    a higher number being very different colors and a lower number being very similar colors.

    Args:
        im (np.ndarray): Input image
        mask (np.ndarray): Input mask.
        n (int, optional): Number of clusters for KMeans. Default is 5.

    Returns:
        np.max(dist_list) (float): Maximum Euclidian distance between the dominant colors.
        np.nan: If there are too few pixels or not enough colors.
    """
    # Scale down image to speed up computation
    im = resize(im, (im.shape[0] // 4, im.shape[1] // 4), anti_aliasing=True)

    # Drop alpha channel if present
    if im.shape[2] == 4:
        im = im[:, :, :3]

    # Ensure mask is in 2D
    if mask.ndim == 3:
        mask = mask[..., 0]

    # Downscale mask and ensure binary values
    mask = resize(
        mask, (mask.shape[0] // 4, mask.shape[1] // 4), anti_aliasing=True
    ) > 0.5  

    # Copy image and zero out pixels in image that is outside the mask
    im2 = im.copy()
    im2[mask == 0] = 0

    columns = im.shape[0]
    rows = im.shape[1]
    col_list = []

    # Collect masked pixel colors
    for i in range(columns):
        for j in range(rows):
            if mask[i][j]:
                col_list.append(im2[i][j] * 256)

    # Not enough pixels for KMeans
    if len(col_list) < n:
        return np.nan

    # Cluster colors using KMeans
    cluster = KMeans(n_clusters=n, n_init=10).fit(col_list)
    com_col_list = get_com_col(cluster, cluster.cluster_centers_)

    dist_list = []
    m = len(com_col_list)

    # Not enough colors
    if m <= 1:
        return np.nan 

    # Compute pairwise Eucledian distance between most common colors
    for i in range(0, m - 1):
        j = i + 1
        col_1 = com_col_list[i]
        col_2 = com_col_list[j]
        dist_list.append(
            np.sqrt(
                (col_1[0] - col_2[0]) ** 2
                + (col_1[1] - col_2[1]) ** 2
                + (col_1[2] - col_2[2]) ** 2
            )
        )

    if len(dist_list) == 0:
        return np.nan

    return np.max(dist_list)