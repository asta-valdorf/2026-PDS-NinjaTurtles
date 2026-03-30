# Code taken from exercise session 05_Feature_Extraction

import cv2
import numpy as np
from skimage.transform import resize
from sklearn.cluster import KMeans

def get_com_col(cluster, centroids):
    com_col_list = []
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins=labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    rect = np.zeros((50, 300, 3), dtype=np.uint8)
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)], key= lambda x:x[0])
    start = 0
    for percent, color in colors:
        if percent > 0.08:
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
    im = resize(im, (im.shape[0] // 4, im.shape[1] // 4), anti_aliasing=True)

    # Fix 1: drop alpha channel if present
    if im.shape[2] == 4:
        im = im[:, :, :3]

    # Fix: collapse mask to 2D if it has multiple channels (e.g. RGB or RGBA)
    if mask.ndim == 3:
        mask = mask[..., 0]

    mask = resize(
        mask, (mask.shape[0] // 4, mask.shape[1] // 4), anti_aliasing=True
    ) > 0.5  # <-- binariser efter resize

    im2 = im.copy()
    im2[mask == 0] = 0

    columns = im.shape[0]
    rows = im.shape[1]
    col_list = []
    for i in range(columns):
        for j in range(rows):
            if mask[i][j]:
                col_list.append(im2[i][j] * 256)

    # Fix 2: return early if too few pixels for KMeans
    if len(col_list) < n:
        return 0.0

    if len(col_list) == 0:
        return 0.0  # <-- returner float, ikke ""

    cluster = KMeans(n_clusters=n, n_init=10).fit(col_list)
    com_col_list = get_com_col(cluster, cluster.cluster_centers_)

    dist_list = []
    m = len(com_col_list)

    if m <= 1:
        return 0.0  # <-- returner float, ikke ""

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
        return 0.0

    return np.max(dist_list)