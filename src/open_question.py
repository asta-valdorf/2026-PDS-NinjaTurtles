from skimage.feature import local_binary_pattern
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

metadata = pd.read_csv('../data/metadata.csv')
IMG_DIR = Path('../data/imgs')
MASK_DIR = Path('../data/masks')


# LBP parameters
radius = 1
points = 8          # 3x3 matrix
method = "uniform"
# uniform with 8 neighbors = values in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

lbp_hists = []
lbp_ids = []    # store id labels
remove = []     # for empty lbp lesions
data_rows = []


for _, row in metadata.iterrows():
    img_id = row["img_id"]
    img_path = IMG_DIR / img_id
    img = cv2.imread(str(img_path))

    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask_path = MASK_DIR / img_id.replace(".png", "_mask.png")
    
    if not mask_path.exists():
        continue

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # resize mask if needed, lbp requires shape alignment
    if mask.shape != gray.shape:
        mask = cv2.resize(
            mask,
            (gray.shape[1], gray.shape[0]),
            interpolation=cv2.INTER_NEAREST         # nearest neighbor interpolation 
        )

    mask = (mask > 0).astype(np.uint8)

    # apply lbp on whole image then mask it
    lbp = local_binary_pattern(gray, points, radius, method)

    # select only lesion pixels, mask-based segmentation
    lbp_lesion = lbp[mask == 1]

    if lbp_lesion.size == 0:
        remove.append(img_id)
        continue

    # histogram 
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(
        lbp_lesion,
        bins=n_bins,
        range=(0, n_bins),
        density=True
    )


    # save results
    lbp_hists.append(lbp_hist)
    lbp_ids.append(row["img_id"])

    # build row 
    row_data = {
        "img_id": img_id,
    }

    # add LBP values
    row_data["lbp_uniform"] = lbp_hist[:-1].sum()
    row_data["lbp_complex"] = lbp_hist[-1]
    data_rows.append(row_data)

# convert to feature matrix
X_lbp = np.array(lbp_hists)


df_lbp = pd.DataFrame(data_rows)
df_lbp.to_csv("lbp_features.csv", index=False)

# get base_features csv
df_base = pd.read_csv("../data/base_features.csv")

# merge using img_id
df_combined = df_base.merge(
    df_lbp[["img_id", "lbp_uniform", "lbp_complex"]],
    on="img_id",
    how="inner"         # keeps only rows that have LBP
)

df_combined.to_csv("base_lbp_features.csv", index=False)