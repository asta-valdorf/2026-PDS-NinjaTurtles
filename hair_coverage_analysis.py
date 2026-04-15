import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import random

# code for hair and pen detection + removal

# full dataset
df = pd.read_csv('md.csv')

IMG_DIR = Path("imgs")
MASK_DIR = Path("masks")

img_id = df["img_id"].iloc[0]
img_path = IMG_DIR / img_id
mask_path = MASK_DIR / img_id.replace(".png", "_mask.png")

# hair detection and removal for light and dark hair
# group into three groups based on amount of coverage

def hair_coverage(img_gray, kernel_size=9, threshold=10):
    """
    Estimate the fraction of the image covered by hair.
    Uses combined blackhat+tophat for complete dark/light hair detection.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
    tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
    
    # Combine both dark- and light-hair responses
    combined = cv2.add(blackhat, tophat)
    
    _, hair_mask = cv2.threshold(combined, threshold, 255, cv2.THRESH_BINARY)
    hair_mask = cv2.medianBlur(hair_mask, 3)  # denoise
    
    total_area = img_gray.shape[0] * img_gray.shape[1]
    hair_area = np.count_nonzero(hair_mask)
    
    return hair_area / total_area


def removeHair_auto(img_org, img_gray, lesion_mask=None):
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

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
    tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)

    _, black_mask = cv2.threshold(blackhat, black_threshold, 255, cv2.THRESH_BINARY)
    _, top_mask = cv2.threshold(tophat, top_threshold, 255, cv2.THRESH_BINARY)

    # Make light-hair detection stricter to avoid detecting light spots on skin instead of hair
    #top_mask = cv2.medianBlur(top_mask, 3)

    black_score = np.count_nonzero(black_mask)
    top_score = np.count_nonzero(top_mask)

    # Only choose light mode if it clearly wins
    if top_score > 1.3 * black_score:
        mode = "light"
        response = tophat
        mask = top_mask
    else:
        mode = "dark"
        response = blackhat
        mask = black_mask

    if lesion_mask is not None:
        if lesion_mask.shape != mask.shape:
            lesion_mask = cv2.resize(lesion_mask, (mask.shape[1], mask.shape[0]))
        if lesion_mask.dtype != mask.dtype:
            lesion_mask = lesion_mask.astype(mask.dtype)
        mask = cv2.bitwise_and(mask, lesion_mask)

    img_out = cv2.inpaint(img_org, mask, radius, cv2.INPAINT_TELEA)

    return mode, response, mask, img_out, coverage, black_score, top_score


coverages = []
df_meta = pd.read_csv('md.csv')

IMG_DIR = Path("images")
MASK_DIR = Path("masks")

# based on histogram
def label_coverage(coverage):
    if coverage < 0.01:
        return "low"
    elif coverage < 0.08:
        return "medium"
    else:
        return "high"


# loop through dataset
data = []

for img_id in df_meta["img_id"]:
    img_path = str(IMG_DIR / img_id)
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Warning: Could not read image {img_path}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cov = hair_coverage(gray)
    coverages.append(cov)
    label = label_coverage(cov)

    data.append({
        "image_path": str(img_path),
        "coverage": cov,
        "label": label
    })

df = pd.DataFrame(data)
print(df["label"].value_counts())       # check distribution


# plots

def show_removal_examples(df, label, n=3):
    subset = df[df["label"] == label]

    print(f"{label} removal samples: {len(subset)} images")

    if len(subset) == 0:
        print(f"No samples for label: {label}")
        return

    samples = subset.sample(min(n, len(subset)), random_state=None)

    for _, row in samples.iterrows():
        img_bgr = cv2.imread(row["image_path"])
        if img_bgr is None:
            print(f"Warning: Could not read image {row['image_path']}")
            continue

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mode, response, mask, img_out, coverage, black_score, top_score = removeHair_auto(img_bgr, gray)

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        fig.suptitle(
            f"{label} | coverage={coverage:.3f} | mode={mode} | "
            f"black_score={black_score} | top_score={top_score}"
        )

        axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original")
        axes[0].axis("off")

        if response is not None:
            axes[1].imshow(response, cmap="gray")
        else:
            axes[1].text(0.5, 0.5, "Skipped", ha="center", va="center")
        axes[1].set_title("Response")
        axes[1].axis("off")

        if mask is not None:
            axes[2].imshow(mask, cmap="gray")
        else:
            axes[2].text(0.5, 0.5, "Skipped", ha="center", va="center")
        axes[2].set_title("Final mask")
        axes[2].axis("off")

        axes[3].imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
        axes[3].set_title("Inpainted")
        axes[3].axis("off")

        plt.tight_layout()
        plt.show()


# Show removal pipeline examples side by side
show_removal_examples(df, "medium")
show_removal_examples(df, "high")


# plot
plt.hist(coverages, bins=30)
plt.yscale('log')
plt.title("Hair Coverage Distribution")
plt.xlabel("Coverage")
plt.ylabel("Frequency")
plt.show()