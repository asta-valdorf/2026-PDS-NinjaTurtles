# OBS, code takes approx. 40 minutes to load CSV

import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import pandas as pd
import os

from src.feature_A import mean_asymmetry
from src.feature_B import convexity_score
from src.feature_C import get_multicolor_rate2

data_path = "../data/"

def features_csv(meta_data , data_path):
    """
    Makes CSV file of skin lesions features (ABC) and wether the skin lesions is cancerous or non-cancerous
    Skin lesions without corresponding masks, are not accounted for in the features CSV
    """

    def load_image_and_mask(image_id, data_path = data_path):
        """ 
        Loads single skin lesion image and its corresponding mask

        """
        img_path = data_path + "imgs/"
        mask_path = data_path + "masks/"

        file_im = img_path + image_id
        file_mask = (mask_path + image_id).replace(".png", "_mask.png")
        im = plt.imread(file_im)
        mask = plt.imread(file_mask)

        if mask.ndim == 3:
            mask = mask[..., 0]

        if im.shape[:2] != mask.shape[:2]:
            mask = resize(mask, im.shape[:2], anti_aliasing=False)
        
        return im, mask

    def load_metadata(meta_data, data_path):
        metadata_path = data_path + meta_data
        metadata = pd.read_csv(metadata_path)

        metadata["cancerous"] = metadata["diagnostic"].isin(["BCC", "MEL", "SCC"]).astype(int)

        # Filter to only rows where the mask file actually exists
        mask_path = data_path + "masks/"
        metadata = metadata[metadata["img_id"].apply(
            lambda x: os.path.exists(mask_path + x.replace(".png", "_mask.png"))
        )]

        print(f"Found {len(metadata)} images with masks out of {len(pd.read_csv(data_path + meta_data))} total")

        return metadata
    
    def return_features(row):
        img_id = row["img_id"]
        im, mask = load_image_and_mask(img_id, data_path)
        
        # let's extract the other columns in this row that we're interested in
        diagnostic = row["cancerous"]

        color = get_multicolor_rate2(im , mask) # good rep
        border = convexity_score(mask)
        mean_asymmetry_score = mean_asymmetry(mask)


        # compute your features
        feats = {
            "img_id": img_id,
            "cancerous": diagnostic,
            "asymmetry": mean_asymmetry_score,
            "border": border, 
            "color": color,

        } # notice how the identifying info (eg. img_id etc are not relevant once we have extractd the features)

        return feats

    def make_csv(sampled_dfs , output_dir = "output/"):
        output_path = os.path.join(output_dir, "features.csv")
        features_df = pd.DataFrame(sampled_dfs.apply(return_features, axis=1).to_list())
        return features_df.to_csv(output_path, index=True)
    
    sampled_dfs = load_metadata(meta_data, data_path)
    make_csv(sampled_dfs , "data/")

features_csv("metadata.csv" , data_path = "data/")
