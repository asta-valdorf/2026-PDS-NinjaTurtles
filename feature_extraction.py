# Code taken from exercise session 05_feature_extraction
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os

from src.feature_A import asymmetry_np_centroid
from src.feature_B import border_irregularity
from src.feature_C import get_multicolor_rate2

data_path = "../data/"

def features_csv(meta_data , data_path):
    """
    Extracts ABC features from skin lesion images and masks and saves them into a CSV file.

    Loads images and the corresponding masks based on the given metadata.
    Only images with existing masks are used. For each image we extract the following features:
        - A1: Mean asymmetry score based on center of image
        - A2: Asymmetry score based on centroid of lesion
        - B1: Border convexity score
        - B2: Border irregularity score
        - C: Multicolor rate

    Features that couldn't be computed are stored as NaN and will be seen as an empty slot in the CSV file.

    Args:
        meta_data (str): Filename of dataset (e.g. "metadata.csv").
        data_path (str): Root path to the location of the dataset.
        
    Returns:
        None: The code loads the CSV file directly into the given datapath

    Notes:
        Code takes 20-30 minutes to run
    """

    def load_image_and_mask(image_id, data_path = data_path):
        """ 
        Loads single skin lesion image and the corresponding mask.

        Args:
            image_id (str): Image id taken from the metadata.
            data_path (str): Root path to the location of the dataset.

        Returns:
            im: np.ndarray of the loaded image
            mask: np.ndarray of the loaded mask
        """
        img_path = data_path + "imgs/"
        mask_path = data_path + "masks/"

        file_im = img_path + image_id
        file_mask = (mask_path + image_id).replace(".png", "_mask.png")
        im = plt.imread(file_im)
        mask = plt.imread(file_mask)

        if mask.ndim == 3: 
            mask = mask[..., 0] # Converts RGB mask into grayscale, by only taking first channel

        if im.shape[:2] != mask.shape[:2]:
            mask = resize(mask, im.shape[:2], anti_aliasing=False) # Resize mask dimension to image if they are different
        
        return im, mask

    def load_metadata(meta_data, data_path):
        """
        Loads the metadata as a pandas dataframe.

        Creates new column, which defines wether a lesion is cancerous or benign in binary values.

        Args:
            meta_data (str): Filename of the dataset (e.g. "metadata.csv")
            data_path (str): Root path to the location of the dataset.

        Returns:
            Dataframe from metadata with column defining cancerous or benign
        """
        metadata_path = data_path + meta_data
        metadata = pd.read_csv(metadata_path)

        # Labels a cancerous lesion with 1 and a benign lesion with 0
        metadata["cancerous"] = metadata["diagnostic"].isin(["BCC", "MEL", "SCC"]).astype(int)

        # Filter to only rows where the mask file actually exists
        mask_path = data_path + "masks/"
        metadata = metadata[metadata["img_id"].apply(
            lambda x: os.path.exists(mask_path + x.replace(".png", "_mask.png"))
        )]

        print(f"Found {len(metadata)} images with masks out of {len(pd.read_csv(data_path + meta_data))} total")

        return metadata
    
    def return_features(row):
        """
        Extracts ABC features for each skin lesion image and the corresponding mask.
        Stores the features in a dictionary

        Args:
            row (pd.Series): A single row from metadata containing "img_id" and "cancerous"
        
        Returns:
            feats (dict): A dictionary with all extracted features for every skin lesion with a mask
        """
        img_id = row["img_id"]
        im, mask = load_image_and_mask(img_id, data_path)
        
        diagnostic = row["cancerous"]

        asymmetry_score_np = asymmetry_np_centroid(mask) #A2 - Centroid (center of lesion)
        border_contours = border_irregularity(mask)  #B2 - Centroid (Only taking the largest lesion if multiple)
        color = get_multicolor_rate2(im , mask) #C - Difference between dominant colors
        

        # computing features
        feats = {
            "img_id": img_id,
            "cancerous": diagnostic,
            "asymmetry_np_centroid" : asymmetry_score_np,
            "border_contours": border_contours,
            "color": color,

        }

        return feats

    def make_csv(df , output_dir = "output/"):
        """
        Outputs the CSV file with the featues computed for the skin lesions

        Args:
            df (pd.DataFrame): The dataframe for the given CSV file
            output_dir (str): The path where the final CSV file should be loaded to

        Returns:
            The final CSV file with the diagnostic, image id and extracted features
        """
        output_path = os.path.join(output_dir, "features.csv")

        results = []

        # Manual tqdm loop
        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc="Extracting features"
        ):

            results.append(return_features(row))


        # Apply return_features to each row and collect the results as a list and converts the list into a dataframe
        features_df = pd.DataFrame(results)

        return features_df.to_csv(output_path, index=False)
    
    df = load_metadata(meta_data, data_path)
    make_csv(df , "data/")

features_csv("metadata.csv" , data_path = "data/")