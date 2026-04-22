
import cv2
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path


from hair_analysis import hair_removal
from pen_analysis import pen_removal

IMG_DIR = Path('data/imgs')
OUTPUT_DIR = 'data/imgs_clean'
os.makedirs(OUTPUT_DIR, exist_ok = True)


for filename in tqdm(os.listdir(IMG_DIR)):
    input_path = IMG_DIR / filename
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    # read image (BGR)
    img = cv2.imread(input_path)

    if img is None:
        continue

    # removal pipeline 
    img_clean = hair_removal(img)
    img_clean = pen_removal(img_clean)

    # save cleaned image
    cv2.imwrite(str(output_path), img_clean)


