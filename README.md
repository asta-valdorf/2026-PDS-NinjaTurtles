# How to Install and Run the Project

### Required Folder Structure
```
2026-PDS_NinjaTurtles
├── data/                               
│   ├── annotations_combined.csv        # annotations of hair and penmarks
│   ├── features.csv
│   ├── metadata.csv                    # metadata is assumed to be inside data/
│   ├── imgs/                           # skin images of type png
│   └── masks/                          # mask images of type png
├── src/
│   ├── __init__.py
│   ├── feature_A.py
│   ├── feature_B.py
│   ├── feature_C.py  
│   ├── open_question.py               
│   ├── extract_features.py           # calls feature extraction functions and generates data/features.csv
│
├── results/
│   ├── figures/                        
│   ├── models/                         
│   ├── predictions/                    
│   └── reports/                        
│
├── main.py                             
└── README.md 
```
## How to Run:

1. Create features CSV file:
    - Run `src/extract_features.py` to extract features from images and masks
    - Ensure path to metadata, images and masks are correct
    - We extract asymmetry, border, color and texture
    - Extracted features will be saved as a CSV file inside `data/`
    - CSV file will have the name `features.csv`

2. Run `main.py` to train model:
    - Set `features_path = "./data/features.csv"`
    - Set `prediction_results_path = "./results/predictions/predictions_MODEL.csv"`
    - Set `model_path = "./results/models/rf_model.joblib"`
    - Set `load_model = False`
    - If `load_model = False`, models (k-NN, Random Forest and Logistic Regression) are trained and best parameters of each model are saved 
    - From evaluating we conclude that Random Forest is the best-performing model.
    - We already trained the model and saved the best parameters inside `results/models`

3. Run `main.py` and produce predictions:
    - Set `features_path = "./data/features.csv"`
    - Set `prediction_results_path = "./results/predictions/predictions_MODEL.csv"`
    - Set `model_path = "./results/models/rf_model.joblib"`
    - Set `load_model = True`

The CSV file with the predictions for your metadata should be saved inside `./results/predictions`

## TA evaluation workflow
1. Update paths in `src/extract_features.py``
2. Run `src/extract_features.py`
3. In `main.py` set `load_model = True`
4. Run `main.py`
5. Predictions are saved in `results/predictions`
