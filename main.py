import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GroupKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, classification_report, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

def main(features_path, prediction_results_path, model_path, load_model):
    """
    Docstring for main
    
    :param features_path: Path to the features csv used as input to the model (e.g. ./data/features.csv).
    :param prediction_results_path: Path to save the output predictions of the model (e.g. ./result/predictions/predictions_MODEL.csv).
    :param model_path: Path to save or load the trained model (e.g. ./result/predictions/predictions_MODEL.csv).
    :param load_model: Boolean to train the model and save it to model_path if False, load it from model_path if True. 
    """

    load_feat = pd.read_csv(features_path)

    # Features
    feature_columns = ['asymmetry_np_centroid', 'border_contours', 'color']

    # Drop rows with NaN in any of the feature columns - there was 3 with at least 1 NaN value
    df_features = df_features.dropna(subset=feature_columns)

    # Reassign X, y, img_ids
    X = df_features[feature_columns].to_numpy() # Independent variables
    y = df_features['cancerous'].to_numpy() # Dependent variables
    img_ids = df_features['img_id'].to_numpy()

    # split data 80/20
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, img_ids, test_size=0.2, stratify=y, random_state=42)

    # Extract patient ID's from img_id
    def get_patient_id(img_id):
        return '_'.join(img_id.split('_')[:2])

    # This is used to avoid data leakage
    df_features['patient_id'] = df_features['img_id'].apply(get_patient_id)

    # Creating a dict to map img_id to patient_id - this is used below, but also for later use for GroupKFold
    img_to_patient = dict(zip(df_features['img_id'] , df_features['patient_id']))

    # Getting patient ID's for temp set using the dict
    patient_ids_train_temp = np.array([img_to_patient[img_id] for img_id in ids_temp])

    # GroupKFold: 5 folds on TRAINING data only (using PATIENT IDs as groups)
    patient_ids_train_temp = np.array([img_to_patient[img_id] for img_id in ids_train])

    gkf = GroupKFold(n_splits = 5) # Amount of splits


    # Display each fold. Myabe not needed, but displayed to see some insights to the different folds
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=patient_train)):
        train_patients = np.unique(patient_train[train_idx])
        val_patients = np.unique(patient_train[val_idx])
        overlap = len(np.intersect1d(train_patients, val_patients))

    # Scale model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if load_model:
        # load the model
        pass
    else:
        ## kNN ##

        # Create pipeline with scaler and k-NN
        pipeline = Pipeline([('scaler', StandardScaler()),('knn', KNeighborsClassifier())])

        # Parameters to test
        param_grid = {
            'knn__n_neighbors': [5, 15, 25, 35, 50]
            }

        # Grid search with GroupKFold

        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv = gkf, 
            scoring = 'roc_auc', 
            n_jobs = -1, 
            verbose=1
            )

        # Fit - groups must be passed as a separate argument
        # The trick: GridSearchCV expects groups in the fit method
        grid_search.fit(X_train, y_train, groups=patient_train)
        best_k = grid_search.best_params_["knn__n_neighbors"]
        pass

    # test the classifier.


    # write test results to CSV.



if __name__ == "__main__":
    features_path = "./data/features.csv"
    prediction_results_path = "./result/predictions/predictions_MODEL.csv"
    model_path = "./result/predictions/predictions_MODEL.csv"
    load_model = False

    main(features_path, prediction_results_path,model_path,load_model)