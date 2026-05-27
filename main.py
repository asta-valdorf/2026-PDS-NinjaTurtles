import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split, GroupKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def main(features_path, prediction_results_path, model_path, load_model):
    """
    Docstring for main
    
    :param features_path: Path to the features csv used as input to the model (e.g. ./data/features.csv).
    :param prediction_results_path: Path to save the output predictions of the model (e.g. ./results/predictions/predictions_MODEL.csv).
    :param model_path: Path to save or load the trained model (e.g. ./results/models/predictions_MODEL.csv).
    :param load_model: Boolean to train the model and save it to model_path if False, load it from model_path if True. 
    """

    df_features = pd.read_csv(features_path)

    feature_columns = ['asymmetry_np_centroid', 'border_contours', 'color']

    # drop rows with nan in any of feature columns - there was 3 with at least 1 NaN value
    df_features = df_features.dropna(subset=feature_columns)

    # reassign X, y, img_ids
    X = df_features[feature_columns].to_numpy() # Independent variables
    y = df_features['cancerous'].to_numpy() # Dependent variables
    img_ids = df_features['img_id'].to_numpy()

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, img_ids, test_size=0.2, stratify=y, random_state=42)

    # extract patient ID's from img_id
    def get_patient_id(img_id):
        return '_'.join(img_id.split('_')[:2])

    # avoid data leakage
    df_features['patient_id'] = df_features['img_id'].apply(get_patient_id)

    # creating dict to map img_id to patient_id - used to show split and for groupkfold
    img_to_patient = dict(zip(df_features['img_id'] , df_features['patient_id']))

    gkf = GroupKFold(n_splits = 5)

    patient_train = np.array([img_to_patient[i] for i in ids_train])

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups = patient_train)):
        train_patients = np.unique(patient_train[train_idx])
        val_patients = np.unique(patient_train[val_idx])
        overlap = len(np.intersect1d(train_patients, val_patients))

    if load_model:
        model = joblib.load(model_path)
        pass

    else:
        ## KNN ##

        pipeline = Pipeline([('scaler', StandardScaler()),('knn', KNeighborsClassifier())])

        param_grid = {
            'knn__n_neighbors': [5, 15, 25, 35, 50]
            }

        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv = gkf, 
            scoring = 'roc_auc', 
            n_jobs = -1, 
            verbose=1
            )

        grid_search.fit(X_train, y_train, groups=patient_train)

        ## RANDOM FOREST ##

        pipeline_rf = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ))
        ])

        param_grid_rf = {
            "rf__n_estimators": [199, 499],
            "rf__max_depth": [2, 4 , 6, 8, 10],
            "rf__min_samples_leaf": [1, 5],
            "rf__max_features": [1,2,3]
        }

        grid_search_rf = GridSearchCV(
            estimator=pipeline_rf,
            param_grid=param_grid_rf,
            cv=gkf,
            scoring="roc_auc",
            n_jobs=-1
        )

        grid_search_rf.fit(X_train, y_train, groups=patient_train)

        ## LOGISTIC REGRESSION ##
        log_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(class_weight="balanced", max_iter=1000))
        ])

        param_grid_log = {
            'model__C': [ 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        }

        grid_search_log = GridSearchCV(
            estimator=log_pipeline,
            param_grid=param_grid_log,
            cv=gkf,
            scoring='roc_auc',
            return_train_score=False,
            n_jobs=-1
        )

        grid_search_log.fit(X_train, y_train, groups=patient_train)

        ## SAVE MODELS ##

        models = {}
        model_paths = {
            "knn": "./results/models/knn_model.joblib",
            "rf": "./results/models/rf_model.joblib",
            "log": "./results/models/log_model.joblib"
        }

        # after knn
        models["knn"] = grid_search.best_estimator_

        # after rf
        models["rf"] = grid_search_rf.best_estimator_

        # after log
        models["log"] = grid_search_log.best_estimator_

        # save models
        os.makedirs("./results/models", exist_ok=True)

        joblib.dump(models["knn"], model_paths["knn"])
        joblib.dump(models["rf"], model_paths["rf"])
        joblib.dump(models["log"], model_paths["log"])

        # best model #
        model = models["rf"]

        pass


    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]


    results_df = pd.DataFrame({
        "img_id": ids_test,
        "y_true": y_test,
        "y_pred": y_pred,
        "y_probability": y_proba
    })


    os.makedirs(os.path.dirname(prediction_results_path), exist_ok=True)
    results_df.to_csv(prediction_results_path, index=False)



if __name__ == "__main__":
    features_path = "./data/base_features.csv"
    prediction_results_path = "./results/predictions/predictions_MODEL.csv"
    model_path = "./results/models/rf_model.joblib"
    load_model = False

    main(features_path, prediction_results_path,model_path,load_model)