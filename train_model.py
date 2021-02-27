from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pandas as pd
from config import *
import os
import joblib

if __name__ == '__main__':
    data = pd.read_pickle(FILE_TRAIN_PATH)
    model = SVC()
    parameters = {
        'kernel': ('linear', 'rbf'),
        'C': [ 10, 100, 1000, 10000]
    }
    clf = GridSearchCV(model, parameters, cv=10, n_jobs=-1, verbose=2)
    clf.fit(data[FEATURES], data[LABEL])
    if not os.path.exists(METRICS_PATH):
        os.mkdir(METRICS_PATH)
    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)
    with open(f"{METRICS_PATH}/train_metrics.txt", "w") as f:
        f.write(f"Train score is {clf.best_score_} \n")
        f.write(f"Best params are {clf.best_params_}")
    joblib.dump(clf.best_estimator_, MODEL_SAVED_PATH)
