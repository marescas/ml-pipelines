import pandas as pd
from config import *
import joblib
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = pd.read_pickle(FILE_TEST_PATH)
    model = joblib.load(MODEL_SAVED_PATH)
    test_score = model.score(data[FEATURES], data[LABEL])
    with open(f"{METRICS_PATH}/test_metrics.txt", "w") as f:
        f.write(f"Test score {test_score} \n")
        f.write(
            f'Precision-macro score {precision_score(y_true=data[LABEL], y_pred=model.predict(data[FEATURES]), average="macro")} \n')
        f.write(
            f'recall-macro score {recall_score(y_true=data[LABEL], y_pred=model.predict(data[FEATURES]), average="macro")} \n')
