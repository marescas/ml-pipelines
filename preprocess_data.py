from config import *
import pandas as pd
from sklearn.model_selection import train_test_split
import os

if __name__ == '__main__':
    dataframe = pd.read_csv(FILE_RAW_PATH,
                            names=FEATURES + [LABEL])
    train, test = train_test_split(dataframe, test_size=0.2, random_state=42,stratify=dataframe[LABEL])
    if not os.path.exists(DATA_PROCESSED_PATH):
        os.mkdir(DATA_PROCESSED_PATH)
    train.to_pickle(FILE_TRAIN_PATH)
    test.to_pickle(FILE_TEST_PATH)
