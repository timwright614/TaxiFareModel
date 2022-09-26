import os
from math import sqrt
import gcsfs
import joblib
import pandas as pd
from TaxiFareModel.params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH, TESTDATAPATH, PATH_TO_GCP_MODEL

from sklearn.metrics import mean_absolute_error, mean_squared_error

PATH_TO_LOCAL_MODEL = 'model.joblib'

AWS_BUCKET_TEST_PATH = "s3://wagon-public-datasets/taxi-fare-test.csv"



def load_joblib():
    fs = gcsfs.GCSFileSystem()
    path = PATH_TO_GCP_MODEL
    with fs.open(path) as f:
        return joblib.load(f)


def get_test_data():
    """method to get the training data (or a portion of it) from google cloud bucket
    To predict we can either obtain predictions from train data or from test data"""
    # Add Client() here
    """method to get the training data (or a portion of it) from google cloud bucket"""
    df = pd.read_csv(TESTDATAPATH)
    return df





def evaluate_model(y, y_pred):
    MAE = round(mean_absolute_error(y, y_pred), 2)
    RMSE = round(sqrt(mean_squared_error(y, y_pred)), 2)
    res = {'MAE': MAE, 'RMSE': RMSE}
    return res


def generate_submission_csv( kaggle_upload=False):
    df_test = get_test_data()
    pipeline = load_joblib()
    if "best_estimator_" in dir(pipeline):
        y_pred = pipeline.best_estimator_.predict(df_test)
    else:
        y_pred = pipeline.predict(df_test)
    df_test["fare_amount"] = y_pred
    df_sample = df_test[["key", "fare_amount"]]
    name = f"predictions_test_ex.csv"
    df_sample.to_csv(name, index=False)
    print("prediction saved under kaggle format")
    # Set kaggle_upload to False unless you install kaggle cli
    if kaggle_upload:
        kaggle_message_submission = name[:-4]
        command = f'kaggle competitions submit -c new-york-city-taxi-fare-prediction -f {name} -m "{kaggle_message_submission}"'
        os.system(command)


if __name__ == '__main__':

    # ⚠️ in order to push a submission to kaggle you need to use the WHOLE dataset
    generate_submission_csv( kaggle_upload=False)
