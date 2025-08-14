import joblib
import pandas as pd
from utils.logger import training_logger
import yaml

try:
  training_logger.info("Loading data and selecting columns...")

  with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

  train = pd.read_csv('data/train.csv', usecols=params["selected_features"] + ["SalePrice"])
  test = pd.read_csv('data/test.csv', usecols=params["selected_features"] + ["SalePrice"])

  X_train = train.drop("SalePrice", axis=1)
  y_train = train["SalePrice"]

  X_test = test.drop("SalePrice", axis=1)
  y_test = test["SalePrice"]

  training_logger.info("Saving data...")

  joblib.dump((X_train, y_train), 'data/train_processed.joblib')
  joblib.dump((X_test, y_test), 'data/test_processed.joblib')

  training_logger.info("Data saved successfully!")

except Exception as e:
  training_logger.error(f"Error in prepare_data.py: {e}")