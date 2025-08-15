import joblib
import pandas as pd
from src.utils.logger import logger
import yaml
from sklearn.model_selection import train_test_split

try:
  logger.info("Loading data and selecting columns...")

  with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

  # test.csv doesn't have SalePrice, so we only use train.csv for train and val
  train = pd.read_csv('./data/train.csv', usecols=params["selected_features"] + ["SalePrice"])

  logger.info("Splitting data...")

  X_train, X_val, y_train, y_val = train_test_split(
      train.drop("SalePrice", axis=1),
      train["SalePrice"],
      test_size=0.2,
      random_state=42
  )

  logger.info("Saving data...")

  joblib.dump((X_train, y_train), 'data/train_data.joblib')
  joblib.dump((X_val, y_val), 'data/val_data.joblib')

  logger.info("Data saved successfully!")

except Exception as e:
  logger.error(f"Error in prepare_data.py: {e}")