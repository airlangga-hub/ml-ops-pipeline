import joblib
import pandas as pd
from src.utils.logger import training_logger
import yaml
from sklearn.model_selection import train_test_split

try:
  training_logger.info("Loading data and selecting columns...")

  with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

  # test.csv doesn't have SalePrice, so we split train.csv for train and test
  train = pd.read_csv('./data/train.csv', usecols=params["selected_features"] + ["SalePrice"])

  training_logger.info("Splitting data...")

  X_train, X_test, y_train, y_test = train_test_split(
      train.drop("SalePrice", axis=1),
      train["SalePrice"],
      test_size=0.2,
      random_state=42
  )

  training_logger.info("Saving data...")

  joblib.dump((X_train, y_train), 'data/train_data.joblib')
  joblib.dump((X_test, y_test), 'data/test_data.joblib')

  training_logger.info("Data saved successfully!")

except Exception as e:
  training_logger.error(f"Error in prepare_data.py: {e}")