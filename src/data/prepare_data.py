import joblib
import pandas as pd

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

X_train = train.drop("SalePrice", axis=1)
y_train = train["SalePrice"]

X_test = test.drop("SalePrice", axis=1)
y_test = test["SalePrice"]

joblib.dump((X_train, X_test, y_train, y_test), 'data/processed_data.joblib')