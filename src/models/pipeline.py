from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

def create_pipeline(num_cols, cat_cols, hyperparams):

  preprocessor = ColumnTransformer([
      ('num', Pipeline([
          ('imputer', SimpleImputer(strategy='median')),
          ('scaler', StandardScaler())
      ]), num_cols),

      ('cat', Pipeline([
          ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
          ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
      ]), cat_cols)
    ],
    remainder='passthrough')

  final_pipeline = Pipeline([
      ('preprocessor', preprocessor),
      ('xgb', XGBRegressor(**hyperparams, random_state=42))
  ])

  return final_pipeline