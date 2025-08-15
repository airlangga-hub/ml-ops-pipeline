import joblib
from pipeline import create_pipeline
import json
import yaml
from src.utils.logger import logger
import dagshub
import mlflow
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

dagshub.init(repo_owner='airlangga-hub', repo_name='ml-ops-pipeline', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/airlangga-hub/ml-ops-pipeline.mlflow")

try:
  logger.info("Loading processed data...")

  X_train, y_train = joblib.load('data/train_data.joblib')

  logger.info("Loading best_params.json and params.yaml ...")

  with open("models/best_params.json", "r") as f:
    hyperparams = json.load(f)

  with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

  logger.info("Training the pipeline...")

  with mlflow.start_run(run_name="model_training"):

    final_pipeline = create_pipeline(X_train.select_dtypes(include="number").columns.tolist(),
                                      X_train.select_dtypes(include="object").columns.tolist(),
                                      hyperparams)

    final_pipeline.fit(X_train, y_train)

    logger.info("Saving trained pipeline...")

    joblib.dump(final_pipeline, 'models/pipeline.joblib')

    logger.info("Trained pipeline saved successfully!")

    logger.info("Evaluating model on training data...")

    predictions = final_pipeline.predict(X_train)

    r2 = r2_score(y_train, predictions)
    rmse = root_mean_squared_error(y_train, predictions)
    mae = mean_absolute_error(y_train, predictions)

    # Log the model, metrics and params
    mlflow.log_artifact("models/pipeline.joblib")
    mlflow.log_params(hyperparams)
    mlflow.log_metrics({
        "train_r2": r2,
        "train_rmse": rmse,
        "train_mae": mae
    })

except Exception as e:
  logger.error(f"Error in train_model.py: {e}")