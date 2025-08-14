from re import X
import joblib
from pipeline import create_pipeline
import json
import yaml
from utils.logger import training_logger
import dagshub
import mlflow

dagshub.init(repo_owner='airlangga-hub', repo_name='ml-ops-pipeline', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/airlangga-hub/ml-ops-pipeline.mlflow")

try:
  training_logger.info("Loading processed data...")

  X_train, y_train = joblib.load('data/train_processed.joblib')

  training_logger.info("Loading best_params.json and params.yaml ...")

  with open("models/best_params.json", "r") as f:
    hyperparams = json.load(f)

  with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

  training_logger.info("Training the pipeline...")

  with mlflow.start_run(run_name="model_training"):

    final_pipeline = create_pipeline(X_train.select_dtypes(include="number").columns.tolist(),
                                      X_train.select_dtypes(include="object").columns.tolist(),
                                      hyperparams)

    final_pipeline.fit(X_train, y_train)

    training_logger.info("Saving trained pipeline...")

    joblib.dump(final_pipeline, 'models/pipeline.joblib')

    training_logger.info("Trained pipeline saved successfully!")

    # Log the model and params
    mlflow.log_artifact("models/pipeline.joblib")
    mlflow.log_params(hyperparams)

except Exception as e:
  training_logger.error(f"Error in train_model.py: {e}")