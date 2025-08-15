import joblib
import json
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from src.utils.logger import logger
import dagshub
import mlflow

dagshub.init(repo_owner='airlangga-hub', repo_name='ml-ops-pipeline', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/airlangga-hub/ml-ops-pipeline.mlflow")

try:

  logger.info("Loading trained pipeline...")

  final_pipeline = joblib.load('models/pipeline.joblib')

  logger.info("Loading processed data...")

  X_val, y_val = joblib.load('data/val_data.joblib')

  logger.info("Evaluating model...")

  with mlflow.start_run(run_name="model_evaluation"):
    predictions = final_pipeline.predict(X_val)

    r2 = r2_score(y_val, predictions)
    rmse = root_mean_squared_error(y_val, predictions)
    mae = mean_absolute_error(y_val, predictions)

    # Log metrics to MLflow
    mlflow.log_metrics({
        "val_r2": r2,
        "val_rmse": rmse,
        "val_mae": mae
    })

    logger.info(f"Model evaluation results - R2: {r2:.2f}, RMSE: {rmse:.0f}, MAE: {mae:.0f}")

    logger.info("Saving evaluation metrics...")

    with open("models/metrics.json", "w") as f:
      json.dump({"R2": r2, "RMSE": rmse, "MAE": mae}, f)

    logger.info("Evaluation metrics saved successfully!")

except Exception as e:
  logger.error(f"Error in evaluate_model.py: {e}")