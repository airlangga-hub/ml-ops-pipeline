import joblib
import json
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from src.utils.logger import training_logger
import dagshub
import mlflow

dagshub.init(repo_owner='airlangga-hub', repo_name='ml-ops-pipeline', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/airlangga-hub/ml-ops-pipeline.mlflow")

try:

  training_logger.info("Loading trained pipeline...")

  final_pipeline = joblib.load('models/pipeline.joblib')

  training_logger.info("Loading processed data...")

  X_test, y_test = joblib.load('data/test_data.joblib')

  training_logger.info("Evaluating model...")

  with mlflow.start_run(run_name="model_evaluation"):
    predictions = final_pipeline.predict(X_test)

    r2 = r2_score(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    # Log metrics to MLflow
    mlflow.log_metrics({
        "test_r2": r2,
        "test_rmse": rmse,
        "test_mae": mae
    })

    training_logger.info(f"Model evaluation results - R2: {r2:.2f}, RMSE: {rmse:.0f}, MAE: {mae:.0f}")

    training_logger.info("Saving evaluation metrics...")

    with open("models/metrics.json", "w") as f:
      json.dump({"R2": r2, "RMSE": rmse, "MAE": mae}, f)

    training_logger.info("Evaluation metrics saved successfully!")

except Exception as e:
  training_logger.error(f"Error in evaluate_model.py: {e}")