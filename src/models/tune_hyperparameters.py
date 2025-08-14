import joblib
import yaml
from utils.logger import training_logger
import optuna
from sklearn.metrics import mean_absolute_error
import json
from pipeline import create_pipeline
import dagshub
import mlflow

dagshub.init(repo_owner='airlangga-hub', repo_name='ml-ops-pipeline', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/airlangga-hub/ml-ops-pipeline.mlflow")

def objective(trial):
  hyperparams = {
    "n_estimators":
      trial.suggest_int("n_estimators",
                        params["hyperparameter_tuning"]["n_estimators"]["low"],
                        params["hyperparameter_tuning"]["n_estimators"]["high"]),
    "max_depth":
      trial.suggest_int("max_depth",
                        params["hyperparameter_tuning"]["max_depth"]["low"],
                        params["hyperparameter_tuning"]["max_depth"]["high"]),
    "learning_rate":
      trial.suggest_loguniform("learning_rate",
                              params["hyperparameter_tuning"]["learning_rate"]["low"],
                              params["hyperparameter_tuning"]["learning_rate"]["high"])
  }

  final_pipeline = create_pipeline(X_train.select_dtypes(include="number").columns.tolist(),
                                    X_train.select_dtypes(include="object").columns.tolist(),
                                    hyperparams)

  final_pipeline.fit(X_train, y_train)

  return mean_absolute_error(y_train, final_pipeline.predict(X_train))

try:
  training_logger.info("Loading processed data...")

  X_train, y_train = joblib.load('data/train_processed.joblib')

  training_logger.info("Starting hyperparameter tuning...")

  with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

  with mlflow.start_run(run_name="hyperparameter_tuning"):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=params["hyperparameter_tuning"]["n_trials"])

    best_params = study.best_params
    best_value = study.best_value

    # Log best parameters and metrics
    mlflow.log_params(best_params)
    mlflow.log_metric("best_mae", best_value)

    training_logger.info(f"Best hyperparameters found: {best_params}\nBest MAE: {best_value:.0f}")

    training_logger.info("Saving best parameters...")

    with open("models/best_params.json", "w") as f:
      json.dump(best_params, f)

    training_logger.info("Best parameters saved successfully!")

except Exception as e:
  training_logger.error(f"Error in tune_hyperparameters.py: {e}")