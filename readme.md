# 🏠 End-to-End MLOps Pipeline: House Price Prediction

This project implements an **end-to-end MLOps pipeline** for a *House Price Prediction* use case.
It covers **data versioning, experiment tracking, model management, and deployment** with the following tools:

- **[DVC](https://dvc.org/)** – Data & model version control
- **[MLflow](https://mlflow.org/)** – Experiment tracking & model registry
- **[DagsHub](https://dagshub.com/)** – Remote repository for datasets, models, and metrics
- **[FastAPI](https://fastapi.tiangolo.com/)** – API for model inference

Visit the **[DagsHub Repository](https://dagshub.com/airlangga-hub/ml-ops-pipeline)** or **[MLflow Dashboard](https://dagshub.com/airlangga-hub/ml-ops-pipeline.mlflow/#/experiments/0?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D)** for more details.

# 📚 Dataset
The dataset for this project can be found **[here.](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview)**

# 🚀 Pipeline Workflow
Below is a high-level overview of the pipeline workflow:
```
                                +--------------------+
                                | data/train.csv.dvc |
                                +--------------------+
                                           *
                                           *
                                           *
                                 +------------------+
                                 | data_preparation |
                               **+------------------+***
                          *****            *            *****
                     *****                  *                *****
                  ***                       *                     *****
+-----------------------+                  **                          ***
| hyperparameter_tuning |                **                              *
+-----------------------+             ***                                *
                    **             ***                                   *
                      ***        **                                      *
                         **    **                                        *
                        +-------+                                      ***
                        | train |                                 *****
                        +-------+**                          *****
                                   **                   *****
                                     **            *****
                                       **       ***
                                     +----------+
                                     | evaluate |
                                     +----------+
```

# 🗂️ Directory Structure
Below is the directory structure of the project:
```
.
├── data
│   ├── test.csv
│   ├── train.csv
│   ├── train.csv.dvc
│   ├── train_data.joblib
│   └── val_data.joblib
├── dvc.lock
├── dvc.yaml
├── logs
│   └── app.log
├── models
│   ├── best_params.json
│   ├── metrics.json
│   └── pipeline.joblib
├── notebooks
│   └── exp.ipynb
├── params.yaml
├── readme.md
├── requirements.txt
└── src
    ├── __init__.py
    ├── __pycache__
    ├── api
    ├── data
    ├── models
    └── utils

11 directories, 16 files
```