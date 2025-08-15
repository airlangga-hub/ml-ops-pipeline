from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from src.utils.logger import logger

app = FastAPI()

class PredictionRequest(BaseModel):
  # Categorical features (object type)
  Condition2: str
  BldgType: str
  RoofMatl: str
  BsmtQual: str
  BsmtExposure: str
  KitchenQual: str
  SaleType: str

  # Numerical features (int64 type)
  OverallQual: int
  GrLivArea: int

  class Config:
      # Example of valid input data
      json_schema_extra = {
          "example": {
              "Condition2": "Norm",
              "BldgType": "1Fam",
              "RoofMatl": "CompShg",
              "BsmtQual": "TA",
              "BsmtExposure": "No",
              "KitchenQual": "TA",
              "SaleType": "WD",
              "OverallQual": 7,
              "GrLivArea": 1500
          }
      }

@app.post("/predict")
def predict(request: PredictionRequest):

  logger.info(f"Received prediction request: {request.model_dump()}")

  logger.info("Loading pipeline for inference...")

  pipeline = joblib.load('models/pipeline.joblib')

  prediction = pipeline.predict([request.model_dump()])

  logger.info(f"Prediction result: {prediction[0]}")

  return {"prediction": prediction[0]}