from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from src.utils.logger import logger

app = FastAPI()

pipeline = joblib.load('models/pipeline.joblib')

class PredictionRequest(BaseModel):
  Condition2: str
  BldgType: str
  RoofMatl: str
  BsmtQual: str
  BsmtExposure: str
  KitchenQual: str
  SaleType: str
  OverallQual: int
  GrLivArea: int

  class Config:
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

  try:
    logger.info(f"Received prediction request: {request.model_dump()}")
    input_df = pd.DataFrame([request.model_dump()])
    prediction = pipeline.predict(input_df)
    result = float(prediction[0])
    logger.info(f"Prediction result: {result}")
    return {"prediction": result}

  except Exception as e:
    logger.error(f"Error occurred during prediction: {e}")
    return {"error": str(e)}
  