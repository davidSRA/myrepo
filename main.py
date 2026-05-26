from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib


class ApiInput(BaseModel):
  ### ESCRIBA SU CÓDIGO AQUÍ ###
  features: List[float]
  ### FIN DEL CÓDIGO ###


class ApiOutput(BaseModel):
  ### ESCRIBA SU CÓDIGO AQUÍ ###
  forecast: float
  ## FIN DEL CÓDIGO ###

app = FastAPI()
model = joblib.load("model.joblib")

# Reemplace esto con su implementación:
@app.post("/predict")
def predict(data: ApiInput) -> ApiOutput:
    ### ESCRIBA SU CÓDIGO AQUÍ ###
    model = joblib.load("model.joblib")
    # Reshape the input features to a 2D array (1 sample, N features)
    features_2d = np.array(data.features).reshape(1, -1)
    predict = model.predict(features_2d).flatten()
    prediction = ApiOutput(forecast=predict[0])
    return prediction
