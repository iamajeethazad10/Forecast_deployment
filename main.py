from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd

# Load models
prophet_model = joblib.load('prophet_model.pkl')
optimized_prophet_model = joblib.load('optimized_prophet_model.pkl')
random_forest_model = joblib.load('random_forest_model.pkl')

app = FastAPI()

class ForecastInput(BaseModel):
    data: dict  # Define your input data structure, e.g., features, time series

@app.post("/forecast/prophet")
def forecast_prophet(input_data: ForecastInput):
    df = pd.DataFrame(input_data.data)
    forecast = prophet_model.predict(df)
    return {"forecast": forecast.to_dict()}

@app.post("/forecast/optimized_prophet")
def forecast_optimized_prophet(input_data: ForecastInput):
    df = pd.DataFrame(input_data.data)
    forecast = optimized_prophet_model.predict(df)
    return {"forecast": forecast.to_dict()}

@app.post("/forecast/random_forest")
def forecast_random_forest(input_data: ForecastInput):
    df = pd.DataFrame(input_data.data)
    forecast = random_forest_model.predict(df)
    return {"forecast": forecast.tolist()}
