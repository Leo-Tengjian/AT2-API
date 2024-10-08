from fastapi import FastAPI, HTTPException
from starlette.responses import JSONResponse
from pydantic import BaseModel
import joblib
from joblib import load
import pandas as pd
import xgboost as xgb
from prophet import Prophet
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

xgb_model = joblib.load('models/xgboost_model.pkl')
prophet_model = joblib.load('models/prophet_model.pkl')
label_encoders = joblib.load('models/label_encoders.joblib')

@app.get("/", response_class=JSONResponse)
async def read_main():
    info = {
        "project_description": "This API serves predictions using XGBoost and Prophet models for sales revenue.",
        "endpoints": [
            {"path": "/predict/xgboost/", "method": "POST", "description": "Predicts sales revenue."},
            {"path": "/predict/prophet/", "method": "POST", "description": "Forecasts next 7 days' sales revenue."}
        ],
        "expected_input": {
            "xgboost": {"date": "YYYY-MM-DD", "store_id": "store identifier", "item_id": "item identifier"},
            "prophet": {"date": "YYYY-MM-DD"}
        },
        "output_format": {
            "xgboost": "prediction result",
            "prophet": "list of daily forecasts for the next 7 days"
        },
    }
    return info

@app.get("/health/", response_class=JSONResponse)
async def health_check():
    return JSONResponse(content={"message": "All ready to go!"}, status_code=200)

class XGBRequest(BaseModel):
    date: str
    store_id: str
    item_id: str

@app.post("/predict/xgboost/", response_class=JSONResponse)
async def predict_xgboost(request: XGBRequest):
    try:
        features = pd.DataFrame([{
            "date": datetime.strptime(request.date, "%Y-%m-%d"),
            "store_id": label_encoders['store_id'].transform([request.store_id])[0],  
            "item_id": label_encoders['item_id'].transform([request.item_id])[0]      
        }])
        
        features['year'] = features['date'].dt.year
        features['month'] = features['date'].dt.month
        features['day'] = features['date'].dt.day
        
        feature_columns = ['year', 'month', 'day', 'store_id', 'item_id']
        
        prediction = xgb_model.predict(features[feature_columns])
        
        return JSONResponse(content={
            "date": request.date,
            "store_id": request.store_id,  
            "item_id": request.item_id,    
            "predicted_sales": prediction[0] if len(prediction) > 0 else None
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class ProphetRequest(BaseModel):
    date: str  

@app.post("/predict/prophet/", response_class=JSONResponse)
async def predict_prophet(request: ProphetRequest):

    start_date = pd.Timestamp(request.date)
    
    future = pd.DataFrame({"ds": pd.date_range(start=start_date + pd.Timedelta(days=1), periods=7)})

    forecast = prophet_model.predict(future)

    result = {row['ds'].strftime('%Y-%m-%d'): row['yhat'] for row in forecast[['ds', 'yhat']].to_dict(orient='records')}

    return JSONResponse(content=result)
