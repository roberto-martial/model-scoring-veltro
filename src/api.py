from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd
from src.shaps import shapley_test


app = FastAPI()

threshold = joblib.load("models/best_threshold.pkl")
model = joblib.load("models/best_model.pkl")

    
class DealData(BaseModel):
    annual_revenue: float
    years_in_operation: int
    ebitda_margin: float
    revenue_growth_3yr: float
    customer_concentration: float
    recurring_revenue_pct: float
    management_depth_score: float
    industry: str
    ownership_type: str

@app.post("/score_deal")
async def score_deal(deal: DealData):
    x = pd.DataFrame([deal.model_dump()])
    prediction = model.predict_proba(x)
    is_flagged = float(prediction[0][1]) >= float(threshold)
    return {"is_flagged": is_flagged, "probability": float(prediction[0][1])}

@app.post("/shapley_analysis")
async def shapley_analysis(deal: DealData):
    shapley_results = shapley_test(deal.model_dump())
    return {"shapley_results": shapley_results}