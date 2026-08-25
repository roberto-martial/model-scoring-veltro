

import joblib
import pandas as pd

threshold = joblib.load("models/best_threshold.pkl")
model = joblib.load("models/best_model.pkl")


## exemple fior the live

dict_deal = {
    "annual_revenue": 1000000,
    "years_in_operation": 5,
    "ebitda_margin": 0.15,
    "revenue_growth_3yr": 0.10,
    "customer_concentration": 0.20,
    "recurring_revenue_pct": 0.50,
    "management_depth_score" : 0.8,
    "industry" : "Software",
    "ownership_type" : "Private"
}

# new dictdeal for the shapley test 

dict_deal_2 = {
    "annual_revenue": 350000,
    "years_in_operation": 2,
    "ebitda_margin": 0.05,
    "revenue_growth_3yr": -0.08,
    "customer_concentration": 0.65,
    "recurring_revenue_pct": 0.20,
    "management_depth_score": 0.35,
    "industry": "Retail",
    "ownership_type": "Founder-Owned"
}


def score_new_deal(dict_deal):
    x = pd.DataFrame([dict_deal])
    prediction = model.predict_proba(x)
    print(f"Prediction proba : {prediction[0][1]:.3f}")
    if(prediction[0][1] >= threshold):
        print("Deal flagged for further review")
    else :
        print("Deal not flagged")


if __name__ == "__main__":
    score_new_deal(dict_deal)



