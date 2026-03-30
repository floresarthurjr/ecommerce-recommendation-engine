"""
FastAPI application for the eCommerce Recommendation Engine.

Exposes a /recommend endpoint that accepts a customer profile
and returns top-K product recommendations ranked by purchase probability.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional
from recommend import RecommendationEngine

# Request/Response schemas

class Gender(str, Enum):
    male = "Male"
    female = "Female"

class Size(str, Enum):
    s = "S"
    m = "M"
    l = "L"
    xl = "XL"

class Frequency(str, Enum):
    annually = "Annually"
    quarterly = "Quarterly"
    every_3_months = "Every 3 Months"
    monthly = "Monthly"
    bi_weekly = "Bi-Weekly"
    fortnightly = "Fortnightly"
    weekly = "Weekly"

class YesNo(str, Enum):
    yes = "Yes"
    no = "No"

class CustomerProfile(BaseModel):
    age: int = Field(..., ge=18, le=70, description="Customer age", examples=[35])
    purchase_amount: float = Field(..., gt=0, le=100, description="Purchase amount in USD", examples=[60.0])
    review_rating: float = Field(..., ge=1.0, le=5.0, description="Review rating", examples=[3.5])
    previous_purchases: int = Field(..., ge=1, le=50, description="Number of previous purchases", examples=[20])
    gender: Gender = Field(..., description="Customer gender", examples=["Male"])
    size: Size = Field(..., description="Preferred size", examples=["M"])
    frequency: Frequency = Field(..., description="Purchase frequency", examples=["Monthly"])
    subscription: YesNo = Field(..., description="Subscription status", examples=["Yes"])
    discount: YesNo = Field(..., description="Discount applied", examples=["Yes"])
    top_k: Optional[int] = Field(10, ge=1, le=25, description="Number of recommendations to return")

class Recommendation(BaseModel):
    item: str
    score: float

class RecommendResponse(BaseModel):
    recommendations: list[Recommendation]
    customer_profile: dict


# App setup

app = FastAPI(
    title="eCommerce Recommendation Engine",
    description=(
        "Product recommendation API powered by an XGBoost Ranker model."
        "Accepts a customer profile and returns top-K product recommendations ranked by purchase probability."
    ),
    version="1.0.0",
)

# Load model artifacts once at startup
engine = RecommendationEngine()


@app.get("/health")
# Verify the API and model artifacts are loaded
def health_check():
    return {
        "status": "healthy",
        "model": "XGBoost Ranker",
        "items_available": len(engine.all_items),
        "features_expected": len(engine.feature_cols),
    }


@app.post("/recommend", response_model=RecommendResponse)
# Generate product recommendations for a customer profile
def recommend(profile: CustomerProfile):
    # Map Pydantic model to the raw_input dict that preprocessing expects
    raw_input = {
        "Age": profile.age,
        "Purchase Amount (USD)": profile.purchase_amount,
        "Review Rating": profile.review_rating,
        "Previous Purchases": profile.previous_purchases,
        "Gender": profile.gender.value,
        "Size": profile.size.value,
        "Frequency of Purchases": profile.frequency.value,
        "Subscription Status": profile.subscription.value,
        "Discount Applied": profile.discount.value,
    }

    try:
        results = engine.recommend(raw_input, top_k=profile.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

    return RecommendResponse(
        recommendations=[Recommendation(**r) for r in results],
        customer_profile=raw_input,
    )