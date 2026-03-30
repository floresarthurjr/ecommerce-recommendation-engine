# eCommerce Recommendation Engine — Deployment Guide

## Overview

FastAPI application that serves product recommendations using an XGBoost Ranker model. Accepts a customer profile and returns the top-K products ranked by predicted purchase probability.

**Champion Model:** XGBoost Ranker (NDCG@10 = 0.176, +7.9% over Popularity Baseline)

## Project Structure

```
app/
├── artifacts/                         # Model and preprocessing assets
│   ├── model5_xgboost_ranker.pkl      # Trained XGBoost model
│   ├── scaler.pkl                     # StandardScaler for 4 numerical features
│   ├── scaler_new.pkl                 # StandardScaler for 3 engineered features
│   ├── item_features_encoded.pkl      # 25 items × 27 one-hot encoded features
│   ├── feature_cols.json              # 39 feature columns in training order
│   └── bin_edges.json                 # Exact pd.qcut tercile boundaries
├── main.py                            # FastAPI app with /health and /recommend endpoints
├── recommend.py                       # Loads artifacts, scores user-item pairs, ranks results
├── preprocessing.py                   # Feature engineering pipeline (replicates notebook logic)
├── requirements.txt                   # Pinned Python dependencies
├── Dockerfile                         # Containerized deployment
└── README.md                          # This file
```

## Quick Start (Local)

**Prerequisites:** Python 3.11+

```bash
cd app
pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive Swagger UI.

## Quick Start (Docker)

```bash
cd app
docker build -t recommendation-engine .
docker run -p 8000:8000 recommendation-engine
```

## API Endpoints

### GET /health

Returns the status of the API and confirms artifacts are loaded.

```json
{
  "status": "healthy",
  "model": "XGBoost Ranker",
  "items_available": 25,
  "features_expected": 39
}
```

### POST /recommend

Accepts a customer profile and returns top-K product recommendations.

**Request body:**

```json
{
  "age": 35,
  "purchase_amount": 60.0,
  "review_rating": 3.5,
  "previous_purchases": 20,
  "gender": "Male",
  "size": "M",
  "frequency": "Monthly",
  "subscription": "Yes",
  "discount": "Yes",
  "top_k": 10
}
```

**Response:**

```json
{
  "recommendations": [
    {"item": "Shirt", "score": 0.5161},
    {"item": "Backpack", "score": 0.5117},
    {"item": "Pants", "score": 0.418}
  ],
  "customer_profile": { ... }
}
```

**Validation rules:**
- age: 18–70
- purchase_amount: 0–100 USD
- review_rating: 1.0–5.0
- previous_purchases: 1–50
- gender: Male / Female
- size: S / M / L / XL
- frequency: Annually / Quarterly / Every 3 Months / Monthly / Bi-Weekly / Fortnightly / Weekly
- subscription: Yes / No
- discount: Yes / No
- top_k: 1–25 (default 10)

## Inference Pipeline

The API replicates the exact feature engineering sequence from the training notebook:

1. **Encode** raw customer input (binary, ordinal mappings from notebook Cell 37)
2. **Scale** 4 numerical features using the saved StandardScaler (notebook Cell 39)
3. **Engineer** 3 features: engagement_score, spending_tier, customer_maturity (notebook Cell 56)
4. **Scale** engineered features using the second StandardScaler (notebook Cell 58)
5. **Score** all 25 user-item pairs by concatenating user + item features and calling predict_proba (notebook Cell 89)
6. **Rank** by purchase probability and return top-K

## MLOps Practices

### Reproducible Environment
- `requirements.txt` with pinned dependency versions
- `Dockerfile` for containerized deployment ensuring consistent runtime

### Config-Driven Runs
- `model_configs.json` in the models directory stores champion hyperparameters and metadata
- Bin edges and feature column order loaded from JSON artifacts rather than hardcoded

### CI/CD
- GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push to main:
  - **Lint:** ruff checks Python code quality
  - **Smoke test:** starts the app and verifies /health returns 200

### Versioning and Rollback
- Model artifacts stored in `models/` with explicit naming (e.g., `model5_xgboost_ranker.pkl`)
- Previous model version (`model5_xgboost_hybrid.pkl`) retained in repo history for rollback
- Git commit history preserves full lineage of changes

## Production Monitoring Plan

If this system were deployed to production, the following monitoring controls would apply. This is structured as a control plan: what to monitor, how to detect issues, and what action to take.

### Input Monitoring
- **What:** Log every incoming request (customer profile fields)
- **Why:** Detect distribution drift — if incoming age, purchase amount, or frequency distributions shift significantly from training data, model predictions become unreliable
- **Threshold:** Flag when any input feature's mean or variance deviates more than 2 standard deviations from training distribution
- **Action:** Trigger a review; if sustained, retrain on updated data

### Output Monitoring
- **What:** Log every prediction (recommended items and scores)
- **Why:** Detect score compression — if the model starts returning similar scores for all items, it has lost discriminative power
- **Threshold:** Flag when the spread between the highest and lowest score in a recommendation set falls below 0.05
- **Action:** Investigate feature pipeline for silent errors; check if new input patterns are outside training distribution

### Fairness Monitoring
- **What:** Track recommendation quality (NDCG@10) segmented by gender and spending tier monthly
- **Why:** The bias audit (Step 5) identified a 15.5% NDCG gap between Male and Female users and 18.1% gap across spending tiers
- **Threshold:** Flag if gender gap exceeds 20% or spending tier gap exceeds 25%
- **Action:** Apply post-processing re-ranking calibration (13.6% adjustment documented in bias audit) or retrain with resampled data

### System Health
- **What:** /health endpoint uptime, response latency, error rates
- **Threshold:** Alert if /health returns non-200, or if p95 latency exceeds 500ms, or if error rate exceeds 1%
- **Action:** Check artifact loading, restart container, review logs
