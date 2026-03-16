# ecommerce-recommendation-engine

## Project Overview
An eCommerce recommendation engine that predicts and ranks the Top-K products a customer is most likely to purchase, using a hybrid approach combining collaborative filtering, content-based filtering, K-Means clustering, and XGBoost as a meta-learner

## Business Objective
Increase Average Order Value (AOV) and customer retention through personalized product recommendations

## Dataset
- Customer Shopping Trends Dataset (Kaggle)
- 3,900 customers × 18 features
- Source: https://www.kaggle.com/datasets/iamsouravbanerjee/customer-shopping-trends-dataset

## Repository Structure
```
├── notebooks/          # Jupyter notebooks
├── src/                # Python scripts
├── data/               # Datasets
├── models/             # Saved models and configs
├── reports/            # Presentations and visuals
├── app/                # Deployment code
├── requirements.txt
└── README.md
```

## Technical Metrics
- Primary: Precision@K, Recall@K, NDCG@K (K=5, K=10)
- Tollgate Metric: NDCG@10
- Supporting: RMSE, Silhouette Score

## Status
- [x] Phase A: Foundation & Provisioning (Steps 1 & 2)
- [ ] Phase B: Analysis & Feature Engineering (Step 3)
- [ ] Phase C: Modeling & Algorithmic Governance (Steps 4 & 5)
- [ ] Phase D: Communication & Packaging (Steps 6, 7, 8, 9)

## Author

Arthur C. Flores Jr.
Director of Business Excellence
AIM AIML Program: Capstone Project

```
