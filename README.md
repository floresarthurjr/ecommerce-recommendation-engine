# ecommerce-recommendation-engine

## Project Overview

An eCommerce recommendation engine that predicts and ranks the Top-K products a customer is most likely to purchase. Five modeling approaches were tested, from a popularity baseline to an XGBoost Ranker, with the goal of outperforming generic "most popular" recommendations through personalization.

## Business Objective

Increase Average Order Value (AOV) and customer retention through personalized product recommendations, replacing the current one-size-fits-all approach.

## Dataset

- **Source:** Customer Shopping Trends Dataset (Kaggle, by Sourav Banerjee)
- **Size:** 3,900 customers x 18 features (+ 4 engineered features)
- **Key characteristics:** 1 transaction per customer, 96% sparse interaction matrix, 68% Male / 32% Female
- **Link:** https://www.kaggle.com/datasets/iamsouravbanerjee/customer-shopping-trends-dataset

## Results

**Champion Model: XGBoost Ranker**

| Model | NDCG@10 | vs Baseline |
|-------|---------|-------------|
| Model 1: Popularity Baseline | 0.163 | -- |
| Model 2: K-Means Cluster (k=3) | 0.169 | +3.6% |
| Model 3: Collaborative Filtering | 0.163 | No lift |
| Model 4: Content-Based (Adapted) | 0.175 | +7.3% |
| Model 5: XGBoost Ranker | 0.176 | +7.9% |

- **Primary metric:** NDCG@10 of 0.176 (+7.9% over baseline)
- **Recall@10:** 0.390 (39% of customers see their preferred product in the top 10)
- **Fairness audited** across gender, age, and spending level with Demographic Parity and Disparate Impact metrics. Mitigations proposed for gender (15.5% gap) and spending tier (18.1% gap) disparities.

## Technical Metrics

- **Primary:** NDCG@K (K=5, K=10) as tollgate metric
- **Supporting:** Precision@K, Recall@K, RMSE, Silhouette Score
- **Fairness:** Demographic Parity Difference, Disparate Impact Ratio, Equalized Odds
- **Explainability:** SHAP (global + beeswarm), LIME, PDP, ICE

## Repository Structure

```
├── notebooks/          # Colab notebook (single notebook, all steps)
├── src/                # Python scripts
├── data/               # Dataset downloaded at runtime via kagglehub
├── models/             # Saved models (.pkl) and configs (.json)
├── reports/            # Technical and business presentation decks
├── app/                # Deployment code (future)
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

## How to Reproduce

1. Open `notebooks/Arthur_C_Flores_Jr__Pillar_5_Capstone_Project.ipynb` in Google Colab
2. Run all cells sequentially (the notebook downloads the dataset via kagglehub at runtime)
3. All models, visualizations, and analysis will be reproduced

## Dependencies

See `requirements.txt`. Key libraries: pandas, numpy, scikit-learn, xgboost, shap, lime, matplotlib, seaborn.

## Project Status

- [x] Step 1: Problem Understanding and Framing
- [x] Step 2: Data Collection and Understanding
- [x] Step 3: Data Preprocessing, Applied EDA and Feature Engineering
- [x] Step 4: Model Implementation and Comparison
- [x] Step 5: Critical Thinking, Ethical AI and Bias Auditing
- [x] Step 6: Final Presentation and Communication (Technical + Business decks)
- [x] Step 7: GitHub Profile and Upload
- [ ] Step 8: Deployment and MLOps (optional)
- [ ] Step 9: Use of Generative AI (optional)

## Author

Arthur C. Flores Jr.
AIM AIML Capstone Project
