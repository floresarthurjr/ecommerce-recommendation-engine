"""
Preprocessing module for the eCommerce Recommendation Engine.
Replicates the exact feature engineering pipeline from training so inference inputs match what the model expects.
"""

import json
import numpy as np
import pandas as pd

# Encoding mappings
GENDER_MAP = {'Male': 1, 'Female': 0}
SUBSCRIPTION_MAP = {'Yes': 1, 'No': 0}
DISCOUNT_MAP = {'Yes': 1, 'No': 0}
SIZE_ORDER = {'S': 0, 'M': 1, 'L': 2, 'XL': 3}
FREQ_ORDER = {
    'Annually': 0, 'Quarterly': 1, 'Every 3 Months': 2,
    'Monthly': 3, 'Bi-Weekly': 4, 'Fortnightly': 5, 'Weekly': 6
}

# The 12 user profile features the model uses
PROFILE_COLS = [
    'Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases',
    'Gender', 'Size', 'Frequency of Purchases', 'Subscription Status',
    'Discount Applied', 'engagement_score', 'spending_tier', 'customer_maturity'
]

# Take a raw user profile dict (human-readable values) and return an encoded dict matching the notebook's encoding scheme
def encode_user_input(raw_input: dict) -> dict:
    encoded = {}

    # Pass-through numerical fields
    encoded['Age'] = float(raw_input['Age'])
    encoded['Purchase Amount (USD)'] = float(raw_input['Purchase Amount (USD)'])
    encoded['Review Rating'] = float(raw_input['Review Rating'])
    encoded['Previous Purchases'] = int(raw_input['Previous Purchases'])

    # Binary encoding
    encoded['Gender'] = GENDER_MAP[raw_input['Gender']]
    encoded['Subscription Status'] = SUBSCRIPTION_MAP[raw_input['Subscription Status']]
    encoded['Discount Applied'] = DISCOUNT_MAP[raw_input['Discount Applied']]

    # Ordinal encoding
    encoded['Size'] = SIZE_ORDER[raw_input['Size']]
    encoded['Frequency of Purchases'] = FREQ_ORDER[raw_input['Frequency of Purchases']]

    return encoded

# Apply engineered features and scaling to match the training pipeline
def engineer_features(encoded: dict, scaler, scaler_new, bin_edges: dict) -> dict:
    # Scale original numerical features
    numerical_cols = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
    raw_numericals = np.array([[encoded[c] for c in numerical_cols]])
    scaled_numericals = scaler.transform(raw_numericals)[0]

    for i, col in enumerate(numerical_cols):
        encoded[col] = scaled_numericals[i]

    # Compute engineered features
    # engagement_score
    raw_prev_purchases = int(raw_numericals[0][3])  # unscaled value
    encoded['engagement_score'] = float(
        raw_prev_purchases + encoded['Subscription Status'] + encoded['Discount Applied']
    )

    # spending_tier: using exact bin edges from training (bin_edges.json)
    raw_purchase = float(raw_numericals[0][1])  # unscaled Purchase Amount
    encoded['spending_tier'] = _assign_spending_tier(raw_purchase, bin_edges['spending_tier'])

    # customer_maturity: using exact bin edges from training (bin_edges.json)
    encoded['customer_maturity'] = _assign_customer_maturity(raw_prev_purchases, bin_edges['customer_maturity'])

    # Scale engineered features
    eng_cols = ['engagement_score', 'spending_tier', 'customer_maturity']
    raw_engineered = np.array([[encoded[c] for c in eng_cols]])
    scaled_engineered = scaler_new.transform(raw_engineered)[0]

    for i, col in enumerate(eng_cols):
        encoded[col] = scaled_engineered[i]

    return encoded

# Extract only the 12 profile columns the model uses
def build_user_feature_vector(encoded: dict) -> dict:
    return {col: encoded[col] for col in PROFILE_COLS}

# Load the exact pd.qcut tercile bin edges saved from training
def load_bin_edges(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

# Assign spending tier using exact bin edges from training
def _assign_spending_tier(purchase_amount: float, bin_edges: list) -> int:
    if purchase_amount < bin_edges[1]:
        return 0
    elif purchase_amount < bin_edges[2]:
        return 1
    else:
        return 2

# Assign customer maturity using exact bin edges from training
def _assign_customer_maturity(previous_purchases: int, bin_edges: list) -> int:
    if previous_purchases < bin_edges[1]:
        return 0
    elif previous_purchases < bin_edges[2]:
        return 1
    else:
        return 2