"""
Recommendation engine for the eCommerce Recommendation System.
Loads model artifacts once at startup which mirrors the inference logic from notebook.
"""

import json
import pickle
from pathlib import Path
import pandas as pd

from preprocessing import (
    encode_user_input,
    engineer_features,
    build_user_feature_vector,
    load_bin_edges,
)

# Default artifacts directory (relative to this file)
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

# Loads artifacts and generates ranked product recommendations
class RecommendationEngine:

    def __init__(self, artifacts_dir: str = None):
        artifacts = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
        self._load_artifacts(artifacts)

    # Load all model artifacts into memory
    def _load_artifacts(self, artifacts: Path):
        with open(artifacts / "model5_xgboost_ranker.pkl", "rb") as f:
            self.model = pickle.load(f)

        with open(artifacts / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        with open(artifacts / "scaler_new.pkl", "rb") as f:
            self.scaler_new = pickle.load(f)

        with open(artifacts / "item_features_encoded.pkl", "rb") as f:
            self.item_features = pickle.load(f)

        with open(artifacts / "feature_cols.json", "r") as f:
            self.feature_cols = json.load(f)

        self.bin_edges = load_bin_edges(artifacts / "bin_edges.json")

        # All 25 items the model knows about
        self.all_items = list(self.item_features.index)
    # Generate top-K recommendations for a customer profile
    def recommend(self, raw_input: dict, top_k: int = 10) -> list[dict]:
        # Encode raw input
        encoded = encode_user_input(raw_input)

        # Engineer features and scale
        encoded = engineer_features(encoded, self.scaler, self.scaler_new, self.bin_edges)

        # Extract the 12 user profile features
        user_feats = build_user_feature_vector(encoded)

        # Score every item
        item_scores = {}
        for item in self.all_items:
            row = user_feats.copy()

            # Concatenate item features with item_ prefix
            item_feats_dict = self.item_features.loc[item].to_dict()
            item_feats_dict = {f"item_{k}": v for k, v in item_feats_dict.items()}
            row.update(item_feats_dict)

            # Build DataFrame with exact column order from training
            row_df = pd.DataFrame([row])[self.feature_cols]
            prob = self.model.predict_proba(row_df)[0][1]
            item_scores[item] = float(prob)

        # Rank by purchase probability, return top-K
        ranked = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"item": item, "score": round(score, 4)} for item, score in ranked[:top_k]]