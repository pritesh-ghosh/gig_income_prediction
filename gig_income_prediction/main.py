"""
Main entrypoint for Gig Income Prediction project.

This script:
  1. Loads the dataset
  2. Runs feature engineering
  3. Trains models
  4. Saves artifacts (metrics + best model)
"""

import sys
from gig_income_prediction.models.train import load_data, train_and_evaluate
from gig_income_prediction.features.feature_engineering import engineer_features

# Path to dataset
DATA_PATH = "gig_income_prediction/data/gig_worker_income_stability_fixed_500.xlsx"

def main():
    try:
        print("📂 Loading dataset...")
        df = load_data(DATA_PATH)
        print(f"✅ Loaded dataset with {len(df)} rows and {len(df.columns)} columns")

        print("⚙️ Running training pipeline...")
        train_and_evaluate(df)

        print("🎉 Training complete. Artifacts saved in ./artifacts/")

    except Exception as e:
        print("❌ Error in main.py:", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
