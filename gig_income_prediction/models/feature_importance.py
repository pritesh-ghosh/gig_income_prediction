# gig_income_prediction/models/feature_importance.py
'''
import joblib
import pandas as pd
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "best_model_Lasso.pkl"


def compute_feature_importance(top_n=20):
    """
    Load best pipeline and compute feature importances with correct feature names.
    """
    # Load pipeline
    pipe = joblib.load(MODEL_PATH)

    # Extract fitted preprocessor and model
    preprocessor = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]

    # Get transformed feature names (handles drop/expansion correctly)
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        # Fallback if sklearn version doesn’t support this
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if hasattr(trans, "get_feature_names_out"):
                feature_names.extend(trans.get_feature_names_out(cols))
            else:
                feature_names.extend(cols)

    # Get importances
    if hasattr(model, "coef_"):  # linear models
        importances = model.coef_.ravel()
    elif hasattr(model, "feature_importances_"):  # tree-based
        importances = model.feature_importances_
    else:
        raise ValueError("Model does not support feature importance extraction.")

    # Safety check
    if len(feature_names) != len(importances):
        print(f"⚠️ Mismatch: {len(feature_names)} features vs {len(importances)} importances")
        min_len = min(len(feature_names), len(importances))
        feature_names = feature_names[:min_len]
        importances = importances[:min_len]

    # Build DataFrame
    feat_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", key=abs, ascending=False)

    return feat_df.head(top_n), feat_df


if __name__ == "__main__":
    top_feats, full_feats = compute_feature_importance()

    # Print top features
    print("\nTop Features:")
    print(top_feats.to_string(index=False))

    # Save full ranking
    out_path = ARTIFACTS_DIR / "feature_importance.csv"
    full_feats.to_csv(out_path, index=False)
    print(f"\n📂 Full feature importance saved to: {out_path}")
'''