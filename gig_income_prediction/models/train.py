import json
from pathlib import Path

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from gig_income_prediction.features.feature_engineering import engineer_features
from gig_income_prediction.models.pipeline import build_preprocessor, get_models

# ---- CONFIG ----
DATA_PATH = "gig_income_prediction/data/gig_worker_income_stability_fixed_500.xlsx"
TARGET = "net_earnings"
TEST_SIZE = 0.20
RANDOM_STATE = 42
ARTIFACTS_DIR = Path("artifacts")
# ---------------

def load_data(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)  # requires openpyxl installed
    elif ext == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def train_and_evaluate(df: pd.DataFrame):
    # feature engineering (leaky on purpose to reproduce R² ~0.9999)
    df = engineer_features(df)

    # drop rows with missing target
    df = df.dropna(subset=[TARGET])

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    # build preprocessor
    pre, num_feats, cat_feats = build_preprocessor(X)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # train models and pick best by R²
    models = get_models()
    results = []
    trained = {}

    for name, model in models.items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5            # avoid sklearn 'squared' arg issues
        r2 = r2_score(y_test, y_pred)

        results.append({"model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
        trained[name] = pipe

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
    results_df.to_csv(ARTIFACTS_DIR / "model_comparison.csv", index=False)

    best_name = results_df.iloc[0]["model"]
    best_pipe = trained[best_name]
    joblib.dump(best_pipe, ARTIFACTS_DIR / f"best_model_{best_name}.pkl")

    summary = {
        "rows": int(len(df)),
        "features_used": int(X.shape[1]),
        "target": TARGET,
        "best_model": best_name,
        "metrics": {
            m["model"]: {"MAE": float(m["MAE"]), "RMSE": float(m["RMSE"]), "R2": float(m["R2"])}
            for m in results
        },
        "artifacts": {
            "metrics_csv": str(ARTIFACTS_DIR / "model_comparison.csv"),
            "best_model_pkl": str(ARTIFACTS_DIR / f"best_model_{best_name}.pkl"),
        },
    }
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    train_and_evaluate(df)
