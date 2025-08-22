from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Basic preprocessor:
      - numeric: median impute + standardize
      - categorical: most_frequent impute + one-hot (dense)
    """
    num_feats = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_feats = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric = PipelineOrNone([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical = PipelineOrNone([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_feats),
            ("cat", categorical, cat_feats),
        ],
        remainder="drop"
    )
    return pre, num_feats, cat_feats


# Small helper so we don't import Pipeline directly in train file
from sklearn.pipeline import Pipeline as _SkPipeline
def PipelineOrNone(steps):
    return _SkPipeline(steps)


def get_models():
    
    return {
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001, max_iter=10000),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
    }
