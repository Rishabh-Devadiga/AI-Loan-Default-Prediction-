from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import TARGET_COLUMN


def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def infer_target_column(df: pd.DataFrame, preferred: str = TARGET_COLUMN) -> str:
    if preferred in df.columns:
        return preferred
    for candidate in ("Status", "default", "loan_default", "target"):
        if candidate in df.columns:
            return candidate
    return df.columns[-1]


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, list, list]:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols


def split_features_target(data: pd.DataFrame, target_column: str | None = None):
    target = target_column or infer_target_column(data)
    X = data.drop(columns=[target])
    y = pd.to_numeric(data[target], errors="coerce")
    valid_mask = ~y.isna()
    return X.loc[valid_mask].reset_index(drop=True), y.loc[valid_mask].astype(int).reset_index(drop=True), target
