from pathlib import Path

import joblib
import pandas as pd

from src.config import RAW_DATA_PATH, TARGET_COLUMN, resolve_artifact_dir


def _artifact_path(filename: str) -> Path:
    return Path(resolve_artifact_dir()) / filename


def load_model(model_path: str | Path | None = None):
    path = Path(model_path) if model_path else _artifact_path("best_model.pkl")
    return joblib.load(path)


def load_preprocessor(preprocessor_path: str | Path | None = None):
    path = Path(preprocessor_path) if preprocessor_path else _artifact_path("preprocessor.pkl")
    return joblib.load(path)


def training_feature_columns(data_path: str | Path = RAW_DATA_PATH, target_column: str = TARGET_COLUMN) -> list[str]:
    df = pd.read_csv(data_path, nrows=1)
    return [c for c in df.columns if c != target_column]


def preprocess_input(input_data: dict, feature_columns: list[str]) -> pd.DataFrame:
    template = pd.read_csv(RAW_DATA_PATH, nrows=1).iloc[0].to_dict()
    template.pop(TARGET_COLUMN, None)
    row = {col: input_data.get(col, template.get(col, None)) for col in feature_columns}
    return pd.DataFrame([row], columns=feature_columns)


def predict(input_data: dict):
    model = load_model()
    preprocessor = load_preprocessor()
    feature_cols = training_feature_columns()

    input_df = preprocess_input(input_data, feature_cols)
    X = preprocessor.transform(input_df)
    label = int(model.predict(X)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] > 1:
            probability = float(probs[0, 1])

    return {"prediction": label, "default_probability": probability}
