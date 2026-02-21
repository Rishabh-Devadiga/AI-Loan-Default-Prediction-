from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from src.config import RANDOM_STATE, RAW_DATA_PATH, TEST_SIZE, TARGET_COLUMN, resolve_artifact_dir
from src.data_processing.preprocess import build_preprocessor, split_features_target


def _safe_roc_auc(model, X_test, y_test) -> float:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        if probs.ndim == 2 and probs.shape[1] > 1:
            return float(roc_auc_score(y_test, probs[:, 1]))
    return float("nan")


def train() -> pd.DataFrame:
    data = pd.read_csv(RAW_DATA_PATH)
    X, y, target = split_features_target(data, target_column=TARGET_COLUMN)

    # Remove obvious identifier fields from training inputs.
    for identifier in ("ID", "id"):
        if identifier in X.columns:
            X = X.drop(columns=[identifier])

    preprocessor, _, _ = build_preprocessor(X)
    X_proc = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_proc,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "logistic": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=350, random_state=RANDOM_STATE),
    }

    rows = []
    best_name = None
    best_score = -np.inf
    best_model = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        roc = _safe_roc_auc(model, X_test, y_test)

        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc,
        }
        rows.append(row)

        if np.isfinite(roc) and roc > best_score:
            best_score = roc
            best_name = name
            best_model = model

    if best_model is None:
        best_name = "logistic"
        best_model = models[best_name]

    artifact_dir = resolve_artifact_dir()
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, Path(artifact_dir) / "preprocessor.pkl")
    for name, model in models.items():
        joblib.dump(model, Path(artifact_dir) / f"{name}.pkl")
    joblib.dump(best_model, Path(artifact_dir) / "best_model.pkl")

    metrics_df = pd.DataFrame(rows).set_index("model").sort_values("roc_auc", ascending=False)
    metrics_df.to_csv(Path(artifact_dir) / "model_metrics_summary.csv")

    print(f"Trained on target: {target}")
    print(f"Best model: {best_name} (roc_auc={best_score:.6f})")
    print(f"Artifacts saved in: {artifact_dir}")
    return metrics_df


if __name__ == "__main__":
    train()
