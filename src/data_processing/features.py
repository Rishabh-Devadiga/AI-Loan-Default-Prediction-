import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE
from src.data_processing.preprocess import infer_target_column


def extract_features(data: pd.DataFrame, target_column: str | None = None):
    target = target_column or infer_target_column(data)
    features = data.drop(columns=[target])
    labels = pd.to_numeric(data[target], errors="coerce")
    valid_mask = ~labels.isna()
    return (
        features.loc[valid_mask].reset_index(drop=True),
        labels.loc[valid_mask].astype(int).reset_index(drop=True),
        target,
    )


def prepare_data(file_path: str, target_column: str | None = None):
    data = pd.read_csv(file_path)
    X, y, target = extract_features(data, target_column=target_column)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, target
