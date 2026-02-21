from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "Status"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Loan_Default.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"

ARTIFACT_CANDIDATES = [
    PROJECT_ROOT / "models",
    WORKSPACE_ROOT / "models",
]


def resolve_artifact_dir() -> Path:
    for path in ARTIFACT_CANDIDATES:
        model_path = path / "best_model.pkl"
        preprocessor_path = path / "preprocessor.pkl"
        if (
            model_path.exists()
            and preprocessor_path.exists()
            and model_path.stat().st_size > 0
            and preprocessor_path.stat().st_size > 0
        ):
            return path
    return PROJECT_ROOT / "models"
