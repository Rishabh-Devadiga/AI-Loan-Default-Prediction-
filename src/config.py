from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "Status"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Loan_Default.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"

ARTIFACT_CANDIDATES = [
    WORKSPACE_ROOT / "models",
    PROJECT_ROOT / "models",
]


def resolve_artifact_dir() -> Path:
    for path in ARTIFACT_CANDIDATES:
        if (path / "best_model.pkl").exists() and (path / "preprocessor.pkl").exists():
            return path
    return ARTIFACT_CANDIDATES[0]
