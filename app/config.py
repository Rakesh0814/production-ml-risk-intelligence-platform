from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "xgboost_model.json"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
FEATURE_ORDER_PATH = ARTIFACT_DIR / "feature_order.json"
DEFAULTS_PATH = ARTIFACT_DIR / "feature_defaults.json"
