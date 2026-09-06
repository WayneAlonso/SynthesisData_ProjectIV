from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
PROCESSED_DIR = ARTIFACTS_DIR / "processed"
LOGS_DIR = ARTIFACTS_DIR / "logs"
CONFIGS_DIR = PROJECT_ROOT / "configs"
