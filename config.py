import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

SECRET_KEY = os.getenv(
    "SECRET_KEY",
    "dev-secret-key-change-later"
)

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3"
)

DATABASE_PATH = BASE_DIR / "database" / "maintenance.db"

MODEL_PATH = BASE_DIR / "models" / "lightgbm_model.pkl"

PREPROCESSING_PATH = BASE_DIR / "models" / "preprocessing.pkl"

# Project-defined machine health thresholds
# These are not manufacturer-certified limits.

HEALTH_WARNING_THRESHOLD = 0.30
HEALTH_CRITICAL_THRESHOLD = 0.70