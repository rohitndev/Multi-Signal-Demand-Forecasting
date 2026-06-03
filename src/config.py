"""Shared paths and small helpers used across the project."""
from pathlib import Path
import os

from dotenv import load_dotenv

# Load .env once, at import time, so every module sees the keys.
load_dotenv()

# ---- Project paths (resolved relative to this file, so cwd doesn't matter) ----
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MLRUNS_DIR = ROOT / "mlruns"

# Common filenames
CLEAN_CSV = DATA_PROCESSED / "clean.csv"
MODEL_CKPT = MODELS_DIR / "tft_model.ckpt"

# Make sure folders exist on import.
for _p in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, MLRUNS_DIR):
    _p.mkdir(parents=True, exist_ok=True)


def get_env(key: str, default: str = "") -> str:
    """Return an environment variable, or a default if it is missing/empty."""
    return os.getenv(key) or default
