"""Application configuration loaded from environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

APP_NAME: str = os.getenv("APP_NAME", "PS Fraud Detector Backend")
DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))

BASELINE_MODEL_DIR = BASE_DIR / os.getenv("BASELINE_MODEL_DIR", "app/models/artefacts")
TRANSFORMER_MODEL_DIR = BASE_DIR / os.getenv(
    "TRANSFORMER_MODEL_DIR", "app/models/artefacts/transformer"
)
DATA_DIR = BASE_DIR / "data"
SAMPLE_DATA_PATH = DATA_DIR / "sample_hinglish.csv"

DEFAULT_LANG: str = os.getenv("DEFAULT_LANG", "hi")  # hi | en

# LLM explanation settings (optional)
LLM_API_KEY: str | None = os.getenv("LLM_API_KEY")
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Google Gemini AI help settings (optional)
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_BASE_URL: str = os.getenv(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/models",
)

# Scoring thresholds
SAFE_THRESHOLD: float = 0.30
SCAM_THRESHOLD: float = 0.70

# Model weight blend: combined = ML_WEIGHT * ml_prob + RULE_WEIGHT * rule_score
ML_WEIGHT: float = 0.55
RULE_WEIGHT: float = 0.45

LABELS = {0: "Safe", 1: "Scam"}
SCORE_LABELS = [
    (0.00, 0.30, "Safe"),
    (0.30, 0.70, "Suspicious"),
    (0.70, 1.01, "Scam"),
]
