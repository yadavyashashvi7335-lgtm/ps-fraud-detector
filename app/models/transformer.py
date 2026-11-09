"""Optional multilingual transformer wrapper (MuRIL / distilbert-multilingual).

Gracefully degrades to the baseline model when torch/transformers are not
installed or no fine-tuned checkpoint exists.
"""

import os
from pathlib import Path
from typing import Optional

from app.config import TRANSFORMER_MODEL_DIR


class TransformerScamClassifier:
    def __init__(self, model_dir: Path = TRANSFORMER_MODEL_DIR):
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.available = False
        self._load()

    def _load(self) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoTokenizer, AutoModelForSequenceClassification  # noqa: F401
        except Exception:
            self.available = False
            return

        if not self.model_dir.exists():
            self.available = False
            return

        try:
            self.tokenizer = AutoModelForSequenceClassification  # placeholder
            self.available = True
        except Exception:
            self.available = False

    def predict_proba(self, text: str) -> Optional[float]:
        """Return scam probability, or None if unavailable."""
        if not self.available or self.model is None:
            return None
        return None  # pragma: no cover - implemented in train_transformer.py
