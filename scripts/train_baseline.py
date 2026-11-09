"""Train the TF-IDF + Logistic Regression baseline and persist artefacts.

Usage:
    python scripts/train_baseline.py [--data data/sample_hinglish.csv]
"""

import argparse
import sys
from pathlib import Path

# Allow running as a script: add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.baseline import load_or_train, train, _load_labelled_data, _default_sample_data
from app.config import SAMPLE_DATA_PATH


def main():
    parser = argparse.ArgumentParser(description="Train baseline scam detector")
    parser.add_argument("--data", default=str(SAMPLE_DATA_PATH), help="Path to labelled CSV")
    args = parser.parse_args()

    data_path = Path(args.data)
    texts, labels = _load_labelled_data(data_path)
    if not texts:
        print(f"No data found at {data_path}; using built-in sample set.")
        texts, labels = _default_sample_data()

    print(f"Training on {len(texts)} messages (scam={sum(labels)}, safe={len(labels)-sum(labels)})")
    pipeline = train(texts, labels, save=True)
    print(f"Artefacts saved to app/models/artefacts/")
    print("Done.")


if __name__ == "__main__":
    main()
