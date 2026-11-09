"""Fine-tune a multilingual transformer (MuRIL / distilbert-multilingual-cased).

Optional step for hours 2-5 of the 12-hour plan. Requires torch + transformers.

Usage:
    python scripts/train_transformer.py --model google/muril-base-cased --epochs 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import TRANSFORMER_MODEL_DIR, SAMPLE_DATA_PATH
from app.models.baseline import _load_labelled_data, _default_sample_data


def main():
    parser = argparse.ArgumentParser(description="Fine-tune multilingual transformer")
    parser.add_argument("--model", default="google/muril-base-cased", help="HF model id")
    parser.add_argument("--data", default=str(SAMPLE_DATA_PATH), help="Labelled CSV")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from transformers import get_linear_schedule_with_warmup
        from torch.optim import AdamW
    except Exception as e:
        print("torch/transformers not installed. Install them to fine-tune:")
        print("  pip install torch transformers")
        print(f"Error: {e}")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    texts, labels = _load_labelled_data(Path(args.data))
    if not texts:
        print("Using built-in sample set.")
        texts, labels = _default_sample_data()

    # Label map: 0=safe, 1=scam
    labels = [1 if l in (1, "scam", "spam") else 0 for l in labels]

    class ScamDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=128):
            self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
            return item

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    model.to(device)

    train_ds = ScamDataset(texts, labels, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} - avg loss: {total_loss/len(train_loader):.4f}")

    TRANSFORMER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(TRANSFORMER_MODEL_DIR)
    tokenizer.save_pretrained(TRANSFORMER_MODEL_DIR)
    print(f"Transformer saved to {TRANSFORMER_MODEL_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
