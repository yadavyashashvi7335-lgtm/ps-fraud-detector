"""TF-IDF + Logistic Regression baseline model.

Trains on a small labelled set (UCI SMS Spam Collection + bundled Hinglish examples)
and persists the vectorizer + classifier as joblib artefacts for fast demo startup.
"""

import os
import csv
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from app.config import BASELINE_MODEL_DIR, SAMPLE_DATA_PATH
from app.preprocessing import preprocess

VECTORIZER_PATH = BASELINE_MODEL_DIR / "tfidf_vectorizer.joblib"
CLASSIFIER_PATH = BASELINE_MODEL_DIR / "logreg_classifier.joblib"
PIPELINE_PATH = BASELINE_MODEL_DIR / "baseline_pipeline.joblib"


def _load_labelled_data(path: Path) -> Tuple[list, list]:
    """Load labelled data from a CSV with columns: text,label[,lang]."""
    texts, labels = [], []
    if not path.exists():
        return texts, labels
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = (row.get("text") or "").strip()
            l = (row.get("label") or "").strip()
            if not t or not l:
                continue
            texts.append(t)
            labels.append(1 if l.lower() in ("scam", "spam", "1") else 0)
    return texts, labels


def _default_sample_data() -> Tuple[list, list]:
    """Fallback tiny labelled Hinglish/English set so the demo works immediately."""
    texts = [
        # Scams
        "आपका खाता ब्लॉक हो जाएगा, तुरंत OTP भेजें",
        "आपका खाता बंद हो जाएगा, पिन शेयर करें",
        "KYC अपडेट करें वरना खाता फ्रीज हो जाएगा, लिंक पर क्लिक करें",
        "आपने लॉटरी जीती है, इनाम क्लेम करने के लिए लिंक खोलें",
        "घर से नौकरी, डेली पैसे, लिंक पर रजिस्टर करें",
        "UPI रिफंड के लिए अपना यूपीआई पिन शेयर करें",
        "आपका पेटीएम अकाउंट सस्पेंड हो गया, वेरिफाई करें",
        "फ्री रिफंड 5000 रुपये, यहां क्लिक करें",
        "Job offer: work from home, daily earning, join now link",
        "Your account will be blocked, send OTP immediately",
        "Congratulations you won a lottery, click here to claim",
        "Share your UPI PIN for instant refund",
        # Safe
        "नमस्ते, आप कैसे हैं?",
        "मीटिंग कल सुबह 10 बजे है",
        "कृपया दस्तावेज जमा करें",
        "आपका भुगतान सफल रहा",
        "नमस्ते सर, कल मिलते हैं",
        "डिनर प्लान के लिए तैयार हैं?",
        "Hello, are you coming to the office tomorrow?",
        "Your OTP for login is 123456 (do not share)",  # safe: informs, doesn't ask
        "Project deadline is Friday",
        "पेमेंट रिसीव हुआ, धन्यवाद",
    ]
    labels = [1] * 12 + [0] * 10
    return texts, labels


def train(texts: list, labels: list, save: bool = True) -> Pipeline:
    """Train and optionally persist the baseline pipeline."""
    # Use preprocessed text (cleaned + transliterated) for consistency
    processed = [preprocess(t)[0] for t in texts]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            lowercase=True,
        )),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipeline.fit(processed, labels)

    if save:
        BASELINE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline.named_steps["tfidf"], VECTORIZER_PATH)
        joblib.dump(pipeline.named_steps["clf"], CLASSIFIER_PATH)
        joblib.dump(pipeline, PIPELINE_PATH)

    return pipeline


def load_or_train(data_path: Path = SAMPLE_DATA_PATH) -> Pipeline:
    """Load persisted artefacts, or train a fresh baseline on bundled data."""
    BASELINE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if PIPELINE_PATH.exists():
        return joblib.load(PIPELINE_PATH)

    texts, labels = _load_labelled_data(data_path)
    if not texts:
        texts, labels = _default_sample_data()
    else:
        # Augment with defaults if the loaded set is tiny
        d_texts, d_labels = _default_sample_data()
        texts = texts + d_texts
        labels = labels + d_labels

    return train(texts, labels, save=True)


def predict_proba(pipeline: Pipeline, text: str) -> float:
    """Return probability of class 1 (scam)."""
    processed = preprocess(text)[0]
    proba = pipeline.predict_proba([processed])[0]
    # class 1 = scam
    classes = list(pipeline.classes_)
    idx = classes.index(1) if 1 in classes else int(np.argmax(classes))
    return float(proba[idx])
