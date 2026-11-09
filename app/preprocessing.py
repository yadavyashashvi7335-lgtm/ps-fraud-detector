"""Text preprocessing: cleaning, Hinglish transliteration, URL extraction."""

import re
from typing import List, Tuple

try:
    from indic_transliteration import sanscript
    HAS_INDIC = True
except Exception:  # pragma: no cover - optional dependency
    HAS_INDIC = False

try:
    from langdetect import detect
    HAS_LANGDETECT = True
except Exception:  # pragma: no cover - optional dependency
    HAS_LANGDETECT = False


# Common Hinglish (Roman-script Hindi) urgency / scam keywords mapped to Devanagari
# so the rule layer can match regardless of script.
HINGLISH_TO_DEVANAGARI = {
    "otp": "ओटीपी",
    "pin": "पिन",
    "password": "पासवर्ड",
    "urgent": "तुरंत",
    "immediately": "तुरंत",
    "quickly": "जल्दी",
    "jaldi": "जल्दी",
    "turant": "तुरंत",
    "abhi": "अभी",
    "block": "ब्लॉक",
    "band": "बंद",
    "kyc": "केवाईसी",
    "refund": "रिफंड",
    "lottery": "लॉटरी",
    "job": "नौकरी",
    "offer": "ऑफर",
    "click": "क्लिक करें",
    "link": "लिंक",
    "bank": "बैंक",
    "upi": "यूपीआई",
    "account": "खाता",
    "verify": "सत्यापित",
    "suspend": "निलंबित",
    "reward": "पुरस्कार",
    "prize": "इनाम",
    "winner": "विजेता",
}

URL_REGEX = re.compile(
    r"(https?://[^\s<>\"]+|www\.[^\s<>\"]+|[a-zA-Z0-9-]+\.[a-zA-Z]{2,}[^\s<>\"]*)"
)
UPI_REGEX = re.compile(r"[\w.\-]{2,}@[\w]+", re.IGNORECASE)
PHONE_REGEX = re.compile(r"(\+91[\s\-]?)?[6-9]\d{9}")


def clean_text(text: str) -> str:
    """Normalize whitespace and strip noise while preserving URLs/Hinglish tokens."""
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_urls(text: str) -> List[str]:
    """Extract candidate URLs from the raw message."""
    return [m.group(0).rstrip(".,;)") for m in URL_REGEX.finditer(text)]


def is_hinglish(text: str) -> bool:
    """Heuristic: detects Roman-script Hindi (Hinglish) by keyword hits."""
    low = text.lower()
    hits = sum(1 for k in HINGLISH_TO_DEVANAGARI if k in low)
    return hits >= 1


def transliterate_hinglish(text: str) -> str:
    """
    Transliterate Hinglish (Roman Hindi) to Devanagari when possible.
    Falls back to a built-in keyword map if indic-transliteration is unavailable.
    """
    if not text:
        return text

    # Try full sentence transliteration (ITRANS -> Devanagari)
    if HAS_INDIC:
        try:
            dev = sanscript.transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
            # Only use it if it actually changed something meaningful
            if dev != text and any("\u0900" <= ch <= "\u097f" for ch in dev):
                return dev
        except Exception:
            pass

    # Fallback: replace known Hinglish keywords with Devanagari equivalents
    result = text
    low = result.lower()
    for hing, dev in sorted(HINGLISH_TO_DEVANAGARI.items(), key=lambda x: -len(x[0])):
        result = re.sub(re.escape(hing), dev, result, flags=re.IGNORECASE)
    return result


def preprocess(text: str) -> Tuple[str, List[str]]:
    """Full preprocessing pipeline. Returns (processed_text, urls)."""
    cleaned = clean_text(text)
    urls = extract_urls(cleaned)
    processed = transliterate_hinglish(cleaned)
    return processed, urls
