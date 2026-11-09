"""Rule-based red-flag layer with Hindi/Hinglish awareness."""

import re
from typing import List, Tuple
from app.schemas import RedFlag

# ---------------------------------------------------------------------------
# Red-flag definitions: (key, label_en, label_hi, weight)
# ---------------------------------------------------------------------------
FLAG_DEFINITIONS = [
    (
        "otp_pin_request",
        "OTP / PIN / password requested",
        "OTP, पिन या पासवर्ड मांगा गया",
        0.40,
    ),
    (
        "urgency_pressure",
        "Urgency / pressure tactics",
        "जल्दबाजी या दबाव डाला गया",
        0.25,
    ),
    (
        "unknown_payment_link",
        "Unknown / suspicious payment link",
        "अज्ञात या संदिग्ध भुगतान लिंक",
        0.30,
    ),
    (
        "lookalike_domain",
        "Lookalike / spoofed domain",
        "नकली दिखने वाला डोमेन",
        0.25,
    ),
    (
        "upi_id",
        "UPI ID mentioned",
        "यूपीआई आईडी मिली",
        0.20,
    ),
    (
        "fake_kyc",
        "Fake KYC / account-block threat",
        "नकली केवाईसी या खाता बंद होने की धमकी",
        0.25,
    ),
    (
        "lottery_job_offer",
        "Lottery / job-offer scam bait",
        "लॉटरी या नौकरी का लालच",
        0.20,
    ),
]

FLAG_MAP = {key: (label_en, label_hi, weight) for key, label_en, label_hi, weight in FLAG_DEFINITIONS}

# ---------------------------------------------------------------------------
# Safe indicators (reduce false positives for legitimate bank messages)
# ---------------------------------------------------------------------------
SAFE_INDICATORS = [
    # English
    "credited", "debited", "balance", "statement", "invoice", "receipt",
    "payment successful", "transaction", "account balance", "available balance",
    "your account", "dear customer", "welcome", "thank you", "regards",
    # Hindi / Hinglish
    "जमा", "राशि", "शेष", "बैलेंस", "स्टेटमेंट", "चालान", "रसीद",
    "भुगतान सफल", "लेन-देन", "आपका खाता", "धन्यवाद", "सादर",
    "जमा हुआ", "क्रेडिट", "डेबिट",
]

URGENCY_WORDS = [
    "urgent", "immediately", "within", "hours", "expire", "expires", "suspend",
    "suspended", "blocked", "block", "last chance", "act now", "hurry",
    "जल्दी", "तुरंत", "अभी", "जल्द", "बंद", "ब्लॉक", "समाप्त", "खतरे",
    "turant", "jaldi", "abhi", "band", "block", "khatre",
]

OTP_PIN_WORDS = [
    "otp", "पिन", "pin", "पासवर्ड", "password", "pwd", "cvp", "mpin",
    "वन टाइम", "वन टाइम पासवर्ड",
]

KYC_BLOCK_WORDS = [
    "kyc", "केवाईसी", "know your customer", "account", "खाता", "बैंक",
    "suspend", "निलंबित", "block", "ब्लॉक", "बंद", "freeze", "फ्रीज",
]

LOTTERY_JOB_WORDS = [
    "lottery", "लॉटरी", "jackpot", "winner", "विजेता", "prize",
    "इनाम", "reward", "पुरस्कार", "job", "नौकरी", "offer", "ऑफर",
    "work from home", "घर से काम", "part time", "भाग समय",
]

PAYMENT_LINK_HINTS = [
    "pay", "payment", "भुगतान", "पेमेंट", "upi", "यूपीआई", "paytm", "phonepe",
    "google pay", "gpay", "bhim", "रुपया", "रुपये",
]

LOOKALIKE_HINTS = [
    "paytm", "phonepe", "googlepay", "gpay", "amazon", "flipkart", "bank",
    "sbi", "hdfc", "icici", "axis", "kotak", "razorpay", "cashfree",
]


def _contains_any(text: str, patterns: List[str]) -> bool:
    low = text.lower()
    return any(p in low for p in patterns)


def detect_red_flags(text: str, urls: List[str]) -> List[RedFlag]:
    """Run the rule layer and return triggered red flags with evidence."""
    flags: List[RedFlag] = []
    low = text.lower()

    # --- Check safe indicators first ---
    safe_hits = [p for p in SAFE_INDICATORS if p in low]
    has_safe = len(safe_hits) > 0

    # --- 1. OTP / PIN / password request ---
    otp_hits = [p for p in OTP_PIN_WORDS if p in low]
    otp_pattern = re.search(r"(send|share|enter|confirm|verify).{0,20}(otp|pin|password)", low)
    if otp_hits or otp_pattern:
        evidence = ", ".join(otp_hits) if otp_hits else "OTP/PIN/password request pattern matched"
        label_en, label_hi, weight = FLAG_MAP["otp_pin_request"]
        flags.append(RedFlag(
            flag="otp_pin_request", label_en=label_en, label_hi=label_hi,
            weight=weight, evidence=evidence,
        ))

    # --- 2. Urgency / pressure ---
    if _contains_any(low, URGENCY_WORDS):
        hits = [p for p in URGENCY_WORDS if p in low][:4]
        label_en, label_hi, weight = FLAG_MAP["urgency_pressure"]
        flags.append(RedFlag(
            flag="urgency_pressure", label_en=label_en, label_hi=label_hi,
            weight=weight, evidence=", ".join(hits),
        ))

    # --- 3. Unknown / suspicious payment link ---
    payment_urls = [u for u in urls if _contains_any(u.lower(), PAYMENT_LINK_HINTS)]
    if payment_urls:
        label_en, label_hi, weight = FLAG_MAP["unknown_payment_link"]
        flags.append(RedFlag(
            flag="unknown_payment_link", label_en=label_en, label_hi=label_hi,
            weight=weight, evidence=", ".join(payment_urls),
        ))

    # --- 4. Lookalike / spoofed domain ---
    brand_in_url = [b for b in LOOKALIKE_HINTS if any(b in u.lower() for u in urls)]
    suspicious_domains = [u for u in urls if re.search(r"(secure|verify|update|login|signin|pay|refund)", u.lower())]
    if brand_in_url or suspicious_domains:
        evidence = ", ".join(brand_in_url or suspicious_domains)
        label_en, label_hi, weight = FLAG_MAP["lookalike_domain"]
        flags.append(RedFlag(
            flag="lookalike_domain", label_en=label_en, label_hi=label_hi,
            weight=weight, evidence=evidence,
        ))

    # --- 5. UPI ID ---
    upi_hits = re.findall(r"[\w.\-]{2,}@[\w]+", text)
    if upi_hits:
        label_en, label_hi, weight = FLAG_MAP["upi_id"]
        flags.append(RedFlag(
            flag="upi_id", label_en=label_en, label_hi=label_hi,
            weight=weight, evidence=", ".join(upi_hits[:5]),
        ))

    # --- 6. Fake KYC / account-block threat ---
    if _contains_any(low, KYC_BLOCK_WORDS):
        hits = [p for p in KYC_BLOCK_WORDS if p in low][:4]
        label_en, label_hi, weight = FLAG_MAP["fake_kyc"]
        flags.append(RedFlag(
            flag="fake_kyc", label_en=label_en, label_hi=label_hi,
            weight=weight, evidence=", ".join(hits),
        ))

    # --- 7. Lottery / job-offer bait ---
    if _contains_any(low, LOTTERY_JOB_WORDS):
        hits = [p for p in LOTTERY_JOB_WORDS if p in low][:4]
        label_en, label_hi, weight = FLAG_MAP["lottery_job_offer"]
        flags.append(RedFlag(
            flag="lottery_job_offer", label_en=label_en, label_hi=label_hi,
            weight=weight, evidence=", ".join(hits),
        ))

    return flags


def compute_rule_score(flags: List[RedFlag], has_safe: bool = False) -> float:
    """Rule score = min(1.0, sum of triggered flag weights).
    Reduce score if safe indicators present (legitimate bank message)."""
    total = sum(f.weight for f in flags)
    if has_safe and total > 0:
        # Legitimate bank message: reduce rule score significantly
        total = total * 0.35
    return min(1.0, total)