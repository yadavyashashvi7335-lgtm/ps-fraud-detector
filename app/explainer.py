"""LLM explanation generator with rule-based Hindi/English fallback templates."""

import os
from typing import List
from app.schemas import RedFlag, AnalyseResponse
from app.config import (
    DEFAULT_LANG, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
)

try:
    import requests
    HAS_REQUESTS = True
except Exception:  # pragma: no cover
    HAS_REQUESTS = False


# ---------------------------------------------------------------------------
# Fallback explanation templates (used when no LLM API key is configured)
# ---------------------------------------------------------------------------
TEMPLATES = {
    "en": {
        "safe": {
            "explanation": "This message looks safe. No strong scam indicators were found.",
            "next_step": "You can ignore it. If you're still unsure, forward it to cybercrime.gov.in.",
        },
        "suspicious": {
            "explanation": "This message has some warning signs but is not confirmed as a scam.",
            "next_step": "Do not click any link or share personal details. Verify through the official app or website. Report at cybercrime.gov.in if you remain doubtful.",
        },
        "scam": {
            "explanation": "This message shows strong scam indicators such as {flags}.",
            "next_step": "Do not click the link, do not share OTP/PIN/password, and do not call the number. Report it immediately at cybercrime.gov.in or call the cybercrime helpline 1930.",
        },
    },
    "hi": {
        "safe": {
            "explanation": "यह संदेश सुरक्षित लग रहा है। इसमें कोई मजबूत घोटाले के संकेत नहीं मिले।",
            "next_step": "आप इसे नज़रअंदाज कर सकते हैं। अगर अभी भी संदेह हो, तो इसे cybercrime.gov.in पर रिपोर्ट करें।",
        },
        "suspicious": {
            "explanation": "इस संदेश में कुछ चेतावनी संकेत हैं, लेकिन यह पुष्ट घोटाला नहीं है।",
            "next_step": "किसी लिंक पर क्लिक न करें और न ही अपनी व्यक्तिगत जानकारी साझा करें। आधिकारिक ऐप या वेबसाइट से पुष्टि करें। संदेह होने पर cybercrime.gov.in पर रिपोर्ट करें।",
        },
        "scam": {
            "explanation": "इस संदेश में मजबूत घोटाले के संकेत दिखते हैं जैसे कि {flags}।",
            "next_step": "लिंक पर क्लिक न करें, OTP/पिन/पासवर्ड साझा न करें और नंबर पर कॉल न करें। इसे तुरंत cybercrime.gov.in पर रिपोर्ट करें या साइबर अपराध हेल्पलाइन 1930 पर कॉल करें।",
        },
    },
}


def _flag_text(flags: List[RedFlag], lang: str) -> str:
    if not flags:
        return "no clear warning signs" if lang == "en" else "कोई स्पष्ट चेतावनी संकेत नहीं"
    key = "label_en" if lang == "en" else "label_hi"
    return ", ".join(getattr(f, key) for f in flags)


def _fallback(label: str, flags: List[RedFlag], lang: str) -> tuple[str, str]:
    tpl = TEMPLATES.get(lang, TEMPLATES["en"]).get(label.lower(), TEMPLATES["en"][label.lower()])
    flags_txt = _flag_text(flags, lang)
    explanation = tpl["explanation"].replace("{flags}", flags_txt)
    return explanation, tpl["next_step"]


def generate_explanation(response: AnalyseResponse, lang: str) -> tuple[str, str]:
    """Produce (explanation, next_step) in the requested language.

    Calls an OpenAI-compatible LLM API when LLM_API_KEY is set; otherwise uses
    rule-based fallback templates enriched with detected red flags.
    """
    lang = lang if lang in ("en", "hi") else DEFAULT_LANG
    if lang not in ("en", "hi"):
        lang = "en"

    # Try LLM API first
    if LLM_API_KEY and HAS_REQUESTS:
        try:
            return _llm_explain(response, flags=response.flags, lang=lang)
        except Exception:
            pass  # fall back to templates below

    return _fallback(response.label, response.flags, lang)


def _llm_explain(response: AnalyseResponse, flags: List[RedFlag], lang: str) -> tuple[str, str]:
    """Call an OpenAI-compatible chat completion endpoint."""
    flag_list = ", ".join(
        (f.label_en if lang == "en" else f.label_hi) for f in flags
    ) or ("none" if lang == "en" else "कोई नहीं")

    system_prompt = (
        "You are a helpful cyber-fraud assistant for Indian users. "
        "Explain in plain language why a message is Safe / Suspicious / Scam, "
        "listing the red flags found, and give one clear next step. "
        "Reply in Hindi if requested, else English. Be concise."
    )
    user_prompt = (
        f"Message (cleaned): {response.processed_text}\n"
        f"Extracted URLs: {', '.join(response.urls) or 'none'}\n"
        f"ML scam probability: {response.ml_probability:.2f}\n"
        f"Rule-based red flags: {flag_list}\n"
        f"Final label: {response.label} (score {response.score:.2f})\n"
        f"Language: {'Hindi' if lang == 'hi' else 'English'}\n\n"
        "Return ONLY: Explanation: <text>\nNext step: <text>"
    )

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 300,
    }

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    resp = requests.post(
        f"{LLM_BASE_URL.rstrip('/')}/chat/completions",
        json=payload,
        headers=headers,
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"].strip()

    # Parse "Explanation: ...\nNext step: ..."
    explanation = content
    next_step = ""
    if "Next step:" in content:
        parts = content.split("Next step:", 1)
        explanation = parts[0].replace("Explanation:", "").strip()
        next_step = parts[1].strip()
    return explanation, next_step
