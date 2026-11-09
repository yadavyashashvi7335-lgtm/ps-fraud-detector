"""FastAPI application: analyse endpoint, health check, model lifecycle."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import APP_NAME, DEBUG, BASE_DIR, BASELINE_MODEL_DIR
from app.schemas import AnalyseRequest, AnalyseResponse, HealthResponse, RedFlag
from app.preprocessing import preprocess
from app.models.baseline import load_or_train, predict_proba
from app.models.transformer import TransformerScamClassifier
from app.rules import detect_red_flags, compute_rule_score
from app.explainer import generate_explanation

app = FastAPI(title=APP_NAME, debug=DEBUG)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend (single HTML file) alongside the API
FRONTEND_DIR = BASE_DIR / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/ui", include_in_schema=False)
def ui():
    return FileResponse(FRONTEND_DIR / "index.html")

# ---- Global model state (loaded at startup) ----
_state = {
    "baseline": None,
    "transformer": None,
    "baseline_available": False,
    "transformer_available": False,
}


@app.on_event("startup")
def load_models() -> None:
    """Load (or quick-train) the baseline model and initialise the transformer wrapper."""
    _state["baseline"] = load_or_train()
    _state["baseline_available"] = True

    _state["transformer"] = TransformerScamClassifier()
    _state["transformer_available"] = _state["transformer"].available


def _label_from_score(score: float) -> str:
    from app.config import SCORE_LABELS
    for low, high, label in SCORE_LABELS:
        if low <= score < high:
            return label
    return "Scam"


@app.get("/", tags=["meta"])
def root():
    return {
        "service": APP_NAME,
        "docs": "/docs",
        "health": "/health",
        "analyse": "POST /analyse",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    from app.config import LLM_API_KEY
    return HealthResponse(
        status="ok",
        model_loaded=_state["baseline"] is not None,
        baseline_available=_state["baseline_available"],
        transformer_available=_state["transformer_available"],
        llm_configured=bool(LLM_API_KEY),
    )


# Helper text for action plan
def _do_not_do(lang: str) -> str:
    if lang == "hi":
        return "लिंक पर क्लिक न करें\nOTP/पिन/पासवर्ड शेयर न करें\nअज्ञात नंबर पर कॉल न करें\nतुरंत कोई निर्णय न लें\nबैंक से सीधे संपर्क करें"
    return "Do not click any link\nDo not share OTP, PIN or password\nDo not call unknown numbers\nDo not take urgent decisions\nContact your bank directly"


def _do_this(lang: str) -> str:
    if lang == "hi":
        return "संदिग्ध संदेश को डिलीट करें\nसाइबर अपराध पोर्टल पर रिपोर्ट करें\nबैंक को सूचित करें\nस्क्रीनशॉट सुरक्षित रखें"
    return "Delete the suspicious message\nReport at cybercrime.gov.in\nInform your bank\nKeep screenshots as evidence"


def _complaint_info(lang: str) -> str:
    if lang == "hi":
        return "साइबर अपराध रिपोर्टिंग पोर्टल: cybercrime.gov.in\nसाइबर हेल्पलाइन: 1930 (24x7)\nराष्ट्रीय साइबर अपराध रिपोर्टिंग पोर्टल पर शिकायत दर्ज कराएं।"
    return "Cyber Crime Reporting Portal: cybercrime.gov.in\nCyber Helpline: 1930 (24x7)\nFile a complaint at the National Cyber Crime Reporting Portal."


@app.post("/analyse", response_model=AnalyseResponse, tags=["analyse"])
def analyse(req: AnalyseRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    # 1. Preprocess: clean + transliterate Hinglish + extract URLs
    processed_text, urls = preprocess(req.text)

    # 2. ML classifier: scam probability
    ml_probability = 0.5
    transformer = _state["transformer"]
    if transformer and transformer.available:
        tp = transformer.predict_proba(req.text)
        if tp is not None:
            ml_probability = tp
    if ml_probability == 0.5:
        ml_probability = predict_proba(_state["baseline"], req.text)

    # 3. Rule layer: red flags
    flags = detect_red_flags(req.text, urls)

    # Safe indicators check for legitimate bank messages
    low = req.text.lower()
    safe_indicators = [
        "credited", "debited", "balance", "statement", "invoice", "receipt",
        "payment successful", "transaction", "account balance", "available balance",
        "जमा", "राशि", "शेष", "बैलेंस", "स्टेटमेंट", "भुगतान सफल",
        "लेन-देन", "आपका खाता", "धन्यवाद", "सादर", "जमा हुआ", "क्रेडिट", "डेबिट",
    ]
    has_safe = any(p in low for p in safe_indicators)

    rule_score = compute_rule_score(flags, has_safe=has_safe)

    # 4. Combined score + label
    from app.config import ML_WEIGHT, RULE_WEIGHT
    score = ML_WEIGHT * ml_probability + RULE_WEIGHT * rule_score
    score = min(max(score, 0.0), 1.0)
    label = _label_from_score(score)

    # 5. Language selection
    lang = req.lang if req.lang in ("en", "hi") else None
    lang = lang or "en"

    # 6. Build response
    response = AnalyseResponse(
        label=label,
        score=round(score, 4),
        ml_probability=round(ml_probability, 4),
        rule_score=round(rule_score, 4),
        flags=flags,
        explanation="",
        next_step="",
        language=lang,
        processed_text=processed_text,
        urls=urls,
        risk_percentage=round(score * 100, 1),
        do_not_do=_do_not_do(lang),
        do_this=_do_this(lang),
        complaint_info=_complaint_info(lang),
    )

    # 7. LLM / template explanation
    explanation, next_step = generate_explanation(response, lang)
    response.explanation = explanation
    response.next_step = next_step

    return response
