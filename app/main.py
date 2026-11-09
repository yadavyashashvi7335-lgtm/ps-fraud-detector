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


@app.post("/analyse", response_model=AnalyseResponse, tags=["analyse"])
def analyse(req: AnalyseRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    # 1. Preprocess: clean + transliterate Hinglish + extract URLs
    processed_text, urls = preprocess(req.text)

    # 2. ML classifier: scam probability (transformer preferred, baseline fallback)
    ml_probability = 0.5
    transformer = _state["transformer"]
    if transformer and transformer.available:
        tp = transformer.predict_proba(req.text)
        if tp is not None:
            ml_probability = tp
    if ml_probability == 0.5:
        ml_probability = predict_proba(_state["baseline"], req.text)

    # 3. Rule layer: red flags
    flags: list[RedFlag] = detect_red_flags(req.text, urls)
    rule_score = compute_rule_score(flags)

    # 4. Combined score + label
    from app.config import ML_WEIGHT, RULE_WEIGHT
    score = ML_WEIGHT * ml_probability + RULE_WEIGHT * rule_score
    score = min(max(score, 0.0), 1.0)
    label = _label_from_score(score)

    # 5. Language selection
    lang = req.lang if req.lang in ("en", "hi") else None

    # 6. Build response with placeholder explanation (filled by explainer)
    response = AnalyseResponse(
        label=label,
        score=round(score, 4),
        ml_probability=round(ml_probability, 4),
        rule_score=round(rule_score, 4),
        flags=flags,
        explanation="",  # filled below
        next_step="",
        language=lang or "en",
        processed_text=processed_text,
        urls=urls,
    )

    # 7. LLM / template explanation
    explanation, next_step = generate_explanation(response, lang or "en")
    response.explanation = explanation
    response.next_step = next_step
    response.language = lang or "en"

    return response

