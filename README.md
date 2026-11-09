# PS Fraud Detector — Backend

Hinglish-aware, **explainable**, and actionable scam detection for Indian SMS/WhatsApp fraud
(fake KYC, UPI refund, lottery, and job-offer scams). Built as a 12-hour hack/demo backend.

## Problem
Existing spam filters (Truecaller, Google Messages) mostly handle **English** and return a bare
"spam" label. Users can't tell **why** a message is dangerous or **what to do**, especially
when the message is in Hindi or Hinglish.

## Solution
Paste a suspicious message → get:

| Output | What it means |
|--------|---------------|
| **Risk score** | Safe / Suspicious / Scam (0–1) |
| **Red flags** | OTP/PIN request, urgency pressure, lookalike domain, unknown payment link, UPI ID, fake KYC, lottery/job bait |
| **Explanation + next step** | Plain-language Hindi or English, e.g. "Don't click; report at cybercrime.gov.in / हेल्पलाइन 1930" |

## Workflow (flowchart)

```
Input message
    ↓
Preprocessing (clean text, transliterate Hinglish, extract URLs)
    ↓
ML classifier (scam probability)   [multilingual transformer OR TF-IDF + LR baseline]
    ↓
Rule layer (OTP/PIN, urgency words, URL/UPI checks) → red flags
    ↓
LLM explanation generator (OpenAI-compatible; rule-based fallback if no API key)
    ↓
Result card: score, flags, advice in Hindi or English
```

## Tech stack
- **Backend:** Python + FastAPI
- **Baseline model:** TF-IDF + Logistic Regression (scikit-learn)
- **Advanced model:** fine-tunable multilingual transformer (`google/muril-base-cased` or `distilbert-base-multilingual-cased`)
- **Explanations:** LLM API (OpenAI / Groq / Ollama) with Hindi + Hindi fallback templates
- **Data:** UCI SMS Spam Collection + small labelled Hinglish set (`data/sample_hinglish.csv`)

## Quick start

```bash
# 1. Enter the project folder
cd ps-fraud-detector-backend

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure (optional LLM key for AI-generated explanations)
copy .env.example .env
# edit .env and set LLM_API_KEY / LLM_BASE_URL / LLM_MODEL if desired

# 5. Train / warm up the baseline model
python scripts/train_baseline.py

# 6. Run the server
uvicorn app.main:app --reload --port 8000
```

### Open the app

| URL | What it is |
|-----|------------|
| `http://localhost:8000/ui` | **Frontend** — paste a message, get score + flags + Hindi/English advice |
| `http://localhost:8000/docs` | Interactive API docs (Swagger) |
| `http://localhost:8000/health` | Backend health check |

The frontend is served from the same port, so no separate dev server or CORS setup is needed.

### Example request

```bash
curl -X POST "http://localhost:8000/analyse" \
  -H "Content-Type: application/json" \
  -d '{"text": "आपका खाता ब्लॉक हो जाएगा, तुरंत OTP भेजें", "lang": "hi"}'
```

### Example response

```json
{
  "label": "Scam",
  "score": 0.82,
  "ml_probability": 0.68,
  "rule_score": 0.35,
  "flags": [
    {"flag": "otp_pin_request", "label_en": "OTP / PIN / password requested", "label_hi": "OTP, पिन या पासवर्ड मांगा गया", "weight": 0.35, "evidence": "otp"},
    {"flag": "urgency_pressure", "label_en": "Urgency / pressure tactics", "label_hi": "जल्दबाजी या दबाव डाला गया", "weight": 0.25, "evidence": "तुरंत"}
  ],
  "explanation": "इस संदेश में मजबूत घोटाले के संकेत दिखते हैं जैसे कि OTP, पिन या पासवर्ड मांगा गया, जल्दबाजी या दबाव डाला गया।",
  "next_step": "लिंक पर क्लिक न करें, OTP/पिन/पासवर्ड साझा न करें... cybercrime.gov.in पर रिपोर्ट करें या हेल्पलाइन 1930 पर कॉल करें।",
  "language": "hi",
  "processed_text": "आपका खाता ब्लॉक हो जाएगा, तुरंत ओटीपी भेजें",
  "urls": []
}
```

## Run the smoke tests

```bash
python test_backend.py
```

## Fine-tune the transformer (hours 2–5)

```bash
pip install torch transformers
python scripts/train_transformer.py --model google/muril-base-cased --epochs 3
```
The server auto-detects a fine-tuned checkpoint under `app/models/artefacts/transformer/`.

## 12-hour plan (from your problem statement)

| Hours | Milestone |
|-------|-----------|
| 0–2 | Repo setup, dataset cleaning, baseline model (`scripts/train_baseline.py`) |
| 2–5 | Fine-tune transformer, build rule layer (`scripts/train_transformer.py`, `app/rules.py`) |
| 5–8 | API + frontend (`app/main.py`, FastAPI `/docs`) |
| 8–10 | LLM explanations, Hindi output, test on real scam examples (`app/explainer.py`) |
| 10–12 | Polish, demo script, README, final commits |

## Data & honesty note
The bundled `data/sample_hinglish.csv` is a **small starter set** (English + Hinglish). For a real demo, augment it with the [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) and your own labelled Hinglish examples. Be transparent about dataset size in the pitch.

### Real figures to cite (look up before the slides)
- **I4C (Indian Cyber Crime Coordination Centre)** / NCRB / RBI cyber-fraud statistics (UPI fraud, KYC scam complaints, etc.). Cite the latest report numbers on your slide rather than guessing.

## Existing work vs your contribution
Tools like Truecaller and Google Messages filter spam but mostly **label** without explaining. This backend's unique contribution: **Hinglish-aware, explainable, and actionable** — red flags + plain-language next steps in Hindi or English, not just a spam label.

## Project layout
```
app/
  main.py          # FastAPI app + /analyse, /health
  config.py        # settings (.env)
  schemas.py       # Pydantic models
  preprocessing.py # clean, transliterate Hinglish, extract URLs
  rules.py         # rule-based red-flag layer (Hindi + Hinglish)
  models/
    baseline.py    # TF-IDF + Logistic Regression
    transformer.py # multilingual transformer wrapper
  explainer.py     # LLM explanation + Hindi/English fallback templates
data/
  sample_hinglish.csv   # small labelled starter set
frontend/
  index.html            # single-file frontend (served at /ui)
scripts/
  train_baseline.py
  train_transformer.py
test_backend.py      # smoke tests
requirements.txt
.env.example
README.md
```

## License
MIT — build, demo, extend.