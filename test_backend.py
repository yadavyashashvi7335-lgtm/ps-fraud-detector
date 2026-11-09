"""Quick smoke tests for the backend (no LLM key required)."""

from app.main import app, _state, load_models
from app.schemas import AnalyseResponse
from fastapi.testclient import TestClient

load_models()
client = TestClient(app)


def _pr(text: str) -> str:
    """Make text safe to print on Windows cp1252 console."""
    return str(text).encode("ascii", errors="replace").decode("ascii")


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["baseline_available"] is True
    print("health OK:", body)


def test_safe_message():
    r = client.post("/analyse", json={"text": "नमस्ते, आप कैसे हैं?", "lang": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in ("Safe", "Suspicious")
    assert 0 <= body["score"] <= 1
    assert body["explanation"]
    assert body["next_step"]
    print("safe:", body["label"], round(body["score"], 3), "|", _pr(body["explanation"][:60]))


def test_scam_message():
    r = client.post("/analyse", json={
        "text": "आपका खाता ब्लॉक हो जाएगा, तुरंत OTP भेजें",
        "lang": "hi",
    })
    assert r.status_code == 200
    body = r.json()
    # Baseline is small; check that scam indicators are caught via flags
    assert len(body["flags"]) >= 1
    assert any(f["flag"] in ("otp_pin_request", "urgency_pressure", "fake_kyc") for f in body["flags"])
    print("scam:", body["label"], round(body["score"], 3), "flags:", [f["flag"] for f in body["flags"]])
    print("   explanation:", _pr(body["explanation"][:80]))
    print("   next_step:", _pr(body["next_step"][:80]))


def test_url_extraction():
    r = client.post("/analyse", json={"text": "Claim refund at https://paytm-secure.verify.xyz/pay", "lang": "en"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["urls"]) >= 1
    print("urls:", body["urls"], "label:", body["label"])


if __name__ == "__main__":
    test_health()
    test_safe_message()
    test_scam_message()
    test_url_extraction()
    print("\nAll smoke tests passed.")
