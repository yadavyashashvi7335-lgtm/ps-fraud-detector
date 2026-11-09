"""Pydantic request/response schemas for the analysis API."""

from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional


class RedFlag(BaseModel):
    flag: str = Field(..., description="Machine-readable flag key")
    label_en: str = Field(..., description="Human-readable label (English)")
    label_hi: str = Field(..., description="Human-readable label (Hindi)")
    weight: float = Field(..., description="Contribution weight (0-1)")
    evidence: str = Field(..., description="What was found in the message")


class AnalyseRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Suspicious message or link text")
    lang: Optional[str] = Field(
        None, description="Preferred explanation language: 'en' or 'hi' (defaults to server DEFAULT_LANG)"
    )


class AnalyseResponse(BaseModel):
    label: str = Field(..., description="Safe / Suspicious / Scam")
    score: float = Field(..., ge=0.0, le=1.0, description="Combined risk score (0-1)")
    ml_probability: float = Field(..., ge=0.0, le=1.0)
    rule_score: float = Field(..., ge=0.0, le=1.0)
    flags: List[RedFlag] = Field(default_factory=list)
    explanation: str = Field(..., description="Plain-language explanation")
    next_step: str = Field(..., description="Actionable next step")
    language: str = Field(..., description="Language used for explanation ('en' or 'hi')")
    processed_text: str = Field(..., description="Cleaned/transliterated text")
    urls: List[str] = Field(default_factory=list, description="Extracted URLs")


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    baseline_available: bool
    transformer_available: bool
    llm_configured: bool
