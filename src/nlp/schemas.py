"""
Data schemas for the news sentiment pipeline.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class SentimentScore(BaseModel):
    label: SentimentLabel
    positive: float = Field(ge=0.0, le=1.0)
    negative: float = Field(ge=0.0, le=1.0)
    neutral: float = Field(ge=0.0, le=1.0)


class Entity(BaseModel):
    text: str
    label: str  # ORG, PERSON, GPE, etc.


class Article(BaseModel):
    ticker: str
    title: str
    text: str
    url: Optional[str] = None
    published_at: Optional[datetime] = None
    source: Optional[str] = None
    sentiment: Optional[SentimentScore] = None
    entities: list[Entity] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# SEC EDGAR / Alternative Assets schemas
# ---------------------------------------------------------------------------

class FilingType(str, Enum):
    EIGHT_K = "8-K"
    TEN_K = "10-K"
    TEN_Q = "10-Q"
    OTHER = "OTHER"


class EdgarFiling(BaseModel):
    """A single SEC EDGAR filing document."""
    ticker: str
    company_name: str
    cik: str
    form_type: FilingType
    filed_at: Optional[datetime] = None
    accession_number: str
    text: str
    url: Optional[str] = None
    sentiment: Optional[SentimentScore] = None
    entities: list[Entity] = Field(default_factory=list)
    extracted: Optional["StructuredExtract"] = None


class StructuredExtract(BaseModel):
    """
    LLM-extracted structured data from a financial filing or fund letter.
    Fields are Optional – the LLM will only populate what it can find.
    """
    fund_or_entity_name: Optional[str] = None
    strategy: Optional[str] = None          # Buyout, Credit, Real Estate, Infra, ...
    geography: Optional[str] = None
    aum_bn_usd: Optional[float] = None      # AUM in billion USD
    net_irr_pct: Optional[float] = None     # Net IRR in %
    gross_irr_pct: Optional[float] = None
    tvpi: Optional[float] = None            # Total Value to Paid-In
    dpi: Optional[float] = None             # Distributions to Paid-In
    rvpi: Optional[float] = None            # Residual Value to Paid-In
    vintage_year: Optional[int] = None
    deployment_pace: Optional[str] = None   # "accelerating" | "slowing" | "stable"
    exit_environment: Optional[str] = None  # "favorable" | "challenging" | "mixed"
    fundraising_outlook: Optional[str] = None
    key_risks: list[str] = Field(default_factory=list)
    key_opportunities: list[str] = Field(default_factory=list)
    overall_sentiment: Optional[str] = None  # "positive" | "cautious" | "negative"
    investment_summary: Optional[str] = None  # 2-3 sentence LLM narrative
    raw_llm_response: Optional[str] = None


EdgarFiling.model_rebuild()
