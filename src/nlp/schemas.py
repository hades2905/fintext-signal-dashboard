"""
Data schemas for the news sentiment pipeline.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

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
    url: str | None = None
    published_at: datetime | None = None
    source: str | None = None
    sentiment: SentimentScore | None = None
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
    filed_at: datetime | None = None
    accession_number: str
    text: str
    url: str | None = None
    sentiment: SentimentScore | None = None
    entities: list[Entity] = Field(default_factory=list)
    extracted: StructuredExtract | None = None


class StructuredExtract(BaseModel):
    """
    LLM-extracted structured data from a financial filing or fund letter.
    Fields are Optional – the LLM will only populate what it can find.
    """
    fund_or_entity_name: str | None = None
    strategy: str | None = None          # Buyout, Credit, Real Estate, Infra, ...
    geography: str | None = None
    aum_bn_usd: float | None = None      # AUM in billion USD
    net_irr_pct: float | None = None     # Net IRR in %
    gross_irr_pct: float | None = None
    tvpi: float | None = None            # Total Value to Paid-In
    dpi: float | None = None             # Distributions to Paid-In
    rvpi: float | None = None            # Residual Value to Paid-In
    vintage_year: int | None = None
    deployment_pace: str | None = None   # "accelerating" | "slowing" | "stable"
    exit_environment: str | None = None  # "favorable" | "challenging" | "mixed"
    fundraising_outlook: str | None = None
    key_risks: list[str] = Field(default_factory=list)
    key_opportunities: list[str] = Field(default_factory=list)
    overall_sentiment: str | None = None  # "positive" | "cautious" | "negative"
    investment_summary: str | None = None  # 2-3 sentence LLM narrative
    raw_llm_response: str | None = None


EdgarFiling.model_rebuild()
