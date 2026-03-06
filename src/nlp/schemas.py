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
