"""
FinBERT-based sentiment analysis for financial text.

Model: ProsusAI/finbert (HuggingFace)
- Trained specifically on financial text (earnings calls, reports, news)
- Labels: positive / negative / neutral
- Runs fully locally, no API key required
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Union

from transformers import pipeline, Pipeline

from .schemas import Article, SentimentLabel, SentimentScore

logger = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"
MAX_TOKENS = 512  # FinBERT context window


@lru_cache(maxsize=1)
def _load_pipeline() -> Pipeline:
    """Load FinBERT pipeline once and cache it."""
    logger.info("Loading FinBERT model '%s' (first call may download ~450MB)…", MODEL_NAME)
    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        top_k=None,          # return all 3 class scores
        truncation=True,
        max_length=MAX_TOKENS,
    )


def _truncate(text: str, max_chars: int = 2000) -> str:
    """Truncate text to avoid feeding extremely long strings to the tokenizer."""
    return text[:max_chars]


def _scores_to_model(raw: list[dict]) -> SentimentScore:
    """Convert raw pipeline output [{label, score}, …] → SentimentScore."""
    label_map = {item["label"].lower(): item["score"] for item in raw}
    dominant = max(label_map, key=lambda k: label_map[k])
    return SentimentScore(
        label=SentimentLabel(dominant),
        positive=label_map.get("positive", 0.0),
        negative=label_map.get("negative", 0.0),
        neutral=label_map.get("neutral", 0.0),
    )


class SentimentAnalyser:
    """
    Wrapper around HuggingFace FinBERT pipeline.

    Usage
    -----
    >>> analyser = SentimentAnalyser()
    >>> score = analyser.score("Earnings beat expectations by a wide margin.")
    >>> score.label
    <SentimentLabel.POSITIVE: 'positive'>
    """

    def __init__(self) -> None:
        self._pipe = _load_pipeline()

    def score(self, text: str) -> SentimentScore:
        """Return a :class:`SentimentScore` for *text*."""
        raw = self._pipe(_truncate(text))
        # pipeline with top_k=None returns list[list[dict]] for one input
        if isinstance(raw[0], list):
            raw = raw[0]
        return _scores_to_model(raw)

    def score_articles(self, articles: list[Article]) -> list[Article]:
        """
        Annotate each article in-place with its sentiment score.
        Returns the same list for convenience.
        """
        for article in articles:
            article.sentiment = self.score(article.title + " " + article.text)
        return articles
