"""
FinBERT-based sentiment analysis for financial text.

Model: ProsusAI/finbert (HuggingFace Inference API)
- Trained specifically on financial text (earnings calls, reports, news)
- Labels: positive / negative / neutral
- Uses HuggingFace InferenceClient – no local model download required
- Requires a free HF_TOKEN (https://huggingface.co/settings/tokens)
"""
from __future__ import annotations

import logging
import os

from huggingface_hub import InferenceClient

from .schemas import Article, SentimentLabel, SentimentScore

logger = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"
MAX_CHARS = 1500  # keep well within FinBERT's 512-token window


def _truncate(text: str, max_chars: int = MAX_CHARS) -> str:
    """Truncate text to avoid exceeding the model's token limit."""
    return text[:max_chars]


def _scores_to_model(raw: list) -> SentimentScore:
    """
    Convert InferenceClient text_classification output to SentimentScore.

    The API returns a list of ClassificationOutput objects with .label / .score,
    or plain dicts – handle both.
    """
    label_map: dict[str, float] = {}
    for item in raw:
        if hasattr(item, "label"):
            label_map[item.label.lower()] = item.score
        else:
            label_map[item["label"].lower()] = item["score"]

    dominant = max(label_map, key=lambda k: label_map[k])
    return SentimentScore(
        label=SentimentLabel(dominant),
        positive=label_map.get("positive", 0.0),
        negative=label_map.get("negative", 0.0),
        neutral=label_map.get("neutral", 0.0),
    )


class SentimentAnalyser:
    """
    Wrapper around the HuggingFace Inference API for FinBERT.

    Parameters
    ----------
    api_key:
        HuggingFace token. Falls back to the ``HF_TOKEN`` environment variable.

    Usage
    -----
    >>> analyser = SentimentAnalyser(api_key="hf-...")
    >>> score = analyser.score("Earnings beat expectations by a wide margin.")
    >>> score.label
    <SentimentLabel.POSITIVE: 'positive'>
    """

    def __init__(self, api_key: str | None = None) -> None:
        token = api_key or os.environ.get("HF_TOKEN", "")
        if not token:
            raise ValueError(
                "A HuggingFace token is required. "
                "Set HF_TOKEN env var or pass api_key=... to SentimentAnalyser()."
            )
        self._client = InferenceClient(
            provider="hf-inference",
            api_key=token,
        )

    def score(self, text: str) -> SentimentScore:
        """Return a :class:`SentimentScore` for *text*."""
        raw = self._client.text_classification(
            _truncate(text),
            model=MODEL_NAME,
        )
        return _scores_to_model(raw)

    def score_articles(self, articles: list[Article]) -> list[Article]:
        """
        Annotate each article in-place with its sentiment score.
        Returns the same list for convenience.
        """
        for article in articles:
            article.sentiment = self.score(article.title + " " + article.text)
        return articles
