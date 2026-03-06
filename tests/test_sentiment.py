"""
Tests for SentimentAnalyser – FinBERT pipeline is mocked.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.nlp.schemas import Article, SentimentLabel
from src.nlp.sentiment import SentimentAnalyser, _scores_to_model


# ---------------------------------------------------------------------------
# _scores_to_model helper
# ---------------------------------------------------------------------------
class TestScoresToModel:
    def test_positive_dominant(self):
        raw = [
            {"label": "positive", "score": 0.85},
            {"label": "negative", "score": 0.05},
            {"label": "neutral",  "score": 0.10},
        ]
        result = _scores_to_model(raw)
        assert result.label == SentimentLabel.POSITIVE
        assert result.positive == pytest.approx(0.85)

    def test_negative_dominant(self):
        raw = [
            {"label": "positive", "score": 0.05},
            {"label": "negative", "score": 0.88},
            {"label": "neutral",  "score": 0.07},
        ]
        result = _scores_to_model(raw)
        assert result.label == SentimentLabel.NEGATIVE

    def test_neutral_dominant(self):
        raw = [
            {"label": "positive", "score": 0.20},
            {"label": "negative", "score": 0.20},
            {"label": "neutral",  "score": 0.60},
        ]
        result = _scores_to_model(raw)
        assert result.label == SentimentLabel.NEUTRAL


# ---------------------------------------------------------------------------
# SentimentAnalyser with mocked pipeline
# ---------------------------------------------------------------------------
# Mock API response objects (simulate huggingface_hub ClassificationOutput)
class _MockLabel:
    def __init__(self, label: str, score: float):
        self.label = label
        self.score = score


MOCK_POSITIVE_RAW = [
    _MockLabel("positive", 0.92),
    _MockLabel("negative", 0.04),
    _MockLabel("neutral",  0.04),
]

MOCK_NEGATIVE_RAW = [
    _MockLabel("positive", 0.03),
    _MockLabel("negative", 0.91),
    _MockLabel("neutral",  0.06),
]


def _make_analyser(mock_output: list) -> SentimentAnalyser:
    analyser = SentimentAnalyser.__new__(SentimentAnalyser)
    mock_client = MagicMock()
    mock_client.text_classification.return_value = mock_output
    analyser._client = mock_client
    return analyser


class TestSentimentAnalyser:
    def test_score_positive(self):
        analyser = _make_analyser(MOCK_POSITIVE_RAW)
        result = analyser.score("Company reports record profits.")
        assert result.label == SentimentLabel.POSITIVE
        assert result.positive == pytest.approx(0.92)

    def test_score_negative(self):
        analyser = _make_analyser(MOCK_NEGATIVE_RAW)
        result = analyser.score("Company faces massive losses.")
        assert result.label == SentimentLabel.NEGATIVE

    def test_score_articles_annotates_all(self):
        analyser = _make_analyser(MOCK_POSITIVE_RAW)
        articles = [
            Article(ticker="AAPL", title="Title 1", text="Text 1"),
            Article(ticker="AAPL", title="Title 2", text="Text 2"),
            Article(ticker="AAPL", title="Title 3", text="Text 3"),
        ]
        result = analyser.score_articles(articles)
        assert all(a.sentiment is not None for a in result)
        assert len(result) == 3

    def test_score_articles_returns_same_list(self):
        analyser = _make_analyser(MOCK_POSITIVE_RAW)
        articles = [Article(ticker="X", title="t", text="t")]
        returned = analyser.score_articles(articles)
        assert returned is articles

    def test_api_called_with_truncated_text(self):
        analyser = _make_analyser(MOCK_POSITIVE_RAW)
        long_text = "word " * 2000
        analyser.score(long_text)
        called_text = analyser._client.text_classification.call_args[0][0]
        assert len(called_text) <= 1500

    def test_missing_token_raises(self):
        import os
        with patch.dict(os.environ, {}, clear=True):
            # ensure HF_TOKEN is not set
            os.environ.pop("HF_TOKEN", None)
            with pytest.raises(ValueError, match="HuggingFace token"):
                SentimentAnalyser(api_key=None)
