"""
Tests for Pydantic schemas – no model loading required.
"""
import pytest
from pydantic import ValidationError

from src.nlp.schemas import Article, Entity, SentimentLabel, SentimentScore


class TestSentimentScore:
    def test_valid(self):
        s = SentimentScore(label="positive", positive=0.9, negative=0.05, neutral=0.05)
        assert s.label == SentimentLabel.POSITIVE

    def test_score_out_of_range(self):
        with pytest.raises(ValidationError):
            SentimentScore(label="positive", positive=1.5, negative=0.0, neutral=0.0)

    def test_all_labels(self):
        for label in ("positive", "negative", "neutral"):
            s = SentimentScore(label=label, positive=0.33, negative=0.33, neutral=0.34)
            assert s.label == label


class TestArticle:
    def test_minimal(self):
        a = Article(ticker="AAPL", title="Apple beats earnings", text="Revenue grew 12%.")
        assert a.sentiment is None
        assert a.entities == []

    def test_with_sentiment(self):
        score = SentimentScore(label="positive", positive=0.91, negative=0.03, neutral=0.06)
        a = Article(ticker="AAPL", title="Great results", text="...", sentiment=score)
        assert a.sentiment.label == SentimentLabel.POSITIVE

    def test_with_entities(self):
        entities = [Entity(text="Apple", label="ORG"), Entity(text="Tim Cook", label="PERSON")]
        a = Article(ticker="AAPL", title="t", text="t", entities=entities)
        assert len(a.entities) == 2

    def test_ticker_stored(self):
        a = Article(ticker="MUV2.DE", title="t", text="t")
        assert a.ticker == "MUV2.DE"
