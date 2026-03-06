"""
Tests for news fetcher – yfinance is mocked.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.nlp.fetcher import fetch_news
from src.nlp.schemas import Article

# ---------------------------------------------------------------------------
# Realistic mock yfinance news payload (new format)
# ---------------------------------------------------------------------------
MOCK_NEWS = [
    {
        "content": {
            "title": "Munich Re posts record profit for Q3 2024",
            "summary": "Munich Re, the world's largest reinsurer, reported a record quarterly net profit.",
            "pubDate": "2024-10-15T08:00:00Z",
            "canonicalUrl": {"url": "https://example.com/muvde-1"},
            "provider": {"displayName": "Reuters"},
        }
    },
    {
        "content": {
            "title": "Reinsurance market faces headwinds from climate risk",
            "summary": "Analysts warn that rising nat-cat losses may pressure reinsurance margins.",
            "pubDate": "2024-10-14T12:30:00Z",
            "canonicalUrl": {"url": "https://example.com/muvde-2"},
            "provider": {"displayName": "Bloomberg"},
        }
    },
    {
        # Item missing title → should be skipped
        "content": {
            "title": "",
            "summary": "Some content without a title.",
        }
    },
]


def _make_mock_ticker(news: list) -> MagicMock:
    mock = MagicMock()
    mock.news = news
    return mock


class TestFetchNews:
    @patch("src.nlp.fetcher.yf.Ticker")
    def test_returns_articles(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _make_mock_ticker(MOCK_NEWS)
        articles = fetch_news("MUV2.DE", max_articles=10)
        # 3 items – 1 without title = 2 valid
        assert len(articles) == 2

    @patch("src.nlp.fetcher.yf.Ticker")
    def test_article_fields(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _make_mock_ticker(MOCK_NEWS)
        articles = fetch_news("MUV2.DE")
        a = articles[0]
        assert a.ticker == "MUV2.DE"
        assert "Munich Re" in a.title
        assert a.url == "https://example.com/muvde-1"
        assert a.source == "Reuters"
        assert isinstance(a.published_at, datetime)

    @patch("src.nlp.fetcher.yf.Ticker")
    def test_ticker_uppercased(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _make_mock_ticker(MOCK_NEWS)
        articles = fetch_news("muv2.de")
        assert articles[0].ticker == "MUV2.DE"

    @patch("src.nlp.fetcher.yf.Ticker")
    def test_empty_news_returns_empty(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _make_mock_ticker([])
        articles = fetch_news("UNKNOWN")
        assert articles == []

    @patch("src.nlp.fetcher.yf.Ticker")
    def test_max_articles_respected(self, mock_ticker_cls):
        many_news = MOCK_NEWS * 10
        mock_ticker_cls.return_value = _make_mock_ticker(many_news)
        articles = fetch_news("AAPL", max_articles=3)
        assert len(articles) <= 3
