"""
Fetch recent financial news articles for a given ticker via yfinance.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import yfinance as yf

from .schemas import Article

logger = logging.getLogger(__name__)


def fetch_news(ticker: str, max_articles: int = 20) -> list[Article]:
    """
    Return up to *max_articles* recent news articles for *ticker*.

    Uses yfinance under the hood – no API key required.

    Parameters
    ----------
    ticker:
        Yahoo Finance ticker symbol (e.g. "AAPL", "MUV2.DE", "BLK").
    max_articles:
        Maximum number of articles to return.
    """
    t = yf.Ticker(ticker)
    raw_news = t.news or []

    articles: list[Article] = []
    for item in raw_news[:max_articles]:
        try:
            content = item.get("content", {})

            # Title
            title = content.get("title", "") or item.get("title", "")
            if not title:
                continue

            # Body text – prefer full summary, fall back to title only
            body = content.get("summary", "") or content.get("body", "") or title

            # URL
            url = None
            cp = content.get("canonicalUrl", {})
            if isinstance(cp, dict):
                url = cp.get("url")
            if not url:
                url = item.get("link", None)

            # Published timestamp
            pub_ts = content.get("pubDate") or item.get("providerPublishTime")
            published_at: Optional[datetime] = None
            if isinstance(pub_ts, (int, float)):
                published_at = datetime.fromtimestamp(pub_ts, tz=timezone.utc)
            elif isinstance(pub_ts, str):
                try:
                    published_at = datetime.fromisoformat(pub_ts.replace("Z", "+00:00"))
                except ValueError:
                    pass

            # Source / provider
            provider = content.get("provider", {})
            source = (
                provider.get("displayName")
                if isinstance(provider, dict)
                else item.get("publisher", None)
            )

            articles.append(
                Article(
                    ticker=ticker.upper(),
                    title=title,
                    text=body,
                    url=url,
                    published_at=published_at,
                    source=source,
                )
            )
        except Exception as exc:
            logger.warning("Skipping news item: %s", exc)
            continue

    logger.info("Fetched %d articles for %s", len(articles), ticker)
    return articles
