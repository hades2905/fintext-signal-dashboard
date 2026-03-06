"""
Integration tests – hit real external APIs (yfinance + SEC EDGAR).

Run with:
    pytest -m integration -v

These are explicitly skipped in CI / normal test runs because they
require network access and may be slow.
"""
from __future__ import annotations

import pytest

# All tests in this module are marked as integration
pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# yfinance / fetcher
# ---------------------------------------------------------------------------

class TestRealNewsIntegration:
    """Fetch real news articles from yfinance for known PE tickers."""

    def test_bx_news_returns_articles(self):
        from src.nlp.fetcher import fetch_news

        articles = fetch_news("BX", max_articles=5)
        assert len(articles) >= 1, "Expected at least one BX article from yfinance"

    def test_article_has_required_fields(self):
        from src.nlp.fetcher import fetch_news

        articles = fetch_news("BX", max_articles=3)
        assert articles, "Expected at least one article"
        art = articles[0]
        assert art.ticker == "BX"
        assert isinstance(art.title, str) and len(art.title) > 5
        assert isinstance(art.url, str) and art.url.startswith("http")

    def test_kkr_news_returns_articles(self):
        from src.nlp.fetcher import fetch_news

        articles = fetch_news("KKR", max_articles=5)
        assert len(articles) >= 1, "Expected at least one KKR article from yfinance"

    def test_ticker_uppercasing_preserved(self):
        from src.nlp.fetcher import fetch_news

        articles = fetch_news("bx", max_articles=3)
        for art in articles:
            assert art.ticker == "BX", "Ticker must be stored in upper-case"


# ---------------------------------------------------------------------------
# SEC EDGAR
# ---------------------------------------------------------------------------

class TestRealEdgarIntegration:
    """Fetch real 8-K filings from SEC EDGAR for BX and KKR."""

    def test_bx_filings_returned(self):
        from src.nlp.edgar import fetch_filings

        filings = fetch_filings("BX", form_type="8-K", max_filings=2)
        assert len(filings) >= 1, "Expected at least one BX 8-K filing"

    def test_filing_fields_populated(self):
        from src.nlp.edgar import fetch_filings

        filings = fetch_filings("BX", form_type="8-K", max_filings=1)
        assert filings, "Expected at least one filing"
        f = filings[0]
        assert f.ticker == "BX"
        assert "Blackstone" in f.company_name
        assert f.cik == "0001393818"
        assert f.accession_number  # non-empty
        assert f.url.startswith("https://")

    def test_filing_text_contains_press_release_content(self):
        """
        The EX-99.1 extraction should yield real financial narrative,
        not XBRL boilerplate (link:definitionLink etc.).
        """
        from src.nlp.edgar import fetch_filings

        filings = fetch_filings("BX", form_type="8-K", max_filings=2)
        # At least one filing should carry readable English prose
        prose_terms = ["Blackstone", "billion", "quarter", "results", "notes"]
        has_good_content = any(
            any(term.lower() in f.text.lower() for term in prose_terms)
            for f in filings
        )
        assert has_good_content, (
            "No filing contained expected press-release keywords. "
            "EX-99.1 extraction may be broken."
        )

    def test_filing_text_not_xbrl_garbage(self):
        from src.nlp.edgar import fetch_filings

        filings = fetch_filings("BX", form_type="8-K", max_filings=2)
        for f in filings:
            if f.text:
                assert "link:definitionLink" not in f.text, (
                    f"Filing {f.accession_number} still contains XBRL garbage"
                )

    def test_kkr_filings_returned(self):
        from src.nlp.edgar import fetch_filings

        filings = fetch_filings("KKR", form_type="8-K", max_filings=2)
        assert len(filings) >= 1, "Expected at least one KKR 8-K filing"

    def test_kkr_filing_company_name(self):
        from src.nlp.edgar import fetch_filings

        filings = fetch_filings("KKR", form_type="8-K", max_filings=1)
        assert filings
        assert "KKR" in filings[0].company_name

    def test_max_filings_respected(self):
        from src.nlp.edgar import fetch_filings

        filings = fetch_filings("BX", form_type="8-K", max_filings=1)
        assert len(filings) <= 1

    def test_unknown_ticker_returns_empty(self):
        from src.nlp.edgar import fetch_filings

        filings = fetch_filings("ZZZNOTTICKER", form_type="8-K", max_filings=2)
        assert filings == [], "Unknown ticker should return empty list"


# ---------------------------------------------------------------------------
# NER on real articles
# ---------------------------------------------------------------------------

class TestRealNERIntegration:
    """Run spaCy NER on real news articles and verify entity extraction."""

    def test_ner_finds_orgs_in_bx_news(self):
        from src.nlp.fetcher import fetch_news
        from src.nlp.ner import annotate_articles

        articles = fetch_news("BX", max_articles=5)
        assert articles

        annotated = annotate_articles(articles)
        all_entities = [ent for art in annotated for ent in (art.entities or [])]
        org_entities = [e for e in all_entities if e.label == "ORG"]
        assert len(org_entities) >= 1, (
            "Expected at least one ORG entity from BX news articles"
        )

    def test_ner_entity_has_required_keys(self):
        from src.nlp.fetcher import fetch_news
        from src.nlp.ner import annotate_articles

        articles = fetch_news("BX", max_articles=3)
        assert articles

        annotated = annotate_articles(articles)
        all_entities = [ent for art in annotated for ent in (art.entities or [])]
        if all_entities:
            e = all_entities[0]
            assert hasattr(e, "text")
            assert hasattr(e, "label")
