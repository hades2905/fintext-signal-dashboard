"""
Tests for LLMExtractor – HuggingFace inference is mocked.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.nlp.extractor import LLMExtractor, _parse_json_response
from src.nlp.schemas import Article, EdgarFiling, FilingType, StructuredExtract


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_extractor(mock_response: str) -> LLMExtractor:
    """Return an LLMExtractor with a mocked InferenceClient."""
    extractor = LLMExtractor.__new__(LLMExtractor)
    mock_client = MagicMock()
    # Simulate: client.chat.completions.create(...).choices[0].message.content
    mock_choice = MagicMock()
    mock_choice.message.content = mock_response
    mock_client.chat.completions.create.return_value.choices = [mock_choice]
    extractor._client = mock_client
    extractor._model = "mock-model"
    return extractor


VALID_JSON_RESPONSE = json.dumps({
    "fund_or_entity_name": "Blackstone Real Estate Partners X",
    "strategy": "Real Estate",
    "geography": "Global",
    "aum_bn_usd": 180.5,
    "net_irr_pct": 19.2,
    "gross_irr_pct": 24.1,
    "tvpi": 1.65,
    "dpi": 0.82,
    "rvpi": 0.83,
    "vintage_year": 2019,
    "deployment_pace": "stable",
    "exit_environment": "mixed",
    "fundraising_outlook": "strong",
    "key_risks": ["rising interest rates", "reduced LP allocations"],
    "key_opportunities": ["infrastructure buildout", "data center demand"],
    "overall_sentiment": "cautious",
    "investment_summary": (
        "Blackstone's RE portfolio shows resilient performance with net IRR of 19.2%, "
        "though the exit environment remains mixed amid elevated rates. "
        "Monitor Q4 deployment activity closely."
    ),
})

MARKDOWN_WRAPPED_JSON = f"```json\n{VALID_JSON_RESPONSE}\n```"
PARTIAL_JSON = '{"fund_or_entity_name": "KKR European Fund VI", "net_irr_pct": 18.7}'
TRAILING_COMMA_JSON = '{"fund_or_entity_name": "Apollo Fund IX", "key_risks": ["rates",], "tvpi": 1.42}'
GARBAGE_RESPONSE = "I cannot extract structured data from this text. Here is my analysis..."


# ---------------------------------------------------------------------------
# _parse_json_response unit tests
# ---------------------------------------------------------------------------
class TestParseJsonResponse:
    def test_clean_json(self):
        result = _parse_json_response(VALID_JSON_RESPONSE)
        assert result.fund_or_entity_name == "Blackstone Real Estate Partners X"
        assert result.net_irr_pct == pytest.approx(19.2)
        assert result.tvpi == pytest.approx(1.65)
        assert result.strategy == "Real Estate"

    def test_markdown_fenced_json(self):
        result = _parse_json_response(MARKDOWN_WRAPPED_JSON)
        assert result.fund_or_entity_name == "Blackstone Real Estate Partners X"
        assert result.dpi == pytest.approx(0.82)

    def test_partial_json_fills_rest_with_none(self):
        result = _parse_json_response(PARTIAL_JSON)
        assert result.fund_or_entity_name == "KKR European Fund VI"
        assert result.net_irr_pct == pytest.approx(18.7)
        assert result.tvpi is None

    def test_trailing_comma_handled(self):
        result = _parse_json_response(TRAILING_COMMA_JSON)
        assert result.fund_or_entity_name == "Apollo Fund IX"
        assert result.tvpi == pytest.approx(1.42)

    def test_garbage_returns_empty_extract(self):
        result = _parse_json_response(GARBAGE_RESPONSE)
        assert isinstance(result, StructuredExtract)
        assert result.fund_or_entity_name is None

    def test_empty_string_returns_empty_extract(self):
        result = _parse_json_response("")
        assert isinstance(result, StructuredExtract)

    def test_key_risks_parsed_as_list(self):
        result = _parse_json_response(VALID_JSON_RESPONSE)
        assert isinstance(result.key_risks, list)
        assert len(result.key_risks) == 2
        assert "rising interest rates" in result.key_risks

    def test_investment_summary_populated(self):
        result = _parse_json_response(VALID_JSON_RESPONSE)
        assert result.investment_summary is not None
        assert "net IRR" in result.investment_summary


# ---------------------------------------------------------------------------
# LLMExtractor.extract
# ---------------------------------------------------------------------------
class TestLLMExtractorExtract:
    def test_extract_happy_path(self):
        extractor = _make_extractor(VALID_JSON_RESPONSE)
        result = extractor.extract("Blackstone Q3 report...")
        assert result.fund_or_entity_name == "Blackstone Real Estate Partners X"
        assert result.net_irr_pct == pytest.approx(19.2)
        assert result.raw_llm_response == VALID_JSON_RESPONSE

    def test_extract_markdown_wrapped(self):
        extractor = _make_extractor(MARKDOWN_WRAPPED_JSON)
        result = extractor.extract("Some text")
        assert result.fund_or_entity_name == "Blackstone Real Estate Partners X"

    def test_extract_inference_failure_returns_empty(self):
        extractor = LLMExtractor.__new__(LLMExtractor)
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API error")
        extractor._client = mock_client
        extractor._model = "mock-model"

        result = extractor.extract("Any text")
        assert isinstance(result, StructuredExtract)
        assert "API error" in (result.raw_llm_response or "")

    def test_missing_token_raises(self):
        import os
        with patch.dict(os.environ, {"HF_TOKEN": ""}):
            with pytest.raises(ValueError, match="HuggingFace token"):
                LLMExtractor(api_key="")

    def test_extract_filing_annotates_filing(self):
        extractor = _make_extractor(VALID_JSON_RESPONSE)
        filing = EdgarFiling(
            ticker="BX",
            company_name="Blackstone Inc.",
            cik="0001393818",
            form_type=FilingType.EIGHT_K,
            accession_number="0001393818-24-000001",
            text="Blackstone Q3 earnings release with fund IRR data.",
        )
        result = extractor.extract_filing(filing)
        assert result.extracted is not None
        assert result.extracted.net_irr_pct == pytest.approx(19.2)
        assert result is filing  # same object, mutated in-place

    def test_extract_filings_all_annotated(self):
        extractor = _make_extractor(VALID_JSON_RESPONSE)
        filings = [
            EdgarFiling(
                ticker="BX", company_name="Blackstone Inc.", cik="0001393818",
                form_type=FilingType.EIGHT_K,
                accession_number=f"0001393818-24-{i:06d}",
                text=f"Filing text {i}",
            )
            for i in range(3)
        ]
        result = extractor.extract_filings(filings)
        assert all(f.extracted is not None for f in result)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# LLMExtractor.investment_summary
# ---------------------------------------------------------------------------
class TestLLMExtractorSummary:
    def test_summary_called_with_articles(self):
        extractor = LLMExtractor.__new__(LLMExtractor)
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = (
            "Blackstone shows positive near-term momentum driven by strong infrastructure "
            "deal flow. Key risk: rate sensitivity in RE portfolio. Recommend monitoring "
            "Q4 deployment figures."
        )
        mock_client.chat.completions.create.return_value.choices = [mock_choice]
        extractor._client = mock_client
        extractor._model = "mock-model"

        from src.nlp.schemas import SentimentScore
        _score = SentimentScore(label="positive", positive=0.85, negative=0.08, neutral=0.07)
        articles = [
            Article(ticker="BX", title="Blackstone raises record fund", text="...", sentiment=_score),
            Article(ticker="BX", title="BX infrastructure AUM hits $200bn", text="...", sentiment=_score),
        ]
        summary = extractor.investment_summary("BX", articles, "Blackstone Inc.")
        assert isinstance(summary, str)
        assert len(summary) > 20
        mock_client.chat.completions.create.assert_called_once()

    def test_summary_no_articles_returns_default(self):
        extractor = _make_extractor("")
        result = extractor.investment_summary("BX", [])
        assert "No recent news" in result
