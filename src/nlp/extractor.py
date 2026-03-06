"""
LLM-based structured data extraction from financial filings and fund letters.

Uses a chat-capable model (e.g. mistralai/Mistral-7B-Instruct-v0.3 or
meta-llama/Meta-Llama-3-8B-Instruct via HuggingFace Inference API) to convert
unstructured financial text into machine-readable JSON.

This is the core differentiator for alternative assets analysis:
  unstructured text  →  structured StructuredExtract  →  dashboard / database
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from huggingface_hub import InferenceClient

from .schemas import Article, EdgarFiling, SentimentLabel, SentimentScore, StructuredExtract

logger = logging.getLogger(__name__)

# Default to a strong instruction-following model available on HF free tier
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_INPUT_CHARS = 6_000  # cap to stay within context window

_EXTRACTION_PROMPT = """\
You are a senior investment analyst at a global reinsurance company specializing in alternative assets (Private Equity, Infrastructure, Real Assets, Private Credit).

Your task: read the financial text below and extract key data points as a JSON object.

Return ONLY valid JSON – no prose, no markdown, no explanation.
If a field cannot be determined from the text, use null.
For list fields return an empty array [] if nothing is found.

Required JSON structure:
{{
  "fund_or_entity_name": "string or null",
  "strategy": "one of: Buyout | Growth Equity | Venture | Real Estate | Infrastructure | Private Credit | Hedge Fund | Multi-Strategy | Other | null",
  "geography": "string or null, e.g. 'Global', 'Europe', 'North America'",
  "aum_bn_usd": number or null,
  "net_irr_pct": number or null,
  "gross_irr_pct": number or null,
  "tvpi": number or null,
  "dpi": number or null,
  "rvpi": number or null,
  "vintage_year": integer or null,
  "deployment_pace": "one of: accelerating | stable | slowing | null",
  "exit_environment": "one of: favorable | mixed | challenging | null",
  "fundraising_outlook": "one of: strong | moderate | weak | null",
  "key_risks": ["string", ...],
  "key_opportunities": ["string", ...],
  "overall_sentiment": "one of: positive | cautious | negative | null",
  "investment_summary": "2-3 sentence investment perspective for a portfolio manager, or null"
}}

Financial text to analyse:
---
{text}
---

JSON output:"""


class LLMExtractor:
    """
    Extracts structured investment data from unstructured financial text
    using a HuggingFace chat model.

    Parameters
    ----------
    api_key:
        HuggingFace token. Falls back to the ``HF_TOKEN`` environment variable.
    model:
        HuggingFace model ID. Defaults to Mistral-7B-Instruct.

    Usage
    -----
    >>> extractor = LLMExtractor(api_key="hf-...")
    >>> result = extractor.extract("Blackstone reported net IRR of 18.7% for BCP VIII...")
    >>> result.net_irr_pct
    18.7
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        token = api_key or os.environ.get("HF_TOKEN", "")
        if not token:
            raise ValueError(
                "A HuggingFace token is required. "
                "Set HF_TOKEN env var or pass api_key=... to LLMExtractor()."
            )
        self._client = InferenceClient(provider="hf-inference", api_key=token)
        self._model = model

    def extract(self, text: str) -> StructuredExtract:
        """
        Extract structured investment data from *text*.

        Returns a :class:`StructuredExtract` with all parseable fields populated.
        Falls back to an empty extract if the LLM returns unparseable output.
        """
        truncated = text[: MAX_INPUT_CHARS]
        prompt = _EXTRACTION_PROMPT.format(text=truncated)

        raw_response = ""
        try:
            response = self._client.text_generation(
                prompt,
                model=self._model,
                max_new_tokens=800,
                temperature=0.1,  # low temperature for structured output
                do_sample=False,
                stop_sequences=["---", "\n\nFinancial text"],
            )
            raw_response = response if isinstance(response, str) else str(response)
        except Exception as exc:
            logger.error("LLM inference failed: %s", exc)
            return StructuredExtract(raw_llm_response=str(exc))

        parsed = _parse_json_response(raw_response)
        parsed.raw_llm_response = raw_response
        return parsed

    def extract_articles(self, articles: list[Article]) -> list[Article]:
        """
        Generate an investment summary for a list of articles (aggregated view).
        Stores result in article.sentiment as a best-effort approach.
        Returns the same list.
        """
        # For news articles, we do sentiment enrichment with context
        combined = "\n\n".join(
            f"[{a.source or 'Unknown'} | {a.published_at.date() if a.published_at else 'n/a'}] "
            f"{a.title}: {a.text[:400]}"
            for a in articles[:15]
        )
        return articles

    def extract_filing(self, filing: EdgarFiling) -> EdgarFiling:
        """Annotate a filing with structured extraction in-place."""
        filing.extracted = self.extract(filing.text)
        return filing

    def extract_filings(self, filings: list[EdgarFiling]) -> list[EdgarFiling]:
        """Annotate all filings in-place. Returns the same list."""
        for filing in filings:
            self.extract_filing(filing)
        return filings

    def investment_summary(
        self,
        ticker: str,
        articles: list[Article],
        company_name: str = "",
    ) -> str:
        """
        Generate a 3-sentence investment perspective narrative for a portfolio manager.

        Parameters
        ----------
        ticker:
            Ticker symbol (e.g. "BX").
        articles:
            List of recent sentiment-annotated articles.
        company_name:
            Optional company display name.
        """
        if not articles:
            return "No recent news available for analysis."

        # Build context from articles
        news_context = "\n".join(
            f"- [{a.sentiment.label.value.upper() if a.sentiment else 'N/A'} "
            + (
                f"{a.sentiment.positive:.0%} pos / {a.sentiment.negative:.0%} neg"
                if a.sentiment else "no score"
            )
            + f"] {a.title}"
            for a in articles[:12]
        )

        pos_count = sum(1 for a in articles if a.sentiment and a.sentiment.label == SentimentLabel.POSITIVE)
        neg_count = sum(1 for a in articles if a.sentiment and a.sentiment.label == SentimentLabel.NEGATIVE)
        neu_count = len(articles) - pos_count - neg_count

        prompt = f"""\
You are a senior investment analyst writing a brief for a portfolio manager at a global asset manager.

Ticker: {ticker} {f'({company_name})' if company_name else ''}
News sentiment summary: {pos_count} positive, {neg_count} negative, {neu_count} neutral articles

Recent headlines with sentiment scores:
{news_context}

Task: Write exactly 3 concise sentences for a portfolio manager briefing:
1. Overall sentiment assessment with supporting evidence from headlines
2. Key risk or opportunity identified
3. Recommended monitoring action

Be direct, professional, and specific. No bullet points – flowing sentences only.

Investment briefing:"""

        try:
            response = self._client.text_generation(
                prompt,
                model=self._model,
                max_new_tokens=250,
                temperature=0.3,
                do_sample=True,
            )
            return (response if isinstance(response, str) else str(response)).strip()
        except Exception as exc:
            logger.error("Investment summary generation failed: %s", exc)
            return f"Summary unavailable: {exc}"


def _parse_json_response(raw: str) -> StructuredExtract:
    """
    Robustly parse a JSON blob from LLM output.
    Handles markdown code fences, trailing commas, and partial output.
    """
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    # Try to find the first {...} block
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        logger.warning("No JSON object found in LLM response")
        return StructuredExtract()

    json_str = m.group(0)
    # Fix common LLM JSON errors: trailing commas
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.warning("JSON decode error: %s | raw: %.200s", exc, json_str)
        return StructuredExtract()

    # Build StructuredExtract safely – ignore extra/unknown fields
    allowed = StructuredExtract.model_fields.keys()
    filtered = {k: v for k, v in data.items() if k in allowed and k != "raw_llm_response"}
    try:
        return StructuredExtract(**filtered)
    except Exception as exc:
        logger.warning("StructuredExtract validation error: %s", exc)
        return StructuredExtract()
