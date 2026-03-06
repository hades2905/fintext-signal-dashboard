"""
SEC EDGAR 8-K / 10-K filing fetcher – no API key required.

Uses the public SEC EDGAR full-text search and submissions API:
  https://data.sec.gov/submissions/CIK{cik}.json
  https://efts.sec.gov/LATEST/search-index?q=...&forms=8-K

Rate limit: 10 requests/second (EDGAR fair-use policy).
We add a small sleep between calls to stay compliant.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Optional

import requests

from .schemas import EdgarFiling, FilingType

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "finbert-news-sentiment research@example.com",  # EDGAR requires a User-Agent
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

_SEARCH_HEADERS = {
    "User-Agent": "finbert-news-sentiment research@example.com",
    "Accept-Encoding": "gzip, deflate",
}

# Well-known CIK numbers for major alternative asset managers
KNOWN_CIKS: dict[str, str] = {
    "BX":   "0001393818",   # Blackstone Inc.
    "KKR":  "0001404912",   # KKR & Co. Inc.
    "APO":  "0001411579",   # Apollo Global Management
    "ARES": "0001555280",   # Ares Management
    "CG":   "0001282266",   # Carlyle Group
    "BAM":  "0001001085",   # Brookfield Asset Management
}

_SLEEP = 0.15  # seconds between requests – stay within 10 req/s


def _cik_for_ticker(ticker: str) -> Optional[str]:
    """Resolve ticker → zero-padded CIK via EDGAR company search."""
    if ticker.upper() in KNOWN_CIKS:
        return KNOWN_CIKS[ticker.upper()]
    try:
        url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms=8-K"
        r = requests.get(url, headers=_SEARCH_HEADERS, timeout=10)
        r.raise_for_status()
        hits = r.json().get("hits", {}).get("hits", [])
        if hits:
            return hits[0].get("_source", {}).get("file_num", None)
    except Exception as exc:
        logger.warning("CIK lookup failed for %s: %s", ticker, exc)
    return None


def _get_company_info(cik: str) -> dict:
    """Fetch the submissions JSON for a CIK."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(url, headers=_HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()


def _fetch_filing_text(accession_raw: str, cik: str) -> tuple[str, str]:
    """
    Download the primary document text for a filing.
    Returns (text, filing_url).
    """
    acc = accession_raw.replace("-", "")
    acc_fmt = f"{accession_raw}"
    # Index page
    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{acc_fmt}-index.htm"
    )
    filing_url = index_url
    try:
        idx_r = requests.get(index_url, headers=_SEARCH_HEADERS, timeout=10)
        # Try to find the primary document link
        # Fallback: just fetch the .txt version
        txt_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{acc_fmt}.txt"
        time.sleep(_SLEEP)
        txt_r = requests.get(txt_url, headers=_SEARCH_HEADERS, timeout=15)
        raw = txt_r.text
        # Strip SEC headers and XML tags to get readable text
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"\s{3,}", "\n\n", text)
        filing_url = txt_url
        return text[:40_000], filing_url  # cap at 40k chars for LLM
    except Exception as exc:
        logger.warning("Could not fetch filing text for %s: %s", accession_raw, exc)
        return "", filing_url


def fetch_filings(
    ticker: str,
    form_type: str = "8-K",
    max_filings: int = 5,
) -> list[EdgarFiling]:
    """
    Fetch recent SEC filings for *ticker* from EDGAR.

    Parameters
    ----------
    ticker:
        Ticker symbol (e.g. "BX", "KKR"). Uses KNOWN_CIKS for major PE firms.
    form_type:
        SEC form type: "8-K", "10-K", "10-Q".
    max_filings:
        Maximum number of filings to retrieve.

    Returns
    -------
    list[EdgarFiling]
        List of filings with full text populated.
    """
    cik = KNOWN_CIKS.get(ticker.upper()) or _cik_for_ticker(ticker)
    if not cik:
        logger.error("Could not resolve CIK for ticker %s", ticker)
        return []

    try:
        info = _get_company_info(cik)
    except Exception as exc:
        logger.error("EDGAR submissions fetch failed for %s: %s", ticker, exc)
        return []

    company_name = info.get("name", ticker.upper())
    filings_data = info.get("filings", {}).get("recent", {})

    forms = filings_data.get("form", [])
    accessions = filings_data.get("accessionNumber", [])
    filed_dates = filings_data.get("filingDate", [])
    descriptions = filings_data.get("primaryDocument", [])

    results: list[EdgarFiling] = []
    for form, acc, filed, _doc in zip(forms, accessions, filed_dates, descriptions):
        if len(results) >= max_filings:
            break
        if form != form_type:
            continue

        try:
            filed_at = datetime.strptime(filed, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            filed_at = None

        time.sleep(_SLEEP)
        text, url = _fetch_filing_text(acc, cik)
        if not text.strip():
            continue

        try:
            ft = FilingType(form_type)
        except ValueError:
            ft = FilingType.OTHER

        results.append(
            EdgarFiling(
                ticker=ticker.upper(),
                company_name=company_name,
                cik=cik,
                form_type=ft,
                filed_at=filed_at,
                accession_number=acc,
                text=text,
                url=url,
            )
        )
        logger.info("Fetched %s filing %s for %s", form_type, acc, ticker)

    return results
