"""
Tests for the SEC EDGAR fetcher – all HTTP calls are mocked.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.nlp.edgar import KNOWN_CIKS, fetch_filings, _cik_for_ticker
from src.nlp.schemas import EdgarFiling, FilingType


# ---------------------------------------------------------------------------
# Minimal realistic EDGAR submissions payload
# ---------------------------------------------------------------------------
def _make_submissions_payload(ticker: str = "BX", n: int = 3) -> dict:
    return {
        "name": "Blackstone Inc.",
        "filings": {
            "recent": {
                "form":            ["8-K", "8-K", "10-K"] * (n // 3 + 1),
                "accessionNumber": [f"0001393818-24-{str(i).zfill(6)}" for i in range(n * 3)],
                "filingDate":      ["2024-10-15", "2024-07-18", "2024-02-29"] * (n // 3 + 1),
                "primaryDocument": ["bx-8k.htm", "bx-8k2.htm", "bx-10k.htm"] * (n // 3 + 1),
            }
        },
    }


FAKE_FILING_TEXT = """
BLACKSTONE INC – FORM 8-K
October 15, 2024

Blackstone announces Q3 2024 distributable earnings of $1.1 billion.

Blackstone Real Estate Partners X achieved a net IRR of 19.2% and a TVPI of 1.65x.
AUM reached $1.06 trillion across all strategies.
DPI stands at 0.82x; RVPI at 0.83x.
Deployment pace remains stable; exit environment is described as mixed due to higher rates.
Key risks include rising interest rates and potential slowdown in LP allocations.
Key opportunities include infrastructure buildout and data center demand.
"""


class TestKnownCIKs:
    def test_bx_known(self):
        assert "BX" in KNOWN_CIKS
        assert KNOWN_CIKS["BX"].startswith("0001")

    def test_kkr_known(self):
        assert "KKR" in KNOWN_CIKS

    def test_all_ciks_zero_padded(self):
        for ticker, cik in KNOWN_CIKS.items():
            assert len(cik) == 10, f"CIK for {ticker} is not 10 chars: {cik}"


class TestFetchFilings:
    @patch("src.nlp.edgar.requests.get")
    def test_returns_filings(self, mock_get):
        # First call: company info; subsequent calls: filing text
        sub_resp = MagicMock()
        sub_resp.json.return_value = _make_submissions_payload("BX", n=3)
        sub_resp.raise_for_status = MagicMock()

        txt_resp = MagicMock()
        txt_resp.text = FAKE_FILING_TEXT
        txt_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [sub_resp] + [txt_resp] * 20

        filings = fetch_filings("BX", form_type="8-K", max_filings=2)
        assert len(filings) == 2
        assert all(isinstance(f, EdgarFiling) for f in filings)

    @patch("src.nlp.edgar.requests.get")
    def test_filing_fields_populated(self, mock_get):
        sub_resp = MagicMock()
        sub_resp.json.return_value = _make_submissions_payload("BX", n=3)
        sub_resp.raise_for_status = MagicMock()

        txt_resp = MagicMock()
        txt_resp.text = FAKE_FILING_TEXT
        txt_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [sub_resp] + [txt_resp] * 10

        filings = fetch_filings("BX", form_type="8-K", max_filings=1)
        f = filings[0]

        assert f.ticker == "BX"
        assert f.company_name == "Blackstone Inc."
        assert f.form_type == FilingType.EIGHT_K
        assert isinstance(f.filed_at, datetime)
        assert len(f.text) > 50

    @patch("src.nlp.edgar.requests.get")
    def test_ticker_uppercased(self, mock_get):
        sub_resp = MagicMock()
        sub_resp.json.return_value = _make_submissions_payload("BX")
        sub_resp.raise_for_status = MagicMock()
        txt_resp = MagicMock()
        txt_resp.text = FAKE_FILING_TEXT
        txt_resp.raise_for_status = MagicMock()
        mock_get.side_effect = [sub_resp] + [txt_resp] * 10

        filings = fetch_filings("bx", form_type="8-K", max_filings=1)
        assert filings[0].ticker == "BX"

    @patch("src.nlp.edgar.requests.get")
    def test_max_filings_respected(self, mock_get):
        sub_resp = MagicMock()
        sub_resp.json.return_value = _make_submissions_payload("BX", n=9)
        sub_resp.raise_for_status = MagicMock()
        txt_resp = MagicMock()
        txt_resp.text = FAKE_FILING_TEXT
        txt_resp.raise_for_status = MagicMock()
        mock_get.side_effect = [sub_resp] + [txt_resp] * 20

        filings = fetch_filings("BX", form_type="8-K", max_filings=2)
        assert len(filings) <= 2

    @patch("src.nlp.edgar.requests.get")
    def test_unknown_ticker_returns_empty(self, mock_get):
        mock_get.side_effect = Exception("network error")
        filings = fetch_filings("ZZZZZ", form_type="8-K", max_filings=3)
        assert filings == []

    @patch("src.nlp.edgar.requests.get")
    def test_empty_filing_text_skipped(self, mock_get):
        sub_resp = MagicMock()
        sub_resp.json.return_value = _make_submissions_payload("BX")
        sub_resp.raise_for_status = MagicMock()
        txt_resp = MagicMock()
        txt_resp.text = "   "  # empty / whitespace only
        txt_resp.raise_for_status = MagicMock()
        mock_get.side_effect = [sub_resp] + [txt_resp] * 20

        filings = fetch_filings("BX", form_type="8-K", max_filings=3)
        assert filings == []
