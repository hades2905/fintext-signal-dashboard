"""
Fetch real data from SEC EDGAR and Yahoo Finance, run NER,
and save example outputs to examples/data/.

No API key required – all public endpoints.
Run:  python examples/fetch_real_data.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp.edgar import fetch_filings
from src.nlp.fetcher import fetch_news
from src.nlp.ner import annotate_articles

OUT = Path(__file__).parent / "data"
OUT.mkdir(parents=True, exist_ok=True)


def _article_to_dict(a) -> dict:
    return {
        "ticker": a.ticker,
        "title": a.title,
        "text": a.text[:800],
        "source": a.source,
        "published_at": a.published_at.isoformat() if a.published_at else None,
        "url": a.url,
        "sentiment": None,  # filled later if token available
        "entities": [{"text": e.text, "label": e.label} for e in a.entities],
    }


def _filing_to_dict(f) -> dict:
    return {
        "ticker": f.ticker,
        "company_name": f.company_name,
        "cik": f.cik,
        "form_type": f.form_type,
        "filed_at": f.filed_at.isoformat() if f.filed_at else None,
        "accession_number": f.accession_number,
        "url": f.url,
        "text_snippet": f.text[:2000],
    }


# -------------------------------------------------------------------------
# 1. News – BX, KKR, APO (yfinance, 10 articles each)
# -------------------------------------------------------------------------
print("=== Fetching news articles (yfinance) ===")
all_articles = []
for ticker in ["BX", "KKR", "APO"]:
    print(f"  {ticker}…", end=" ", flush=True)
    articles = fetch_news(ticker, max_articles=10)
    print(f"{len(articles)} articles")
    all_articles.extend(articles)

print(f"  Running NER on {len(all_articles)} articles…")
annotate_articles(all_articles)

news_out = [_article_to_dict(a) for a in all_articles]
news_path = OUT / "news_articles.json"
news_path.write_text(json.dumps(news_out, indent=2, ensure_ascii=False))
print(f"  → Saved {len(news_out)} articles to {news_path}")

# -------------------------------------------------------------------------
# 2. EDGAR Filings – BX and KKR  8-K, 2 filings each
# -------------------------------------------------------------------------
print("\n=== Fetching SEC EDGAR filings ===")
all_filings = []
for ticker in ["BX", "KKR"]:
    print(f"  {ticker} 8-K…", end=" ", flush=True)
    filings = fetch_filings(ticker, form_type="8-K", max_filings=2)
    print(f"{len(filings)} filings")
    all_filings.extend(filings)

filings_out = [_filing_to_dict(f) for f in all_filings]
filings_path = OUT / "edgar_filings.json"
filings_path.write_text(json.dumps(filings_out, indent=2, ensure_ascii=False))
print(f"  → Saved {len(filings_out)} filings to {filings_path}")

# -------------------------------------------------------------------------
# 3. Summary report (CSV-style)
# -------------------------------------------------------------------------
import csv
csv_path = OUT / "news_articles.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["ticker", "published_at", "source", "title", "url", "entities"])
    writer.writeheader()
    for a in all_articles:
        writer.writerow({
            "ticker": a.ticker,
            "published_at": a.published_at.isoformat() if a.published_at else "",
            "source": a.source or "",
            "title": a.title,
            "url": a.url or "",
            "entities": "; ".join(f"{e.text} [{e.label}]" for e in a.entities),
        })
print(f"\n  → Saved CSV to {csv_path}")

# -------------------------------------------------------------------------
# 4. NER entity summary
# -------------------------------------------------------------------------
from collections import Counter
entity_counter: Counter = Counter()
for a in all_articles:
    for e in a.entities:
        entity_counter[(e.text, e.label)] += 1

top_20 = entity_counter.most_common(20)
ner_path = OUT / "top_entities.json"
ner_path.write_text(json.dumps(
    [{"entity": t, "label": l, "mentions": n} for (t, l), n in top_20],
    indent=2,
))
print(f"  → Saved top entities to {ner_path}")

print("\n✅ All real example data saved to examples/data/")
print(f"   Articles : {len(all_articles)}")
print(f"   Filings  : {len(all_filings)}")
print(f"   Unique entities in top-20: {len(top_20)}")
