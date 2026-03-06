# Financial Intelligence Dashboard

> **FinBERT sentiment · LLM structured extraction · SEC EDGAR filings · Portfolio monitor — built for alternative assets analysis.**

Multi-ticker portfolio scanner and SEC filing analyser targeting the workflows of institutional alternative asset managers (Private Equity, Infrastructure, Real Assets, Private Credit).

Originally designed as a demonstration of applied NLP for the Munich Re GIM Alternative Assets team.

---

## What it does

| Feature | Description |
|---|---|
| **News Sentiment** | FinBERT classifies live Yahoo Finance news per ticker; spaCy NER extracts entities |
| **LLM Investment Briefing** | Mistral-7B generates a 3-sentence portfolio-manager brief from headlines |
| **SEC EDGAR Filing Ingest** | Fetches real 8-K / 10-K from `data.sec.gov` — no API key required |
| **LLM Structured Extraction** | Extracts IRR, AUM, TVPI, DPI, key risks & opportunities as typed JSON from free text |
| **Portfolio Monitor** | Batch-scan multiple tickers; colour-coded heatmap of sentiment across holdings |
| **CSV Export** | All results downloadable for Databricks / Power BI integration |

---

## Demo

```bash
streamlit run dashboard/app.py
```

Three tabs: **📰 News Sentiment** · **📂 SEC EDGAR Filings** · **🏦 Portfolio Monitor**

---

## Screenshots

| Sentiment Distribution | Sentiment Over Time |
|:---:|:---:|
| ![Sentiment pie](docs/screenshots/sentiment_pie.png) | ![Timeline](docs/screenshots/sentiment_timeline.png) |

| Score Distribution | Sentiment by Ticker |
|:---:|:---:|
| ![Score distribution](docs/screenshots/score_distribution.png) | ![By ticker](docs/screenshots/sentiment_by_ticker.png) |

**Portfolio Sentiment Heatmap**

![Portfolio heatmap](docs/screenshots/portfolio_heatmap.png)

**LLM-Extracted Fund KPIs from SEC EDGAR Filings**

![EDGAR KPI extraction](docs/screenshots/edgar_kpi_extraction.png)

**Top Named Entities**

![Top entities](docs/screenshots/top_entities.png)

---

## Architecture

```
Yahoo Finance (yfinance)          SEC EDGAR (data.sec.gov)
        │                                  │
        ▼  list[Article]                   ▼  list[EdgarFiling]
 ┌─────────────────┐              ┌──────────────────────┐
 │    fetcher.py   │              │      edgar.py         │
 │  fetch_news()   │              │  fetch_filings()      │
 └────────┬────────┘              └────────┬─────────────┘
          │                                │
          ▼                                ▼
 ┌──────────────────────────┐    ┌──────────────────────────────┐
 │  sentiment.py            │    │  extractor.py – LLMExtractor  │
 │  ProsusAI/FinBERT        │    │  Mistral-7B-Instruct          │
 │  → pos / neg / neutral   │    │  → StructuredExtract (JSON)   │
 │    + confidence scores   │    │    IRR · AUM · TVPI · DPI     │
 └────────┬─────────────────┘    │    risks · opportunities      │
          │                      │    + investment_summary()     │
          ▼                      └────────────┬─────────────────┘
 ┌─────────────────────────────┐              │
 │  ner.py – spaCy NER         │              │
 │  → ORG, PERSON, GPE, ...   │              │
 └────────┬────────────────────┘              │
          │                                   │
          └──────────────┬────────────────────┘
                         ▼
               ┌─────────────────┐
               │  Streamlit UI   │
               │  Tab 1: News    │
               │  Tab 2: EDGAR   │
               │  Tab 3: Portfolio│
               └─────────────────┘
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/<you>/finbert-news-sentiment
cd finbert-news-sentiment

# 2. Create virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy model
python -m spacy download en_core_web_sm

# 5. Run dashboard (FinBERT downloads ~450 MB on first run)
streamlit run dashboard/app.py
```

No API key, no cloud service, no cost.

---

## Models

| Task | Model | Backend | Cost |
|---|---|---|---|
| Sentiment | [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert) | HuggingFace Inference API | Free tier |
| Structured Extraction | [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | HuggingFace Inference API | Free tier |
| NER | spaCy `en_core_web_sm` | Local (~12 MB) | Free |
| News | Yahoo Finance | yfinance | Free |
| Filings | SEC EDGAR | `data.sec.gov` public API | Free |

FinBERT is fine-tuned on 10,000 financial sentences and significantly outperforms generic BERT on financial text. Mistral-7B extracts fund-level KPIs (IRR, AUM, TVPI, DPI) and qualitative signals from unstructured filing text. No local GPU or model download required.

---

## Supported Tickers (EDGAR)

| Ticker | Company | Strategy |
|---|---|---|
| BX | Blackstone Inc. | PE / RE / Credit / Infra |
| KKR | KKR & Co. | PE / Credit / Infra |
| APO | Apollo Global Management | PE / Credit |
| ARES | Ares Management | Credit / PE / RE |
| CG | Carlyle Group | PE / Credit / RE |
| BAM | Brookfield Asset Management | Infra / RE / PE |

All others via Yahoo Finance news (ticker symbol). CIK auto-lookup for unknown tickers.

---

## Example Output (BX – Blackstone Q3 8-K)

**LLM Structured Extract:**
```json
{
  "fund_or_entity_name": "Blackstone Real Estate Partners X",
  "strategy": "Real Estate",
  "geography": "Global",
  "aum_bn_usd": 180.5,
  "net_irr_pct": 19.2,
  "tvpi": 1.65,
  "dpi": 0.82,
  "deployment_pace": "stable",
  "exit_environment": "mixed",
  "key_risks": ["rising interest rates", "reduced LP allocations"],
  "key_opportunities": ["infrastructure buildout", "data center demand"],
  "overall_sentiment": "cautious",
  "investment_summary": "Blackstone's RE portfolio shows resilient performance..."
}
```

**Investment Briefing (LLM-generated):**
> _"News sentiment for BX is cautiously positive (54% positive, 28% negative across 14 articles), driven by strong infrastructure fundraising and data centre deal flow. Key risk remains rate sensitivity in the core real estate portfolio with 3 articles flagging elevated vacancy in European office. Recommend monitoring Q4 deployment activity and LP re-up rates ahead of earnings."_

---

## Development

### Run tests

```bash
pytest                                        # all 46 tests, no model loading, no network
pytest --cov=src --cov-report=term-missing    # with coverage
```

All tests run **offline** (yfinance and FinBERT are mocked).

To regenerate the dashboard screenshots from mock data:

```bash
python scripts/generate_screenshots.py
```

### Lint

```bash
ruff check .
```

---

## Project Structure

```
finbert-news-sentiment/
├── src/nlp/
│   ├── schemas.py      # Pydantic models (Article, EdgarFiling, StructuredExtract, …)
│   ├── fetcher.py      # Yahoo Finance news fetcher
│   ├── edgar.py        # SEC EDGAR 8-K / 10-K fetcher (no API key)
│   ├── sentiment.py    # SentimentAnalyser wrapping FinBERT
│   ├── extractor.py    # LLMExtractor – structured JSON extraction via Mistral-7B
│   └── ner.py          # Named Entity Recognition via spaCy
├── dashboard/
│   └── app.py          # Streamlit app (3 tabs: News · EDGAR · Portfolio)
├── tests/
│   ├── test_schemas.py
│   ├── test_sentiment.py
│   ├── test_fetcher.py
│   ├── test_edgar.py
│   └── test_extractor.py
├── scripts/
│   └── generate_screenshots.py  # Regenerate docs/screenshots from mock data
├── docs/
│   └── screenshots/    # Static plot previews embedded in README
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Limitations

- yfinance news coverage varies by ticker; some non-US tickers return fewer articles.
- FinBERT truncates input at 512 tokens; long articles are analysed based on the first ~400 words.
- spaCy `en_core_web_sm` is optimised for English; non-English text may produce lower NER quality.
- LLM extraction quality depends on filing structure; very short or boilerplate 8-Ks may yield sparse extracts.
- EDGAR rate limit: 10 requests/second (built-in 150ms sleep between calls).

---

## Relevance to Alternative Asset Management

This project is directly applicable to the following institutional workflows:

- **Automated GP letter / fund update ingestion** – LLM extracts fund KPIs without manual reading
- **Portfolio monitoring** – batch sentiment scan of portfolio companies identifies emerging risks
- **Regulatory filing surveillance** – 8-K monitoring for material events in portfolio holdings
- **Quantitative research** – sentiment scores as factor signals for allocation models

---

## License

MIT
