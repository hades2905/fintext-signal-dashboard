# Financial Intelligence Dashboard

> **FinBERT sentiment · LLM structured extraction · SEC EDGAR filings · Portfolio monitor — built for alternative assets analysis.**

Multi-ticker portfolio scanner and SEC filing analyser targeting the workflows of institutional alternative asset managers (Private Equity, Infrastructure, Real Assets, Private Credit).

Originally designed as a demonstration of applied NLP for the Munich Re GIM Alternative Assets team.

---

## What it does

| Feature | Description |
|---|---|
| **News Sentiment** | FinBERT classifies live Yahoo Finance news per ticker; spaCy NER extracts entities |
| **LLM Investment Briefing** | Qwen2.5-72B generates a 3-sentence portfolio-manager brief from headlines |
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
 │  ProsusAI/FinBERT        │    │  Qwen2.5-72B-Instruct         │
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
git clone https://github.com/hades2905/fintext-signal-dashboard
cd fintext-signal-dashboard

# 2. Create virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy model
python -m spacy download en_core_web_sm

# 5. Set your free HuggingFace token (https://huggingface.co/settings/tokens)
export HF_TOKEN=hf_...

# 6. Run dashboard
streamlit run dashboard/app.py
```

All data sources (Yahoo Finance, SEC EDGAR) are free with no API key. Only the LLM calls (FinBERT + Qwen2.5-72B) require a free HuggingFace token — no billing, no GPU.

---

## Models

| Task | Model | Backend | Cost |
|---|---|---|---|
| Sentiment | [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert) | HuggingFace Inference API | Free tier |
| Structured Extraction | [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) | HuggingFace Inference API | Free tier |
| NER | spaCy `en_core_web_sm` | Local (~12 MB) | Free |
| News | Yahoo Finance | yfinance | Free |
| Filings | SEC EDGAR | `data.sec.gov` public API | Free |

FinBERT is fine-tuned on 10,000 financial sentences and significantly outperforms generic BERT on financial text. Qwen2.5-72B-Instruct extracts fund-level KPIs (IRR, AUM, TVPI, DPI) and qualitative signals from unstructured filing text via `chat.completions` — no local GPU or model download required.

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

## Real Example Data

All outputs below are fetched from **live APIs** (no mocks). Refresh anytime:

```bash
python examples/fetch_real_data.py
```

### Live News Headlines (BX / KKR / APO — 30 articles)

```
[BX] Blackstone Redemptions Test Liquidity As Shares Trade Below Fair Value
[BX] Is Blackstone (BX) Now At A Reasonable Price After Recent Share Price Slide
[BX] Is There Opportunity in Private Credit Stocks or Still Too Much Risk?
[BX] Blue Owl Stock Has Plunged on Private Credit Fears. This Analyst Says They're Overblown.
[BX] KKR Co-CEOs Keep Buying the Stock Dip
[BX] Sector Update: Financial Stocks Advance Pre-Bell Wednesday
```

### Top Named Entities (spaCy NER · 30 articles)

| Entity | Type | Mentions |
|---|---|---|
| KKR | ORG | 8 |
| Blackstone | ORG | 6 |
| NYSE | ORG | 5 |
| Blue Owl Capital | ORG | 2 |
| Chris Kotowski | PERSON | 2 |
| Oppenheimer | ORG | 2 |

### Real SEC EDGAR 8-K Filings (fetched from `data.sec.gov`)

**BX — 2026-01-29 (Q4 2025 Earnings)**
> _"Blackstone Reports Fourth Quarter and Full Year 2025 Results. New York, January 29, 2026: Blackstone (NYSE:BX) today reported its fourth quarter and full year 2025 results. Stephen A. Schwarzman, Chairman and CEO, said, 'Blackstone's extraordinary fourth-quarter results capped a record year for the firm. We delivered again for our limited partners, leading to $71 billion of inflows in the quarter...'"_

**BX — 2025-11-03 (Senior Notes)**
> _"Blackstone Completes Senior Notes Offering. Blackstone (NYSE: BX) announced the completion of the offering of $600 million of 4.300% senior notes due 2030 and $600 million of 4.950% senior notes due 2036..."_

**KKR — 2026-02-05 (Q4 2025 Earnings)**
> _"KKR & Co. Inc. Reports Fourth Quarter 2025 Financial Results. 2025 was a strong year for KKR with record annual figures across key metrics, including Fee Related Earnings, Adjusted Net Income per share, capital raised and capital invested..."_

**KKR — 2026-02-05 (Arctos Acquisition)**
> _"KKR to Acquire Arctos, Establishing a New Platform for Sports, GP Solutions and Secondaries in a Strategic Transaction Initially Valued at $1.4 Billion. Transaction expected to be accretive per share across key financial metrics immediately post-closing..."_

Full example data files are committed to the repo under `examples/data/`:
- `news_articles.json` — 30 real articles with NER annotations
- `news_articles_scored.json` — same + FinBERT sentiment scores per article
- `edgar_filings.json` — 4 real 8-K filings (BX + KKR) with press-release text
- `llm_extractions.json` — Qwen2.5-72B structured extracts from each filing
- `investment_briefings.json` — Qwen2.5-72B PM briefings for BX / KKR / APO
- `top_entities.json` — top 20 named entities by mention count

### LLM Structured Extract — real output (Qwen2.5-72B · `examples/data/llm_extractions.json`)

**BX — Q4 2025 Earnings (2026-01-29)**
```json
{
  "fund_or_entity_name": "Blackstone",
  "geography": "Global",
  "aum_bn_usd": 1300.0,
  "overall_sentiment": "positive",
  "investment_summary": "Blackstone's strong performance in Q4 2025, with record inflows and a focus on large-scale investments in digital and energy infrastructure, positions it well for continued growth."
}
```

**KKR — Arctos Acquisition (2026-02-05)**
```json
{
  "fund_or_entity_name": "Arctos Partners",
  "strategy": "Other",
  "geography": "North America",
  "aum_bn_usd": 15.0,
  "vintage_year": 2019,
  "overall_sentiment": "positive",
  "key_opportunities": [
    "Better serve the sports industry and the sponsor community",
    "Access to strategic, financial and operational resources to accelerate existing businesses",
    "Leverage KKR's broad range of products and capabilities"
  ],
  "investment_summary": "Arctos Partners, a leader in sports franchise investments and GP solutions, is being acquired by KKR, enhancing its capabilities and positioning for growth."
}
```

### Investment Briefing — real output (Qwen2.5-72B · `examples/data/investment_briefings.json`)

**BX:**
> _"The overall sentiment for Blackstone (BX) is decidedly negative, as evidenced by multiple headlines highlighting issues such as redemptions testing liquidity, shares trading below fair value, and significant price slides, which overshadow the few neutral updates on sector performance and leadership actions. A key risk identified is the potential for further liquidity constraints and valuation pressures in BX's private credit and real estate portfolios, particularly given the company's recent challenges in Asia and ongoing market skepticism. It is recommended to closely monitor BX's quarterly earnings reports and any updates on redemption activities, as well as to assess the broader implications of these risks on the alternative asset management sector."_

**KKR:**
> _"The overall sentiment for KKR is mixed, leaning slightly negative, as evidenced by a higher proportion of negative headlines, including concerns over technical inflection points in private-credit stocks and record redemptions in Blackstone's private credit fund, which may reflect broader industry challenges. A key opportunity lies in the positive outlook from RBC Capital's initiation of coverage with an outperform call, suggesting potential upside despite current market skepticism. It is recommended to closely monitor KKR's performance in private credit and any further insider buying activity, as these factors could provide early signals of recovery or continued distress."_

### FinBERT Sentiment — real output (`examples/data/news_articles_scored.json`)

```
[NEGATIVE  2%p] Blackstone Redemptions Test Liquidity As Shares Trade Below Fair Value
[NEGATIVE  1%p] Is Blackstone (BX) Now At A Reasonable Price After Recent Share Price Slide
[NEUTRAL  23%p] Is There Opportunity in Private Credit Stocks or Still Too Much Risk?
[NEGATIVE  1%p] Blue Owl Stock Has Plunged on Private Credit Fears
[NEUTRAL   5%p] KKR Co-CEOs Keep Buying the Stock Dip
```

---

## Development

### Run tests

```bash
# Unit tests (offline, no network, ~7 s)
pytest -m "not integration"

# Integration tests (live yfinance + EDGAR, ~60 s)
pytest -m integration -v

# All tests + coverage
pytest --cov=src --cov-report=term-missing
```

Unit tests run **fully offline** (all HTTP mocked). Integration tests hit live SEC EDGAR and Yahoo Finance and assert on real content quality — including a dedicated guard against XBRL garbage in filing text.

To regenerate the dashboard screenshots from real example data:

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
│   └── extractor.py    # LLMExtractor – structured JSON extraction via Qwen2.5-72B
│   └── ner.py          # Named Entity Recognition via spaCy
├── dashboard/
│   └── app.py          # Streamlit app (3 tabs: News · EDGAR · Portfolio)
├── tests/
│   ├── test_schemas.py
│   ├── test_sentiment.py
│   ├── test_fetcher.py
│   ├── test_edgar.py
│   ├── test_extractor.py
│   └── test_integration.py  # 14 live-network tests (pytest -m integration)
├── examples/
│   ├── fetch_real_data.py   # Fetch + save real data from yfinance + EDGAR
│   └── data/
│       ├── news_articles.json          # 30 real articles with NER annotations
│       ├── news_articles_scored.json   # same + FinBERT sentiment scores (live)
│       ├── edgar_filings.json          # 4 real 8-K filings (BX + KKR)
│       ├── llm_extractions.json        # Qwen2.5-72B structured extracts (live)
│       ├── investment_briefings.json   # Qwen2.5-72B PM briefings per ticker (live)
│       ├── news_articles.csv
│       └── top_entities.json           # Top 20 NER entities by mention count
├── scripts/
│   └── generate_screenshots.py  # Regenerate docs/screenshots from real example data
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
