# Financial News Sentiment Analyser

> **FinBERT sentiment classification + Named Entity Recognition on live financial news – no API key required, runs fully locally.**

Enter a stock ticker (e.g. `MUV2.DE` for Munich Re) and get an instant sentiment breakdown of the latest news,
extracted entities, and an interactive Streamlit dashboard.

---

## Demo

```bash
streamlit run dashboard/app.py
```

Enter `MUV2.DE` → click **Analyse** → results appear in seconds.

---

## Architecture

```
Yahoo Finance (yfinance)
        │
        ▼  list[Article]
 ┌─────────────────┐
 │    fetcher.py   │  fetch_news(ticker) → title, text, date, source, url
 └────────┬────────┘
          │
          ▼
 ┌─────────────────────────────────┐
 │  sentiment.py – SentimentAnalyser│
 │  Model: ProsusAI/FinBERT        │
 │  → positive / negative / neutral │
 │    + per-class confidence scores │
 └────────┬────────────────────────┘
          │
          ▼
 ┌─────────────────────────────┐
 │  ner.py – extract_entities  │
 │  Model: spaCy en_core_web_sm│
 │  → ORG, PERSON, GPE, ...    │
 └────────┬────────────────────┘
          │
          ▼
 ┌─────────────────┐
 │  Streamlit UI   │  KPI cards · pie · timeline · entity chart · CSV download
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

| Task | Model | Size | Source |
|---|---|---|---|
| Sentiment | [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert) | ~440 MB | HuggingFace Hub |
| NER | spaCy `en_core_web_sm` | ~12 MB | spaCy |
| News | Yahoo Finance | — | yfinance |

FinBERT is a BERT model fine-tuned on 10,000 financial sentences from analyst reports and news, specifically designed for financial sentiment – significantly outperforming generic BERT on financial text.

---

## Example Output (MUV2.DE – Munich Re)

```
Total articles:  15
Positive 🟢:     7  (47%)
Negative 🔴:     4  (27%)
Neutral  ⚪:     4  (27%)
Avg positive score: 0.54

Top entities:
  Munich Re     [ORG]      – 12 mentions
  Torsten Jeworrek [PERSON] – 3 mentions
  Germany       [GPE]      – 5 mentions
```

---

## Development

### Run tests

```bash
pytest                                        # all 18 tests, no model loading
pytest --cov=src --cov-report=term-missing    # with coverage
```

All tests run **offline** (yfinance and FinBERT are mocked).

### Lint

```bash
ruff check .
```

---

## Project Structure

```
finbert-news-sentiment/
├── src/nlp/
│   ├── schemas.py      # Pydantic models (Article, SentimentScore, Entity)
│   ├── fetcher.py      # Yahoo Finance news fetcher
│   ├── sentiment.py    # SentimentAnalyser wrapping FinBERT
│   └── ner.py          # Named Entity Recognition via spaCy
├── dashboard/
│   └── app.py          # Streamlit application
├── tests/
│   ├── test_schemas.py
│   ├── test_sentiment.py
│   └── test_fetcher.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Limitations

- yfinance news coverage varies by ticker; some non-US tickers return fewer articles.
- FinBERT truncates input at 512 tokens; long articles are analysed based on the first ~400 words.
- spaCy `en_core_web_sm` is optimised for English; non-English text may produce lower NER quality.

---

## License

MIT
