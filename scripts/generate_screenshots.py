"""
Generate dashboard screenshots from REAL example data and save to docs/screenshots/.
Data source: examples/data/ (fetched from live yfinance + SEC EDGAR + FinBERT + Qwen2.5-72B)
Run:  python scripts/generate_screenshots.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import write_image

OUT  = Path(__file__).parent.parent / "docs" / "screenshots"
DATA = Path(__file__).parent.parent / "examples" / "data"
OUT.mkdir(parents=True, exist_ok=True)

SENTIMENT_COLORS = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral":  "#95a5a6",
}

# ── Load real data ────────────────────────────────────────────────────────────
articles_raw  = json.load(open(DATA / "news_articles_scored.json"))
top_ents_raw  = json.load(open(DATA / "top_entities.json"))
extractions   = json.load(open(DATA / "llm_extractions.json"))

rows = []
for a in articles_raw:
    sent = a.get("sentiment")
    if not sent:
        continue
    pub = a.get("published_at") or a.get("published")
    try:
        pub_date = datetime.fromisoformat(str(pub)).date() if pub else None
    except Exception:
        pub_date = None
    rows.append({
        "ticker":    a.get("ticker", ""),
        "published": pub_date,
        "source":    a.get("source", ""),
        "label":     sent["label"],
        "positive":  sent["positive"],
        "negative":  sent["negative"],
        "neutral":   sent["neutral"],
    })

df = pd.DataFrame(rows)

# -------------------------------------------------------------------
# 1. Sentiment distribution pie  (real counts)
# -------------------------------------------------------------------
pie_df = df["label"].value_counts().reset_index()
pie_df.columns = ["label", "count"]
fig_pie = px.pie(
    pie_df, names="label", values="count", color="label",
    color_discrete_map=SENTIMENT_COLORS, hole=0.4,
    title="Sentiment Distribution — BX / KKR / APO (30 real articles)",
)
fig_pie.update_layout(margin=dict(t=55, b=10), height=380, width=560)
write_image(fig_pie, OUT / "sentiment_pie.png", scale=2)
print("Saved sentiment_pie.png")

# -------------------------------------------------------------------
# 2. Sentiment over time (real dates)
# -------------------------------------------------------------------
time_df = df.dropna(subset=["published"]).groupby(["published", "label"]).size().reset_index(name="count")
fig_time = px.bar(
    time_df, x="published", y="count", color="label",
    color_discrete_map=SENTIMENT_COLORS, barmode="stack",
    title="Sentiment Over Time — real articles (March 2026)",
    labels={"published": "Date", "count": "Articles"},
)
fig_time.update_layout(height=380, width=760, margin=dict(t=55, b=10))
write_image(fig_time, OUT / "sentiment_timeline.png", scale=2)
print("Saved sentiment_timeline.png")

# -------------------------------------------------------------------
# 3. Sentiment by ticker (real counts)
# -------------------------------------------------------------------
ticker_df = df.groupby(["ticker", "label"]).size().reset_index(name="count")
fig_tick = px.bar(
    ticker_df, x="ticker", y="count", color="label",
    color_discrete_map=SENTIMENT_COLORS, barmode="group",
    title="Sentiment by Ticker — BX / KKR / APO",
    labels={"ticker": "Ticker", "count": "Articles"},
)
fig_tick.update_layout(height=380, width=560, margin=dict(t=55, b=10))
write_image(fig_tick, OUT / "sentiment_by_ticker.png", scale=2)
print("Saved sentiment_by_ticker.png")

# -------------------------------------------------------------------
# 4. Score distribution (real confidence scores)
# -------------------------------------------------------------------
fig_box = go.Figure()
for label, color in SENTIMENT_COLORS.items():
    subset = df[df["label"] == label]
    if subset.empty:
        continue
    fig_box.add_trace(go.Box(y=subset[label], name=label.capitalize(), marker_color=color, boxmean=True))
fig_box.update_layout(
    title="FinBERT Confidence Score Distribution (real scores)",
    yaxis_title="Confidence Score",
    height=360, width=560, margin=dict(t=55, b=10),
)
write_image(fig_box, OUT / "score_distribution.png", scale=2)
print("Saved score_distribution.png")

# -------------------------------------------------------------------
# 5. Top named entities (real NER from spaCy)
# -------------------------------------------------------------------
ent_rows = []
for e in top_ents_raw:
    ent_rows.append({"Entity": e["entity"], "Type": e["label"], "Mentions": e["mentions"]})
ent_df = pd.DataFrame(ent_rows).head(15)
fig_ent = px.bar(
    ent_df, x="Mentions", y="Entity", color="Type",
    orientation="h",
    title="Top Named Entities — spaCy NER on 30 real articles",
    height=460, width=680,
)
fig_ent.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(t=55, b=10))
write_image(fig_ent, OUT / "top_entities.png", scale=2)
print("Saved top_entities.png")

# -------------------------------------------------------------------
# 6. Portfolio sentiment heatmap (real FinBERT scores per ticker)
# -------------------------------------------------------------------
portfolio_tickers = sorted(df["ticker"].unique())
pos_vals, neg_vals, neu_vals = [], [], []
for tkr in portfolio_tickers:
    sub = df[df["ticker"] == tkr]
    total = len(sub)
    pos_vals.append(round(100 * (sub["label"] == "positive").sum() / total, 1))
    neg_vals.append(round(100 * (sub["label"] == "negative").sum() / total, 1))
    neu_vals.append(round(100 * (sub["label"] == "neutral").sum() / total, 1))

fig_heat = go.Figure()
fig_heat.add_trace(go.Heatmap(
    z=[pos_vals], x=portfolio_tickers, y=["Positive %"],
    colorscale=[[0, "#f0faf0"], [1, "#2ecc71"]],
    zmin=0, zmax=100,
    text=[[f"{v:.0f}%" for v in pos_vals]], texttemplate="%{text}",
    showscale=False,
))
fig_heat.add_trace(go.Heatmap(
    z=[neg_vals], x=portfolio_tickers, y=["Negative %"],
    colorscale=[[0, "#fff5f5"], [1, "#e74c3c"]],
    zmin=0, zmax=100,
    text=[[f"{v:.0f}%" for v in neg_vals]], texttemplate="%{text}",
    showscale=False,
))
fig_heat.add_trace(go.Heatmap(
    z=[neu_vals], x=portfolio_tickers, y=["Neutral %"],
    colorscale=[[0, "#f8f9fa"], [1, "#7f8c8d"]],
    zmin=0, zmax=100,
    text=[[f"{v:.0f}%" for v in neu_vals]], texttemplate="%{text}",
    showscale=False,
))
fig_heat.update_layout(
    title="Portfolio Sentiment Heatmap — real FinBERT scores",
    height=320, width=720, margin=dict(t=55, b=10),
)
write_image(fig_heat, OUT / "portfolio_heatmap.png", scale=2)
print("Saved portfolio_heatmap.png")

# -------------------------------------------------------------------
# 7. LLM-Extracted KPIs from real EDGAR filings (Qwen2.5-72B)
# -------------------------------------------------------------------
kpi_rows = []
for ex in extractions:
    e = ex["extracted"]
    label = f"{ex['ticker']} {ex['filed_at']}"
    if e.get("aum_bn_usd"):
        kpi_rows.append({"Filing": label, "Metric": "AUM ($bn)", "Value": e["aum_bn_usd"]})
    if e.get("net_irr_pct"):
        kpi_rows.append({"Filing": label, "Metric": "Net IRR (%)", "Value": e["net_irr_pct"]})
    if e.get("tvpi"):
        kpi_rows.append({"Filing": label, "Metric": "TVPI (×10 for scale)", "Value": e["tvpi"] * 10})

if kpi_rows:
    kpi_df = pd.DataFrame(kpi_rows)
    fig_kpi = px.bar(
        kpi_df, x="Filing", y="Value", color="Metric",
        barmode="group",
        title="Qwen2.5-72B Extracted KPIs from Real SEC 8-K Filings",
        labels={"Value": "Value"},
        height=420, width=800,
        text="Value",
    )
    fig_kpi.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig_kpi.update_layout(
        margin=dict(t=55, b=60), xaxis_tickangle=-20,
        yaxis_type="log", yaxis_title="Value (log scale)",
    )
else:
    # Fallback: show entity + sentiment from real extractions
    entities = [ex["extracted"].get("fund_or_entity_name", "unknown") for ex in extractions]
    sentiments_map = {"positive": 1, "cautious": 0.5, "negative": -1, None: 0}
    svals = [sentiments_map.get(ex["extracted"].get("overall_sentiment"), 0) for ex in extractions]
    labels_list = [f"{ex['ticker']} {ex['filed_at']}" for ex in extractions]
    fig_kpi = px.bar(
        x=labels_list, y=svals,
        color=svals, color_continuous_scale=["#e74c3c", "#95a5a6", "#2ecc71"],
        title="Qwen2.5-72B: LLM Sentiment per Real SEC 8-K Filing",
        labels={"x": "Filing", "y": "Sentiment Score"},
        height=360, width=760,
        text=entities,
    )
    fig_kpi.update_traces(textposition="outside")
    fig_kpi.update_layout(margin=dict(t=55, b=60), xaxis_tickangle=-20)

write_image(fig_kpi, OUT / "edgar_kpi_extraction.png", scale=2)
print("Saved edgar_kpi_extraction.png")

print(f"\nAll screenshots saved to {OUT}")


