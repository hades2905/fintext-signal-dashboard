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
aum_rows = []
for ex in extractions:
    e = ex["extracted"]
    if e.get("aum_bn_usd"):
        entity = e.get("fund_or_entity_name") or ex["ticker"]
        aum_rows.append({
            "Entity": f"{entity}\n({ex['ticker']} · {ex['filed_at']})",
            "AUM ($bn)": e["aum_bn_usd"],
        })

if aum_rows:
    aum_df = pd.DataFrame(aum_rows).sort_values("AUM ($bn)")
    fig_kpi = go.Figure(go.Bar(
        x=aum_df["AUM ($bn)"],
        y=aum_df["Entity"],
        orientation="h",
        marker_color=["#4a90d9", "#2ecc71"],
        text=[f"${v:,.0f} bn" for v in aum_df["AUM ($bn)"]],
        textposition="outside",
        cliponaxis=False,
    ))
    fig_kpi.update_layout(
        title="Qwen2.5-72B — AUM Extracted from Real SEC 8-K Filings",
        xaxis_title="AUM ($ bn)",
        height=300, width=800,
        margin=dict(t=55, b=30, l=220, r=120),
        xaxis=dict(range=[0, max(aum_df["AUM ($bn)"]) * 1.25]),
    )
else:
    # Fallback: show sentiment signal per filing
    sentiments_map = {"positive": 1, "cautious": 0.5, "negative": -1, None: 0}
    labels_list, svals, entities = [], [], []
    for ex in extractions:
        labels_list.append(f"{ex['ticker']} {ex['filed_at']}")
        svals.append(sentiments_map.get(ex["extracted"].get("overall_sentiment"), 0))
        entities.append(ex["extracted"].get("fund_or_entity_name", ex["ticker"]))
    colors = ["#2ecc71" if s > 0 else "#e74c3c" if s < 0 else "#95a5a6" for s in svals]
    fig_kpi = go.Figure(go.Bar(
        x=labels_list, y=svals, marker_color=colors,
        text=entities, textposition="outside",
    ))
    fig_kpi.update_layout(
        title="Qwen2.5-72B: Sentiment Signal per SEC 8-K Filing",
        yaxis_title="Sentiment (−1 neg · 0 neutral · +1 pos)",
        height=380, width=800, margin=dict(t=55, b=80),
        xaxis_tickangle=-20,
    )

write_image(fig_kpi, OUT / "edgar_kpi_extraction.png", scale=2)
print("Saved edgar_kpi_extraction.png")

print(f"\nAll screenshots saved to {OUT}")


