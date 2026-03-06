"""
Generate sample dashboard screenshots from mock data and save to docs/screenshots/.
Run:  python scripts/generate_screenshots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# make src importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import write_image

OUT = Path(__file__).parent.parent / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

SENTIMENT_COLORS = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral":  "#95a5a6",
}

random.seed(42)

# -------------------------------------------------------------------
# Mock dataset  (30 articles, realistic PE distribution)
# -------------------------------------------------------------------
labels   = random.choices(["positive", "negative", "neutral"], weights=[12, 8, 10], k=30)
today    = date(2026, 3, 5)
dates    = [today - timedelta(days=random.randint(0, 13)) for _ in range(30)]
tickers  = random.choices(["BX", "KKR", "APO", "ARES", "CG"], k=30)
sources  = random.choices(["Reuters", "Bloomberg", "FT", "WSJ", "CNBC"], k=30)


def fake_scores(label: str) -> dict:
    if label == "positive":
        pos = random.uniform(0.65, 0.97)
        neg = random.uniform(0.01, 0.15)
    elif label == "negative":
        neg = random.uniform(0.65, 0.97)
        pos = random.uniform(0.01, 0.15)
    else:
        pos = neg = random.uniform(0.10, 0.30)
    neu = max(0.0, 1.0 - pos - neg)
    return {"positive": round(pos, 4), "negative": round(neg, 4), "neutral": round(neu, 4)}


rows = []
for i, (lbl, d, tkr, src) in enumerate(zip(labels, dates, tickers, sources)):
    sc = fake_scores(lbl)
    rows.append({"ticker": tkr, "published": d, "source": src, "label": lbl, **sc})

df = pd.DataFrame(rows)

# -------------------------------------------------------------------
# 1. Sentiment distribution pie
# -------------------------------------------------------------------
pie_df = df["label"].value_counts().reset_index()
pie_df.columns = ["label", "count"]
fig_pie = px.pie(
    pie_df, names="label", values="count", color="label",
    color_discrete_map=SENTIMENT_COLORS, hole=0.4,
    title="Sentiment Distribution",
)
fig_pie.update_layout(margin=dict(t=50, b=10), height=380, width=560)
write_image(fig_pie, OUT / "sentiment_pie.png", scale=2)
print("Saved sentiment_pie.png")

# -------------------------------------------------------------------
# 2. Sentiment over time (stacked bar)
# -------------------------------------------------------------------
time_df = df.groupby(["published", "label"]).size().reset_index(name="count")
fig_time = px.bar(
    time_df, x="published", y="count", color="label",
    color_discrete_map=SENTIMENT_COLORS, barmode="stack",
    title="Sentiment Over Time",
    labels={"published": "Date", "count": "Articles"},
)
fig_time.update_layout(height=380, width=760, margin=dict(t=50, b=10))
write_image(fig_time, OUT / "sentiment_timeline.png", scale=2)
print("Saved sentiment_timeline.png")

# -------------------------------------------------------------------
# 3. Sentiment by ticker (grouped bar)
# -------------------------------------------------------------------
ticker_df = df.groupby(["ticker", "label"]).size().reset_index(name="count")
fig_tick = px.bar(
    ticker_df, x="ticker", y="count", color="label",
    color_discrete_map=SENTIMENT_COLORS, barmode="group",
    title="Sentiment by Ticker",
    labels={"ticker": "Ticker", "count": "Articles"},
)
fig_tick.update_layout(height=380, width=560, margin=dict(t=50, b=10))
write_image(fig_tick, OUT / "sentiment_by_ticker.png", scale=2)
print("Saved sentiment_by_ticker.png")

# -------------------------------------------------------------------
# 4. Score distribution box plot
# -------------------------------------------------------------------
fig_box = go.Figure()
for label, color in SENTIMENT_COLORS.items():
    subset = df[df["label"] == label]
    fig_box.add_trace(go.Box(y=subset[label], name=label.capitalize(), marker_color=color, boxmean=True))
fig_box.update_layout(
    title="Confidence Score Distribution",
    yaxis_title="Confidence Score",
    height=360, width=560, margin=dict(t=50, b=10),
)
write_image(fig_box, OUT / "score_distribution.png", scale=2)
print("Saved score_distribution.png")

# -------------------------------------------------------------------
# 5. Top named entities (horizontal bar)
# -------------------------------------------------------------------
entities_raw = [
    ("Blackstone", "ORG"), ("Blackstone", "ORG"), ("Blackstone", "ORG"),
    ("KKR", "ORG"), ("KKR", "ORG"), ("KKR", "ORG"), ("KKR", "ORG"),
    ("Apollo", "ORG"), ("Apollo", "ORG"),
    ("Steve Schwarzman", "PERSON"), ("Steve Schwarzman", "PERSON"),
    ("United States", "GPE"), ("United States", "GPE"),
    ("Europe", "GPE"), ("Europe", "GPE"), ("Europe", "GPE"),
    ("NYSE", "ORG"), ("NYSE", "ORG"),
    ("S&P 500", "PRODUCT"),
    ("Federal Reserve", "ORG"), ("Federal Reserve", "ORG"),
    ("ECB", "ORG"), ("ECB", "ORG"),
    ("Jon Gray", "PERSON"),
    ("London", "GPE"), ("London", "GPE"),
    ("Goldman Sachs", "ORG"),
]
ent_df = pd.DataFrame(entities_raw, columns=["Entity", "Type"])
top_ents = (
    ent_df.groupby(["Entity", "Type"]).size()
    .reset_index(name="Mentions")
    .sort_values("Mentions", ascending=False)
    .head(15)
)
fig_ent = px.bar(
    top_ents, x="Mentions", y="Entity", color="Type",
    orientation="h", title="Top Named Entities", height=440, width=660,
)
fig_ent.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(t=50, b=10))
write_image(fig_ent, OUT / "top_entities.png", scale=2)
print("Saved top_entities.png")

# -------------------------------------------------------------------
# 6. Portfolio heatmap (NEW)
# -------------------------------------------------------------------
portfolio_tickers = ["BX", "KKR", "APO", "ARES", "CG"]
random.seed(99)
pos_vals  = [random.uniform(30, 70) for _ in portfolio_tickers]
neg_vals  = [random.uniform(10, 40) for _ in portfolio_tickers]
neu_vals  = [max(0, 100 - p - n) for p, n in zip(pos_vals, neg_vals)]

fig_heat = go.Figure(data=go.Heatmap(
    z=[pos_vals, neg_vals, neu_vals],
    x=portfolio_tickers,
    y=["Positive %", "Negative %", "Neutral %"],
    colorscale=[[0.0, "#f8f9fa"], [0.4, "#fff3cd"], [1.0, "#2ecc71"]],
    text=[[f"{v:.0f}%" for v in pos_vals],
          [f"{v:.0f}%" for v in neg_vals],
          [f"{v:.0f}%" for v in neu_vals]],
    texttemplate="%{text}",
    showscale=True,
))
fig_heat.update_layout(
    title="Portfolio Sentiment Heatmap",
    height=320, width=760, margin=dict(t=50, b=10),
)
write_image(fig_heat, OUT / "portfolio_heatmap.png", scale=2)
print("Saved portfolio_heatmap.png")

# -------------------------------------------------------------------
# 7. LLM Structured Extraction – mock KPI summary bar chart (NEW)
# -------------------------------------------------------------------
funds = {
    "BREP X":         {"net_irr": 19.2, "tvpi": 1.65, "dpi": 0.82},
    "KKR Infra IV":   {"net_irr": 16.4, "tvpi": 1.48, "dpi": 0.61},
    "Apollo Fund X":  {"net_irr": 21.1, "tvpi": 1.72, "dpi": 1.10},
    "Ares Credit V":  {"net_irr": 12.8, "tvpi": 1.31, "dpi": 0.95},
    "Carlyle VII":    {"net_irr": 17.5, "tvpi": 1.58, "dpi": 0.74},
}
irr_data = pd.DataFrame([
    {"Fund": k, "Metric": "Net IRR (%)", "Value": v["net_irr"]} for k, v in funds.items()
] + [
    {"Fund": k, "Metric": "TVPI (x)", "Value": v["tvpi"] * 10} for k, v in funds.items()
] + [
    {"Fund": k, "Metric": "DPI (x)", "Value": v["dpi"] * 10} for k, v in funds.items()
])
fig_kpi = px.bar(
    irr_data, x="Fund", y="Value", color="Metric",
    barmode="group", title="LLM-Extracted Fund KPIs (IRR · TVPI · DPI)",
    labels={"Value": "Value (IRR in %, multiples ×10 for scale)"},
    height=400, width=760,
)
fig_kpi.update_layout(margin=dict(t=50, b=10))
write_image(fig_kpi, OUT / "edgar_kpi_extraction.png", scale=2)
print("Saved edgar_kpi_extraction.png")

print(f"\nAll screenshots saved to {OUT}")

