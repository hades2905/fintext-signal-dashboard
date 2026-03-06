"""
Financial News Sentiment Analyser
===================================
Enter a stock ticker to fetch recent news, run FinBERT sentiment analysis
and spaCy NER – all locally, no API key required.

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp.fetcher import fetch_news
from src.nlp.sentiment import SentimentAnalyser
from src.nlp.ner import annotate_articles
from src.nlp.schemas import Article, SentimentLabel

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial News Sentiment",
    page_icon="📰",
    layout="wide",
)

SENTIMENT_COLORS = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral":  "#95a5a6",
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    tickers_input = st.text_input(
        "Ticker(s)",
        value="MUV2.DE",
        help="One or more Yahoo Finance tickers, comma-separated. E.g. MUV2.DE, BLK, AXA.PA",
    )
    max_articles = st.slider("Max articles per ticker", 5, 30, 15)
    run_ner = st.checkbox("Extract named entities (spaCy)", value=True)
    st.divider()
    st.markdown(
        "**Models used**\n\n"
        "- Sentiment: [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert)\n"
        "- NER: spaCy `en_core_web_sm`\n"
        "- News: Yahoo Finance via yfinance\n\n"
        "All models run **locally** – no API key required."
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("📰 Financial News Sentiment Analyser")
st.caption(
    "FinBERT sentiment classification + Named Entity Recognition on live financial news."
)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

run_btn = st.button("🔍 Analyse", type="primary", disabled=not tickers)

# Cache analyser in session state so model is not reloaded on every interaction
if "analyser" not in st.session_state:
    st.session_state["analyser"] = None

if run_btn:
    all_articles: list[Article] = []

    with st.spinner("Loading FinBERT model (first run downloads ~450 MB)…"):
        if st.session_state["analyser"] is None:
            st.session_state["analyser"] = SentimentAnalyser()

    analyser: SentimentAnalyser = st.session_state["analyser"]

    for ticker in tickers:
        with st.spinner(f"Fetching news for {ticker}…"):
            articles = fetch_news(ticker, max_articles=max_articles)
        if not articles:
            st.warning(f"No news found for {ticker}.")
            continue

        with st.spinner(f"Running FinBERT on {len(articles)} articles for {ticker}…"):
            analyser.score_articles(articles)

        if run_ner:
            with st.spinner(f"Extracting entities for {ticker}…"):
                annotate_articles(articles)

        all_articles.extend(articles)

    if all_articles:
        st.session_state["articles"] = all_articles
        st.success(f"✅ Analysed {len(all_articles)} articles across {len(tickers)} ticker(s).")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
articles: list[Article] | None = st.session_state.get("articles")

if articles:
    # -----------------------------------------------------------------------
    # Build flat DataFrame
    # -----------------------------------------------------------------------
    rows = []
    for a in articles:
        if a.sentiment is None:
            continue
        rows.append({
            "ticker":      a.ticker,
            "title":       a.title,
            "source":      a.source or "—",
            "published":   a.published_at.date() if a.published_at else None,
            "label":       a.sentiment.label,
            "positive":    round(a.sentiment.positive, 4),
            "negative":    round(a.sentiment.negative, 4),
            "neutral":     round(a.sentiment.neutral, 4),
            "url":         a.url or "",
            "entities":    ", ".join(
                f"{e.text} [{e.label}]" for e in a.entities
            ) if a.entities else "—",
        })
    df = pd.DataFrame(rows)

    st.divider()

    # -----------------------------------------------------------------------
    # KPI row
    # -----------------------------------------------------------------------
    total = len(df)
    n_pos = (df["label"] == SentimentLabel.POSITIVE).sum()
    n_neg = (df["label"] == SentimentLabel.NEGATIVE).sum()
    n_neu = (df["label"] == SentimentLabel.NEUTRAL).sum()
    avg_pos = df["positive"].mean()
    avg_neg = df["negative"].mean()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Articles", total)
    k2.metric("Positive 🟢", n_pos, f"{n_pos/total:.0%}")
    k3.metric("Negative 🔴", n_neg, f"{n_neg/total:.0%}")
    k4.metric("Neutral ⚪", n_neu, f"{n_neu/total:.0%}")
    k5.metric("Avg Positive Score", f"{avg_pos:.2f}")

    st.divider()

    left, right = st.columns(2)

    # -----------------------------------------------------------------------
    # Sentiment distribution pie
    # -----------------------------------------------------------------------
    with left:
        st.subheader("Sentiment Distribution")
        pie_df = df["label"].value_counts().reset_index()
        pie_df.columns = ["label", "count"]
        pie_df["color"] = pie_df["label"].map(SENTIMENT_COLORS)
        fig_pie = px.pie(
            pie_df,
            names="label",
            values="count",
            color="label",
            color_discrete_map=SENTIMENT_COLORS,
            hole=0.4,
        )
        fig_pie.update_layout(margin=dict(t=10, b=10), height=320)
        st.plotly_chart(fig_pie, use_container_width=True)

    # -----------------------------------------------------------------------
    # Sentiment over time
    # -----------------------------------------------------------------------
    with right:
        st.subheader("Sentiment Over Time")
        df_time = df.dropna(subset=["published"])
        if not df_time.empty:
            time_df = (
                df_time.groupby(["published", "label"])
                .size()
                .reset_index(name="count")
            )
            fig_time = px.bar(
                time_df,
                x="published",
                y="count",
                color="label",
                color_discrete_map=SENTIMENT_COLORS,
                barmode="stack",
            )
            fig_time.update_layout(
                xaxis_title="Date",
                yaxis_title="Articles",
                height=320,
                margin=dict(t=10, b=10),
                legend_title="Sentiment",
            )
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("No publication dates available for timeline.")

    # -----------------------------------------------------------------------
    # Per-ticker sentiment (if multiple tickers)
    # -----------------------------------------------------------------------
    if len(tickers) > 1:
        st.subheader("Sentiment by Ticker")
        ticker_df = (
            df.groupby(["ticker", "label"])
            .size()
            .reset_index(name="count")
        )
        fig_tick = px.bar(
            ticker_df,
            x="ticker",
            y="count",
            color="label",
            color_discrete_map=SENTIMENT_COLORS,
            barmode="group",
        )
        fig_tick.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig_tick, use_container_width=True)

    # -----------------------------------------------------------------------
    # Score distribution per label
    # -----------------------------------------------------------------------
    st.subheader("Score Distribution by Article")
    fig_box = go.Figure()
    for label, color in SENTIMENT_COLORS.items():
        subset = df[df["label"] == label]
        score_col = label  # column name matches label
        if not subset.empty and score_col in subset.columns:
            fig_box.add_trace(go.Box(
                y=subset[score_col],
                name=label.capitalize(),
                marker_color=color,
                boxmean=True,
            ))
    fig_box.update_layout(
        yaxis_title="Confidence Score",
        height=300,
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # -----------------------------------------------------------------------
    # Top entities
    # -----------------------------------------------------------------------
    if run_ner:
        st.divider()
        st.subheader("Top Named Entities")
        all_entities = []
        for a in articles:
            for e in a.entities:
                all_entities.append((e.text, e.label))

        if all_entities:
            ent_df = pd.DataFrame(all_entities, columns=["Entity", "Type"])
            top_ents = (
                ent_df.groupby(["Entity", "Type"])
                .size()
                .reset_index(name="Mentions")
                .sort_values("Mentions", ascending=False)
                .head(20)
            )
            col_ent, col_type = st.columns([2, 1])
            with col_ent:
                fig_ent = px.bar(
                    top_ents,
                    x="Mentions",
                    y="Entity",
                    color="Type",
                    orientation="h",
                    height=420,
                )
                fig_ent.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    margin=dict(t=10, b=10),
                )
                st.plotly_chart(fig_ent, use_container_width=True)

            with col_type:
                type_counts = ent_df["Type"].value_counts().reset_index()
                type_counts.columns = ["Type", "Count"]
                st.dataframe(type_counts, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # Article table
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("All Articles")

    sentiment_filter = st.multiselect(
        "Filter by sentiment",
        options=["positive", "negative", "neutral"],
        default=["positive", "negative", "neutral"],
    )
    display_df = df[df["label"].isin(sentiment_filter)].copy()

    # Make title a clickable link where URL exists
    def make_link(row: pd.Series) -> str:
        if row["url"]:
            return f'<a href="{row["url"]}" target="_blank">{row["title"]}</a>'
        return row["title"]

    st.dataframe(
        display_df[["ticker", "published", "source", "label", "positive", "negative", "neutral", "entities", "title"]]
        .rename(columns={
            "ticker": "Ticker",
            "published": "Date",
            "source": "Source",
            "label": "Sentiment",
            "positive": "P(pos)",
            "negative": "P(neg)",
            "neutral": "P(neu)",
            "entities": "Entities",
            "title": "Title",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------------------------------------------------
    # CSV download
    # -----------------------------------------------------------------------
    st.download_button(
        "⬇️ Download CSV",
        data=display_df.to_csv(index=False),
        file_name="sentiment_results.csv",
        mime="text/csv",
    )
