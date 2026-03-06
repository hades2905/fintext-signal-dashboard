"""
Financial Intelligence Dashboard
==================================
Three-in-one analysis platform for alternative asset intelligence:

  Tab 1 – News Sentiment:   FinBERT sentiment on live news (yfinance)
  Tab 2 – EDGAR Filings:    SEC 8-K / 10-K structured extraction via LLM
  Tab 3 – Portfolio Monitor: Sentiment heatmap across a portfolio of tickers

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp.edgar import KNOWN_CIKS, fetch_filings
from src.nlp.extractor import LLMExtractor
from src.nlp.fetcher import fetch_news
from src.nlp.ner import annotate_articles
from src.nlp.schemas import Article, EdgarFiling, SentimentLabel
from src.nlp.sentiment import SentimentAnalyser

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
)

SENTIMENT_COLORS = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral":  "#95a5a6",
}

PE_TICKERS = list(KNOWN_CIKS.keys())  # BX, KKR, APO, ARES, CG, BAM

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    hf_token = st.text_input(
        "HuggingFace Token",
        type="password",
        value=os.environ.get("HF_TOKEN", ""),
        help="Free token from https://huggingface.co/settings/tokens",
    )
    st.divider()
    st.markdown(
        "**Models**\n\n"
        "- Sentiment: [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert)\n"
        "- Extraction: [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)\n"
        "- NER: spaCy `en_core_web_sm`\n"
        "- Filings: SEC EDGAR (no key)\n"
        "- News: Yahoo Finance (no key)\n\n"
        "All inference via HuggingFace API."
    )
    st.divider()
    st.caption("Munich Re GIM · Alternative Assets Research · 2026")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _require_token() -> bool:
    if not hf_token:
        st.warning("⚠️ Please enter your HuggingFace token in the sidebar.")
        return False
    return True


@st.cache_resource(show_spinner=False)
def _get_analyser(token: str) -> SentimentAnalyser:
    return SentimentAnalyser(api_key=token)


@st.cache_resource(show_spinner=False)
def _get_extractor(token: str) -> LLMExtractor:
    return LLMExtractor(api_key=token)


def _build_sentiment_df(articles: list[Article]) -> pd.DataFrame:
    rows = []
    for a in articles:
        if a.sentiment is None:
            continue
        rows.append({
            "ticker":    a.ticker,
            "title":     a.title,
            "source":    a.source or "—",
            "published": a.published_at.date() if a.published_at else None,
            "label":     a.sentiment.label,
            "positive":  round(a.sentiment.positive, 4),
            "negative":  round(a.sentiment.negative, 4),
            "neutral":   round(a.sentiment.neutral, 4),
            "url":       a.url or "",
            "entities":  ", ".join(f"{e.text} [{e.label}]" for e in a.entities) if a.entities else "—",
        })
    return pd.DataFrame(rows)


# ===========================================================================
# Tab layout
# ===========================================================================
tab1, tab2, tab3 = st.tabs([
    "📰 News Sentiment",
    "📂 SEC EDGAR Filings",
    "🏦 Portfolio Monitor",
])


# ===========================================================================
# TAB 1 – News Sentiment
# ===========================================================================
with tab1:
    st.header("News Sentiment Analysis")
    st.caption("FinBERT classification + NER on live financial news via Yahoo Finance.")

    col_a, col_b, col_c = st.columns([3, 1, 1])
    with col_a:
        tickers_input = st.text_input(
            "Ticker(s) — comma-separated",
            value="BX, KKR",
            help="Yahoo Finance tickers: BX, KKR, APO, ARES, MUV2.DE …",
            key="news_tickers",
        )
    with col_b:
        max_articles = st.slider("Max articles / ticker", 5, 30, 15, key="news_max")
    with col_c:
        run_ner = st.checkbox("NER (spaCy)", value=True, key="news_ner")
        generate_summary = st.checkbox("LLM Briefing", value=True, key="news_summary")

    run_news = st.button("🔍 Analyse News", type="primary", key="btn_news")

    if run_news:
        if not _require_token():
            st.stop()

        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        analyser = _get_analyser(hf_token)
        extractor = _get_extractor(hf_token)
        all_articles: list[Article] = []

        for ticker in tickers:
            with st.spinner(f"Fetching news for {ticker}…"):
                articles = fetch_news(ticker, max_articles=max_articles)
            if not articles:
                st.warning(f"No news found for {ticker}.")
                continue
            with st.spinner(f"Running FinBERT on {len(articles)} articles for {ticker}…"):
                analyser.score_articles(articles)
            if run_ner:
                with st.spinner(f"Running NER for {ticker}…"):
                    annotate_articles(articles)
            all_articles.extend(articles)

        if all_articles:
            st.session_state["news_articles"] = all_articles
            st.session_state["news_tickers_list"] = tickers
            st.success(f"✅ Analysed {len(all_articles)} articles across {len(tickers)} ticker(s).")

    articles_result: list[Article] | None = st.session_state.get("news_articles")
    if articles_result:
        df = _build_sentiment_df(articles_result)
        tickers_used: list[str] = st.session_state.get("news_tickers_list", [])

        if df.empty:
            st.info("No sentiment results yet.")
        else:
            total = len(df)
            n_pos = (df["label"] == SentimentLabel.POSITIVE).sum()
            n_neg = (df["label"] == SentimentLabel.NEGATIVE).sum()
            n_neu = (df["label"] == SentimentLabel.NEUTRAL).sum()

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Articles", total)
            k2.metric("Positive 🟢", n_pos, f"{n_pos/total:.0%}")
            k3.metric("Negative 🔴", n_neg, f"{n_neg/total:.0%}")
            k4.metric("Neutral ⚪", n_neu, f"{n_neu/total:.0%}")
            k5.metric("Avg Positive Score", f"{df['positive'].mean():.2f}")

            st.divider()

            if generate_summary and hf_token:
                extractor = _get_extractor(hf_token)
                for ticker in tickers_used:
                    t_articles = [a for a in articles_result if a.ticker == ticker]
                    if not t_articles:
                        continue
                    with st.spinner(f"Generating investment briefing for {ticker}…"):
                        summary = extractor.investment_summary(ticker, t_articles)
                    st.info(f"**📋 Investment Briefing — {ticker}**\n\n{summary}")

            st.divider()
            c1, c2 = st.columns(2)

            with c1:
                st.subheader("Sentiment Distribution")
                pie_df = df["label"].value_counts().reset_index()
                pie_df.columns = ["label", "count"]
                fig_pie = px.pie(
                    pie_df, names="label", values="count", color="label",
                    color_discrete_map=SENTIMENT_COLORS, hole=0.4,
                )
                fig_pie.update_layout(margin=dict(t=10, b=10), height=300)
                st.plotly_chart(fig_pie, use_container_width=True)

            with c2:
                st.subheader("Sentiment Over Time")
                df_time = df.dropna(subset=["published"])
                if not df_time.empty:
                    tdf = (
                        df_time.groupby(["published", "label"])
                        .size()
                        .reset_index(name="count")
                    )
                    fig_t = px.bar(
                        tdf, x="published", y="count", color="label",
                        color_discrete_map=SENTIMENT_COLORS, barmode="stack",
                        labels={"published": "Date", "count": "Articles"},
                    )
                    fig_t.update_layout(height=300, margin=dict(t=10, b=10))
                    st.plotly_chart(fig_t, use_container_width=True)
                else:
                    st.info("No date information available.")

            if len(tickers_used) > 1:
                st.subheader("Sentiment by Ticker")
                tkr_df = (
                    df.groupby(["ticker", "label"])
                    .size()
                    .reset_index(name="count")
                )
                fig_tkr = px.bar(
                    tkr_df, x="ticker", y="count", color="label",
                    color_discrete_map=SENTIMENT_COLORS, barmode="group",
                )
                fig_tkr.update_layout(height=280, margin=dict(t=10, b=10))
                st.plotly_chart(fig_tkr, use_container_width=True)

            st.subheader("Confidence Score Distribution")
            fig_box = go.Figure()
            for label, color in SENTIMENT_COLORS.items():
                sub = df[df["label"] == label]
                if not sub.empty:
                    fig_box.add_trace(go.Box(
                        y=sub[label], name=label.capitalize(),
                        marker_color=color, boxmean=True,
                    ))
            fig_box.update_layout(yaxis_title="Score", height=280, margin=dict(t=10, b=10))
            st.plotly_chart(fig_box, use_container_width=True)

            if run_ner:
                all_ents = [(e.text, e.label) for a in articles_result for e in a.entities]
                if all_ents:
                    st.divider()
                    st.subheader("Top Named Entities")
                    ent_df = pd.DataFrame(all_ents, columns=["Entity", "Type"])
                    top = (
                        ent_df.groupby(["Entity", "Type"]).size()
                        .reset_index(name="Mentions")
                        .sort_values("Mentions", ascending=False)
                        .head(20)
                    )
                    col_e, col_t = st.columns([2, 1])
                    with col_e:
                        fig_ent = px.bar(
                            top, x="Mentions", y="Entity", color="Type",
                            orientation="h", height=440,
                        )
                        fig_ent.update_layout(
                            yaxis={"categoryorder": "total ascending"},
                            margin=dict(t=10, b=10),
                        )
                        st.plotly_chart(fig_ent, use_container_width=True)
                    with col_t:
                        type_counts = ent_df["Type"].value_counts().reset_index()
                        type_counts.columns = ["Type", "Count"]
                        st.dataframe(type_counts, use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("All Articles")
            filt = st.multiselect(
                "Filter by sentiment",
                ["positive", "negative", "neutral"],
                default=["positive", "negative", "neutral"],
                key="news_filter",
            )
            disp = df[df["label"].isin(filt)].rename(columns={
                "ticker": "Ticker", "published": "Date", "source": "Source",
                "label": "Sentiment", "positive": "P(pos)", "negative": "P(neg)",
                "neutral": "P(neu)", "entities": "Entities", "title": "Title",
            })
            st.dataframe(
                disp[["Ticker", "Date", "Source", "Sentiment",
                      "P(pos)", "P(neg)", "P(neu)", "Entities", "Title"]],
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "⬇️ Download CSV",
                data=disp.to_csv(index=False),
                file_name="sentiment_results.csv",
                mime="text/csv",
            )


# ===========================================================================
# TAB 2 – EDGAR Filings
# ===========================================================================
with tab2:
    st.header("SEC EDGAR Filing Analysis")
    st.caption(
        "Fetch real 8-K / 10-K filings from SEC EDGAR and extract structured "
        "investment data (IRR, AUM, TVPI, DPI, Risks…) using a Mistral-7B LLM."
    )

    col_e1, col_e2, col_e3 = st.columns([2, 1, 1])
    with col_e1:
        edgar_ticker = st.selectbox(
            "Ticker",
            options=PE_TICKERS,
            index=0,
            help="Major alternative asset managers with known CIKs on EDGAR",
            key="edgar_ticker",
        )
    with col_e2:
        edgar_form = st.selectbox("Form type", ["8-K", "10-K", "10-Q"], key="edgar_form")
    with col_e3:
        edgar_max = st.slider("Max filings", 1, 10, 3, key="edgar_max")

    run_edgar = st.button("📂 Fetch & Analyse Filings", type="primary", key="btn_edgar")

    if run_edgar:
        if not _require_token():
            st.stop()

        extractor = _get_extractor(hf_token)
        analyser = _get_analyser(hf_token)

        with st.spinner(f"Fetching {edgar_form} filings for {edgar_ticker} from SEC EDGAR…"):
            filings = fetch_filings(edgar_ticker, form_type=edgar_form, max_filings=edgar_max)

        if not filings:
            st.error(
                "No filings found. EDGAR may be temporarily unavailable, "
                "or the CIK could not be resolved."
            )
        else:
            st.success(
                f"✅ Retrieved {len(filings)} {edgar_form} filing(s) for {edgar_ticker}."
            )
            for i, filing in enumerate(filings):
                with st.spinner(f"Running LLM extraction on filing {i + 1}/{len(filings)}…"):
                    extractor.extract_filing(filing)
                # Sentiment on filing snippet
                try:
                    from src.nlp.schemas import Article as _A
                    tmp = _A(
                        ticker=filing.ticker,
                        title=filing.accession_number,
                        text=filing.text[:1500],
                    )
                    analyser.score_articles([tmp])
                    filing.sentiment = tmp.sentiment
                except Exception:
                    pass

            st.session_state["edgar_filings"] = filings

    filings_result: list[EdgarFiling] | None = st.session_state.get("edgar_filings")
    if filings_result:
        for filing in filings_result:
            ext = filing.extracted
            label_str = (
                f" | Sentiment: {filing.sentiment.label.value.upper()}"
                if filing.sentiment else ""
            )
            with st.expander(
                f"📄 {filing.form_type} — {filing.company_name} | "
                f"Filed: {filing.filed_at.date() if filing.filed_at else 'n/a'}"
                f"{label_str}",
                expanded=True,
            ):
                if ext is None:
                    st.warning("No extraction result available.")
                    continue

                cols = st.columns(5)
                cols[0].metric("Entity / Fund", ext.fund_or_entity_name or "—")
                cols[1].metric("Strategy", ext.strategy or "—")
                cols[2].metric("AUM ($bn)", f"{ext.aum_bn_usd:.1f}" if ext.aum_bn_usd else "—")
                cols[3].metric("Net IRR", f"{ext.net_irr_pct:.1f}%" if ext.net_irr_pct else "—")
                cols[4].metric("TVPI", f"{ext.tvpi:.2f}x" if ext.tvpi else "—")

                cols2 = st.columns(5)
                cols2[0].metric("DPI", f"{ext.dpi:.2f}x" if ext.dpi else "—")
                cols2[1].metric("RVPI", f"{ext.rvpi:.2f}x" if ext.rvpi else "—")
                cols2[2].metric("Vintage", ext.vintage_year or "—")
                cols2[3].metric("Deployment", ext.deployment_pace or "—")
                cols2[4].metric("Exit Env.", ext.exit_environment or "—")

                if filing.sentiment:
                    lbl = filing.sentiment.label.value
                    icon = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(lbl, "")
                    st.markdown(
                        f"**Filing Sentiment:** {icon} `{lbl.upper()}` — "
                        f"pos {filing.sentiment.positive:.0%} / "
                        f"neg {filing.sentiment.negative:.0%} / "
                        f"neu {filing.sentiment.neutral:.0%}"
                    )

                if ext.investment_summary:
                    st.info(f"**💡 Investment Summary**\n\n{ext.investment_summary}")

                r1, r2 = st.columns(2)
                with r1:
                    if ext.key_risks:
                        st.markdown("**⚠️ Key Risks**")
                        for risk in ext.key_risks:
                            st.markdown(f"- {risk}")
                with r2:
                    if ext.key_opportunities:
                        st.markdown("**✅ Key Opportunities**")
                        for opp in ext.key_opportunities:
                            st.markdown(f"- {opp}")

                with st.expander("📝 Raw Filing Text (first 3 000 chars)"):
                    st.text(filing.text[:3000])

                if filing.url:
                    st.markdown(f"[🔗 View on SEC EDGAR]({filing.url})")

        rows_dl = []
        for f in filings_result:
            e = f.extracted
            if e:
                rows_dl.append({
                    "ticker": f.ticker, "company": f.company_name,
                    "form": f.form_type,
                    "filed_at": f.filed_at.date() if f.filed_at else None,
                    "fund_or_entity": e.fund_or_entity_name,
                    "strategy": e.strategy, "geography": e.geography,
                    "aum_bn_usd": e.aum_bn_usd, "net_irr_pct": e.net_irr_pct,
                    "tvpi": e.tvpi, "dpi": e.dpi,
                    "overall_sentiment": e.overall_sentiment,
                    "key_risks": "; ".join(e.key_risks),
                    "investment_summary": e.investment_summary,
                })
        if rows_dl:
            st.divider()
            st.download_button(
                "⬇️ Download Structured Extracts (CSV)",
                data=pd.DataFrame(rows_dl).to_csv(index=False),
                file_name="edgar_extracts.csv",
                mime="text/csv",
            )


# ===========================================================================
# TAB 3 – Portfolio Monitor
# ===========================================================================
with tab3:
    st.header("Portfolio Sentiment Monitor")
    st.caption(
        "Batch-analyse a portfolio of tickers. "
        "Identify which holdings show deteriorating or improving news momentum at a glance."
    )

    portfolio_input = st.text_input(
        "Portfolio tickers — comma-separated",
        value="BX, KKR, APO, ARES, CG",
        key="portfolio_tickers",
    )
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        portfolio_max = st.slider("Articles per ticker", 5, 20, 10, key="portfolio_max")
    with col_p2:
        portfolio_ner = st.checkbox("Run NER", value=False, key="portfolio_ner")
        portfolio_briefings = st.checkbox("LLM Briefings per ticker", value=True, key="portfolio_brief")

    run_portfolio = st.button("🏦 Run Portfolio Scan", type="primary", key="btn_portfolio")

    if run_portfolio:
        if not _require_token():
            st.stop()

        tickers = [t.strip().upper() for t in portfolio_input.split(",") if t.strip()]
        analyser = _get_analyser(hf_token)
        portfolio_data: dict[str, list[Article]] = {}
        progress = st.progress(0.0, text="Starting portfolio scan…")

        for i, ticker in enumerate(tickers):
            progress.progress(i / len(tickers), text=f"Analysing {ticker}…")
            articles = fetch_news(ticker, max_articles=portfolio_max)
            if articles:
                analyser.score_articles(articles)
                if portfolio_ner:
                    annotate_articles(articles)
            portfolio_data[ticker] = articles

        progress.progress(1.0, text="Done.")
        st.session_state["portfolio_data"] = portfolio_data
        st.session_state["portfolio_tickers"] = tickers
        st.success(f"✅ Scanned {len(tickers)} tickers.")

    portfolio_data_res: dict[str, list[Article]] | None = st.session_state.get("portfolio_data")
    if portfolio_data_res:
        tickers_used = st.session_state.get("portfolio_tickers", list(portfolio_data_res.keys()))

        heatmap_rows = []
        for ticker, articles in portfolio_data_res.items():
            scored = [a for a in articles if a.sentiment]
            if not scored:
                heatmap_rows.append({
                    "Ticker": ticker, "Articles": 0,
                    "Positive%": 0.0, "Negative%": 0.0, "Neutral%": 0.0,
                    "Avg Pos Score": 0.0, "Avg Neg Score": 0.0,
                    "Signal": "no data",
                })
                continue
            n = len(scored)
            pos_pct = sum(1 for a in scored if a.sentiment.label == SentimentLabel.POSITIVE) / n
            neg_pct = sum(1 for a in scored if a.sentiment.label == SentimentLabel.NEGATIVE) / n
            neu_pct = 1.0 - pos_pct - neg_pct
            avg_pos = sum(a.sentiment.positive for a in scored) / n
            avg_neg = sum(a.sentiment.negative for a in scored) / n
            signal = "🟢 Positive" if pos_pct >= 0.5 else ("🔴 Watch" if neg_pct >= 0.4 else "⚪ Neutral")
            heatmap_rows.append({
                "Ticker": ticker, "Articles": n,
                "Positive%": round(pos_pct * 100, 1),
                "Negative%": round(neg_pct * 100, 1),
                "Neutral%": round(neu_pct * 100, 1),
                "Avg Pos Score": round(avg_pos, 3),
                "Avg Neg Score": round(avg_neg, 3),
                "Signal": signal,
            })

        heatmap_df = pd.DataFrame(heatmap_rows)
        st.divider()
        st.subheader("Portfolio Sentiment Heatmap")

        fig_heat = go.Figure(data=go.Heatmap(
            z=[
                heatmap_df["Positive%"].tolist(),
                heatmap_df["Negative%"].tolist(),
                heatmap_df["Neutral%"].tolist(),
            ],
            x=heatmap_df["Ticker"].tolist(),
            y=["Positive %", "Negative %", "Neutral %"],
            colorscale=[[0.0, "#f8f9fa"], [0.4, "#fff3cd"], [1.0, "#2ecc71"]],
            text=[
                [f"{v:.0f}%" for v in heatmap_df["Positive%"]],
                [f"{v:.0f}%" for v in heatmap_df["Negative%"]],
                [f"{v:.0f}%" for v in heatmap_df["Neutral%"]],
            ],
            texttemplate="%{text}",
            showscale=True,
        ))
        fig_heat.update_layout(height=260, margin=dict(t=10, b=10))
        st.plotly_chart(fig_heat, use_container_width=True)

        st.subheader("Portfolio Summary Table")
        st.dataframe(
            heatmap_df.style
                .background_gradient(subset=["Positive%"], cmap="Greens")
                .background_gradient(subset=["Negative%"], cmap="Reds"),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Sentiment Breakdown by Ticker")
        melt = heatmap_df.melt(
            id_vars="Ticker",
            value_vars=["Positive%", "Negative%", "Neutral%"],
            var_name="Sentiment",
            value_name="Percentage",
        )
        melt["Sentiment"] = melt["Sentiment"].str.replace("%", "").str.lower()
        fig_bar = px.bar(
            melt, x="Ticker", y="Percentage", color="Sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            barmode="stack",
            labels={"Percentage": "%"},
        )
        fig_bar.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

        if portfolio_briefings and hf_token:
            st.divider()
            st.subheader("💡 AI Investment Briefings")
            extractor = _get_extractor(hf_token)
            for ticker in tickers_used:
                arts = portfolio_data_res.get(ticker, [])
                if arts:
                    with st.spinner(f"Generating briefing for {ticker}…"):
                        summary = extractor.investment_summary(ticker, arts)
                    st.info(f"**{ticker}** — {summary}")

        st.divider()
        st.download_button(
            "⬇️ Download Portfolio Report (CSV)",
            data=heatmap_df.to_csv(index=False),
            file_name="portfolio_sentiment.csv",
            mime="text/csv",
        )

