"""
FazDane Analytics - Tier 2
Stock Sentiment Analysis
"""

from datetime import datetime, timedelta
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup

from modules.base_module import FazDaneModule
from utils.universe_manager import format_ticker_display, get_ticker_names, render_universe_manager


FINVIZ_URL = "https://finviz.com/quote.ashx?t={ticker}"

POSITIVE_TERMS = {
    "beat", "beats", "surge", "surges", "rally", "rallies", "upgrade", "upgraded",
    "bullish", "growth", "profit", "profits", "strong", "record", "raises", "raised",
    "outperform", "buy", "gain", "gains", "higher", "optimistic", "winner", "wins",
}
NEGATIVE_TERMS = {
    "miss", "misses", "drop", "drops", "plunge", "plunges", "downgrade", "downgraded",
    "bearish", "loss", "losses", "weak", "probe", "lawsuit", "cut", "cuts", "lower",
    "warning", "concern", "concerns", "risk", "risks", "sell", "falls", "slump",
}


@st.cache_resource(show_spinner=False)
def get_vader():
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        try:
            return SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            return SentimentIntensityAnalyzer()
    except Exception:
        return None


def fallback_sentiment(text: str) -> float:
    words = {word.strip(".,;:!?()[]{}'\"").lower() for word in text.split()}
    pos = len(words & POSITIVE_TERMS)
    neg = len(words & NEGATIVE_TERMS)
    if pos == 0 and neg == 0:
        return 0.0
    return max(min((pos - neg) / max(pos + neg, 1), 1.0), -1.0)


def score_sentiment(text: str) -> float:
    vader = get_vader()
    if vader is not None:
        try:
            return float(vader.polarity_scores(text)["compound"])
        except Exception:
            pass
    return fallback_sentiment(text)


def sentiment_label(score: float) -> str:
    if score > 0.05:
        return "Positive"
    if score < -0.05:
        return "Negative"
    return "Neutral"


def impact_label(score: float, price_change: float | None = None) -> str:
    magnitude = abs(score)
    price_mag = abs(price_change) if pd.notna(price_change) else 0
    if magnitude >= 0.55 or price_mag >= 3:
        return "High"
    if magnitude >= 0.25 or price_mag >= 1.25:
        return "Medium"
    return "Low"


def parse_finviz_timestamp(raw_timestamp: str, current_date) -> tuple[pd.Timestamp | None, str]:
    stamp = raw_timestamp.strip()
    parts = stamp.split()
    if not parts:
        return None, ""

    if len(parts) == 1:
        return pd.Timestamp(current_date), parts[0]

    date_part, time_part = parts[0], parts[1]
    if date_part.lower() == "today":
        return pd.Timestamp(current_date), time_part
    if date_part.lower() == "yesterday":
        return pd.Timestamp(current_date - timedelta(days=1)), time_part

    try:
        return pd.Timestamp(datetime.strptime(date_part, "%b-%d-%y").date()), time_part
    except Exception:
        return None, time_part


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_finviz_news(ticker: str, days: int, max_headlines: int) -> pd.DataFrame:
    ticker = ticker.strip().upper()
    today = datetime.now().date()
    cutoff = today - timedelta(days=days)
    url = FINVIZ_URL.format(ticker=ticker)
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        html = urlopen(Request(url, headers=headers), timeout=12).read()
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Date", "Time", "Headline", "Sentiment", "Label", "Source", "Url"])

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="fullview-news-outer")
    rows = []
    if not table:
        return pd.DataFrame(columns=["Ticker", "Date", "Time", "Headline", "Sentiment", "Label", "Source", "Url"])

    for row in table.find_all("tr"):
        if len(rows) >= max_headlines:
            break
        link = row.find("a")
        cell = row.find("td")
        if not link or not cell:
            continue
        date, time_text = parse_finviz_timestamp(cell.get_text(strip=True), today)
        if date is None or date.date() < cutoff:
            continue
        headline = link.get_text(strip=True)
        href = link.get("href", "")
        source = urlparse(href).netloc.replace("www.", "") if href else "finviz"
        score = score_sentiment(headline)
        rows.append(
            {
                "Ticker": ticker,
                "Date": date.normalize(),
                "Time": time_text,
                "Headline": headline,
                "Sentiment": score,
                "Label": sentiment_label(score),
                "Source": source or "finviz",
                "Url": href,
            }
        )

    return pd.DataFrame(rows)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_yahoo_news(ticker: str, days: int, max_headlines: int) -> pd.DataFrame:
    ticker = ticker.strip().upper()
    cutoff = pd.Timestamp(datetime.now() - timedelta(days=days), tz="UTC")
    rows = []

    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        items = []

    for item in items[:max_headlines]:
        content = item.get("content", item) if isinstance(item, dict) else {}
        if not isinstance(content, dict):
            continue

        headline = content.get("title") or item.get("title", "")
        if not headline:
            continue

        raw_date = content.get("pubDate") or content.get("displayTime") or item.get("providerPublishTime")
        try:
            if isinstance(raw_date, (int, float)):
                published = pd.Timestamp(raw_date, unit="s", tz="UTC")
            else:
                published = pd.Timestamp(raw_date).tz_convert("UTC")
        except Exception:
            published = pd.Timestamp(datetime.now(), tz="UTC")

        if published < cutoff:
            continue

        provider = content.get("provider", {}) if isinstance(content.get("provider", {}), dict) else {}
        canonical = content.get("canonicalUrl", {}) if isinstance(content.get("canonicalUrl", {}), dict) else {}
        click = content.get("clickThroughUrl", {}) if isinstance(content.get("clickThroughUrl", {}), dict) else {}
        url = canonical.get("url") or click.get("url") or ""
        score = score_sentiment(headline)
        local_date = published.tz_convert(None).normalize()
        rows.append(
            {
                "Ticker": ticker,
                "Date": local_date,
                "Time": published.tz_convert(None).strftime("%H:%M"),
                "Headline": headline,
                "Sentiment": score,
                "Label": sentiment_label(score),
                "Source": provider.get("displayName") or "Yahoo Finance",
                "Url": url,
                "Feed": "Yahoo Finance",
            }
        )

    return pd.DataFrame(rows)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_stock_news(ticker: str, days: int, max_headlines: int) -> pd.DataFrame:
    finviz = fetch_finviz_news(ticker, days, max_headlines)
    if not finviz.empty:
        finviz = finviz.copy()
        finviz["Feed"] = "Finviz"
        return finviz
    return fetch_yahoo_news(ticker, days, max_headlines)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_price_performance(ticker: str, days: int) -> pd.DataFrame:
    ticker = ticker.strip().upper()
    data = yf.download(ticker, period=f"{days + 8}d", auto_adjust=True, progress=False)
    if data.empty:
        return pd.DataFrame(columns=["Date", "Close", "Pct_Change"])

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ["_".join(str(part) for part in col if part).strip() for col in data.columns]
    close_cols = [col for col in data.columns if "Close" in str(col)]
    if not close_cols:
        return pd.DataFrame(columns=["Date", "Close", "Pct_Change"])

    df = data[[close_cols[0]]].copy()
    df.columns = ["Close"]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df["Pct_Change"] = df["Close"].pct_change() * 100
    df = df.reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    return df[["Date", "Close", "Pct_Change"]].dropna(subset=["Pct_Change"])


def merge_news_price(news: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    if news.empty:
        return news
    news = news.copy()
    news["Date"] = pd.to_datetime(news["Date"], errors="coerce").dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")
    if perf.empty:
        merged = news.copy()
        merged["PriceDate"] = pd.NaT
        merged["Pct_Change"] = np.nan
        merged["Close"] = np.nan
        merged["PriceMatch"] = "No price data"
    else:
        news_sorted = news.sort_values("Date").copy()
        perf_sorted = perf[["Date", "Pct_Change", "Close"]].sort_values("Date").copy()
        perf_sorted["Date"] = pd.to_datetime(perf_sorted["Date"], errors="coerce").dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")
        news_sorted = news_sorted.dropna(subset=["Date"])
        perf_sorted = perf_sorted.dropna(subset=["Date"])
        merged = pd.merge_asof(
            news_sorted,
            perf_sorted.rename(columns={"Date": "PriceDate"}),
            left_on="Date",
            right_on="PriceDate",
            direction="forward",
            tolerance=pd.Timedelta(days=4),
        )
        merged["PriceMatch"] = np.where(merged["PriceDate"].notna(), "Next trading day", "No price data")
        missing = merged["PriceDate"].isna()
        if missing.any():
            missing_dates = merged.loc[missing, ["Date"]].copy()
            missing_news = news_sorted[news_sorted["Date"].isin(missing_dates["Date"])].sort_values("Date")
            backward = pd.merge_asof(
                missing_news,
                perf_sorted.rename(columns={"Date": "PriceDate"}),
                left_on="Date",
                right_on="PriceDate",
                direction="backward",
                tolerance=pd.Timedelta(days=4),
            )
            for column in ["PriceDate", "Pct_Change", "Close"]:
                merged.loc[missing, column] = backward[column].values
            merged.loc[missing & merged["PriceDate"].notna(), "PriceMatch"] = "Previous trading day"
    merged["AbsSentiment"] = merged["Sentiment"].abs()
    merged["Impact"] = merged.apply(lambda row: impact_label(row["Sentiment"], row.get("Pct_Change")), axis=1)
    merged["DirectionMatch"] = np.where(
        merged["Pct_Change"].isna(),
        "No price data",
        np.where(
            ((merged["Sentiment"] > 0.05) & (merged["Pct_Change"] > 0))
            | ((merged["Sentiment"] < -0.05) & (merged["Pct_Change"] < 0)),
            "Aligned",
            "Diverged",
        ),
    )
    return merged


def daily_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    if news.empty:
        return pd.DataFrame(columns=["Date", "Avg_Sentiment", "News_Count", "Positive", "Negative", "Neutral"])
    stats = news.groupby("Date").agg(
        Avg_Sentiment=("Sentiment", "mean"),
        News_Count=("Sentiment", "count"),
        Positive=("Label", lambda x: (x == "Positive").sum()),
        Negative=("Label", lambda x: (x == "Negative").sum()),
        Neutral=("Label", lambda x: (x == "Neutral").sum()),
    ).reset_index()
    return stats.sort_values("Date")


def sentiment_scorecard(news: pd.DataFrame, merged: pd.DataFrame) -> dict:
    if news.empty:
        return {"score": 0.0, "label": "No News", "count": 0, "high": 0, "alignment": 0.0}
    score = news["Sentiment"].mean()
    high_count = (merged["Impact"] == "High").sum() if not merged.empty else 0
    priced = merged[merged["DirectionMatch"].isin(["Aligned", "Diverged"])] if not merged.empty else pd.DataFrame()
    alignment = (priced["DirectionMatch"] == "Aligned").mean() * 100 if not priced.empty else 0
    return {
        "score": score,
        "label": sentiment_label(score),
        "count": len(news),
        "high": int(high_count),
        "alignment": alignment,
    }


def plot_daily_sentiment(daily: pd.DataFrame, ticker: str):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=daily["Date"],
            y=daily["Avg_Sentiment"],
            name="Avg Sentiment",
            marker_color=np.where(daily["Avg_Sentiment"] >= 0, "#22c55e", "#ef4444"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily["Date"],
            y=daily["News_Count"],
            name="News Count",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="#facc15", width=2),
        )
    )
    fig.update_layout(
        title=f"Daily Sentiment & News Volume - {ticker}",
        template="plotly_dark",
        paper_bgcolor="#0d1b2e",
        plot_bgcolor="#0d1b2e",
        height=440,
        yaxis=dict(title="Avg Sentiment", range=[-1, 1]),
        yaxis2=dict(title="News Count", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h"),
    )
    return fig


def plot_sentiment_vs_price(merged: pd.DataFrame, ticker: str):
    chart = merged.dropna(subset=["Pct_Change"]).copy()
    if chart.empty:
        return None
    chart["BubbleSize"] = chart["AbsSentiment"] * 30 + 8
    fig = px.scatter(
        chart,
        x="Date",
        y="Pct_Change",
        size="BubbleSize",
        color="Label",
        color_discrete_map={"Positive": "#22c55e", "Negative": "#ef4444", "Neutral": "#facc15"},
        symbol="Impact",
        hover_data=["Headline", "Sentiment", "Pct_Change", "Date", "PriceDate", "PriceMatch", "Source", "Impact", "DirectionMatch"],
        title=f"{ticker} News Sentiment vs Daily Price Move",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e", height=500)
    return fig


def scan_universe(tickers: list[str], days: int, max_headlines: int) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        news = fetch_stock_news(ticker, days, max_headlines)
        perf = fetch_price_performance(ticker, days)
        merged = merge_news_price(news, perf)
        card = sentiment_scorecard(news, merged)
        latest_move = perf["Pct_Change"].dropna().iloc[-1] if not perf.empty and perf["Pct_Change"].notna().any() else np.nan
        rows.append(
            {
                "Ticker": ticker,
                "Sentiment Score": card["score"],
                "Sentiment": card["label"],
                "Headlines": card["count"],
                "Feed": news["Feed"].iloc[0] if not news.empty and "Feed" in news.columns else "No News",
                "High Impact": card["high"],
                "Price Alignment %": card["alignment"],
                "Latest Daily %": latest_move,
            }
        )
    return pd.DataFrame(rows).sort_values(["Sentiment Score", "Headlines"], ascending=[False, False])


class StockSentimentModule(FazDaneModule):
    MODULE_NAME = "Stock Sentiment Analysis"
    MODULE_ICON = "📰"
    MODULE_DESCRIPTION = "Headline-level news sentiment, price reaction, and universe sentiment screening"
    TIER = 2
    SOURCE_NOTEBOOK = "Finviz + VADER News Sentiment"
    CACHE_TTL = 1800
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["Finviz", "yfinance", "VADER"]

    def render_sidebar(self):
        st.markdown("**Sentiment Universe**")
        self.universe_name, tickers, _ = render_universe_manager(
            key_prefix="sentiment",
            show_benchmark=False,
            label="Ticker Universe:",
        )
        ticker_names = get_ticker_names(self.universe_name)
        self.tickers = [ticker for ticker in tickers if not ticker.startswith("^") and "=" not in ticker]

        if self.tickers:
            default_index = self.tickers.index("NVDA") if "NVDA" in self.tickers else 0
            if st.session_state.get("sentiment_ticker") not in self.tickers:
                st.session_state["sentiment_ticker"] = self.tickers[default_index]
            self.ticker = st.selectbox(
                "Select Ticker:",
                self.tickers,
                index=default_index,
                key="sentiment_ticker",
                format_func=lambda ticker: format_ticker_display(ticker, ticker_names),
            )
        else:
            self.ticker = ""

        self.days = int(st.slider("News Lookback Days:", 3, 30, 7, key="sentiment_days"))
        self.max_headlines = int(st.slider("Max Headlines:", 10, 100, 50, step=5, key="sentiment_max_headlines"))
        self.scan_limit = int(st.slider("Universe Scan Limit:", 3, 30, min(12, max(len(self.tickers), 3)), key="sentiment_scan_limit"))
        self.show_urls = st.checkbox("Show headline URLs", value=False, key="sentiment_urls")

        if st.button("Refresh Sentiment", width="stretch", type="primary", key="sentiment_refresh"):
            fetch_finviz_news.clear()
            fetch_yahoo_news.clear()
            fetch_stock_news.clear()
            fetch_price_performance.clear()
            st.rerun()

    def render_main(self):
        self.render_section_header(
            "📰 Stock Sentiment Analysis",
            "Finviz headline sentiment, price reaction, and universe-level sentiment screening",
        )

        if not self.ticker:
            st.warning("Select a universe with equity tickers. Index and futures symbols are filtered out for news sentiment.")
            return

        with st.spinner(f"Fetching Finviz headlines and price data for {self.ticker}..."):
            news = fetch_stock_news(self.ticker, self.days, self.max_headlines)
            perf = fetch_price_performance(self.ticker, self.days)
            merged = merge_news_price(news, perf)

        if news.empty:
            st.warning(f"No headlines found for {self.ticker} in the last {self.days} days.")
            return

        daily = daily_sentiment(news)
        card = sentiment_scorecard(news, merged)
        avg_price = merged["Pct_Change"].mean() if not merged.empty and merged["Pct_Change"].notna().any() else np.nan

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ticker", self.ticker)
        c2.metric("Sentiment", card["label"], f"{card['score']:.2f}")
        c3.metric("Headlines", str(card["count"]), f"{card['high']} high impact")
        feed_name = news["Feed"].iloc[0] if "Feed" in news.columns and not news.empty else "News"
        c4.metric("Feed", feed_name, "Avg move N/A" if pd.isna(avg_price) else f"{avg_price:.2f}% avg move")

        tab_overview, tab_impact, tab_headlines, tab_universe = st.tabs(
            ["Combined Charts", "Impact Summary", "Headlines", "Universe Scan"]
        )

        with tab_overview:
            st.plotly_chart(
                plot_daily_sentiment(daily, self.ticker),
                width="stretch",
                key=f"sentiment_daily_{self.ticker}_{self.days}",
            )
            fig_impact = plot_sentiment_vs_price(merged, self.ticker)
            if fig_impact is None:
                st.info("Price data did not align with recent headline dates.")
            else:
                st.plotly_chart(
                    fig_impact,
                    width="stretch",
                    key=f"sentiment_bubble_overview_{self.ticker}_{self.days}",
                )
            left, right = st.columns([1, 1])
            with left:
                mix = news["Label"].value_counts().rename_axis("Sentiment").reset_index(name="Count")
                fig_mix = px.pie(
                    mix,
                    names="Sentiment",
                    values="Count",
                    color="Sentiment",
                    color_discrete_map={"Positive": "#22c55e", "Negative": "#ef4444", "Neutral": "#facc15"},
                    title="Headline Sentiment Mix",
                )
                fig_mix.update_layout(template="plotly_dark", paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e", height=360)
                st.plotly_chart(
                    fig_mix,
                    width="stretch",
                    key=f"sentiment_mix_{self.ticker}_{self.days}",
                )
            with right:
                label_daily = daily[["Date", "Avg_Sentiment", "News_Count", "Positive", "Negative", "Neutral"]].copy()
                st.dataframe(label_daily.round(3), width="stretch", hide_index=True)

        with tab_impact:
            fig_impact = plot_sentiment_vs_price(merged, self.ticker)
            if fig_impact is None:
                st.info("Price data did not align with recent headline dates.")
            else:
                st.plotly_chart(
                    fig_impact,
                    width="stretch",
                    key=f"sentiment_bubble_impact_{self.ticker}_{self.days}",
                )
            st.dataframe(daily.round(3), width="stretch", hide_index=True)
            impact_summary = merged.groupby(["Impact", "Label"], as_index=False).agg(
                Headlines=("Headline", "count"),
                Avg_Sentiment=("Sentiment", "mean"),
                Avg_Price_Move=("Pct_Change", "mean"),
            )
            st.dataframe(impact_summary.round(3), width="stretch", hide_index=True)

        with tab_headlines:
            display_cols = ["Date", "Time", "Ticker", "Headline", "Sentiment", "Label", "Impact", "Feed", "PriceDate", "PriceMatch", "Pct_Change", "DirectionMatch", "Source"]
            if self.show_urls:
                display_cols.append("Url")
            table = merged[display_cols].sort_values(["Date", "Time"], ascending=[False, False])
            st.dataframe(table.round({"Sentiment": 3, "Pct_Change": 2}), width="stretch", hide_index=True)
            st.download_button(
                "Download Headlines CSV",
                data=table.to_csv(index=False),
                file_name=f"sentiment_{self.ticker.lower()}_{self.days}d.csv",
                mime="text/csv",
                width="stretch",
            )

        with tab_universe:
            scan_tickers = self.tickers[: self.scan_limit]
            if st.button("Run Universe Sentiment Scan", width="stretch", key="sentiment_scan_button"):
                with st.spinner(f"Scanning {len(scan_tickers)} tickers from {self.universe_name}..."):
                    scan = scan_universe(scan_tickers, self.days, min(self.max_headlines, 30))
                fig_scan = px.bar(
                    scan,
                    x="Ticker",
                    y="Sentiment Score",
                    color="Sentiment",
                    color_discrete_map={"Positive": "#22c55e", "Negative": "#ef4444", "Neutral": "#facc15", "No News": "#64748b"},
                    hover_data=["Headlines", "High Impact", "Price Alignment %", "Latest Daily %"],
                    title="Universe Sentiment Score",
                )
                fig_scan.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
                fig_scan.update_layout(template="plotly_dark", paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e", height=460)
                st.plotly_chart(
                    fig_scan,
                    width="stretch",
                    key=f"sentiment_scan_{self.universe_name}_{self.days}_{self.scan_limit}",
                )
                st.dataframe(scan.round(2), width="stretch", hide_index=True)
            else:
                st.info("Click the scan button to rank selected universe tickers by recent news sentiment.")
