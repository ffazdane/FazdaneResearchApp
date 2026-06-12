"""
FazDane Analytics - Tier 2
Social Stock Stories
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from modules.base_module import FazDaneModule
from modules.tier2.stock_sentiment import score_sentiment, sentiment_label
from utils.universe_manager import format_ticker_display, get_ticker_names, render_universe_manager


DEFAULT_SUBREDDITS = ["stocks", "investing", "wallstreetbets", "options", "SecurityAnalysis"]
REDDIT_URL = "https://www.reddit.com/r/{subreddit}/{sort}.json"
STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
USER_AGENT = "FazDaneResearchApp/1.0"
COMMON_FALSE_TICKERS = {
    "A", "AI", "ALL", "ARE", "AT", "BE", "BIG", "BY", "CAN", "DD", "DO", "FOR",
    "GO", "HAS", "HE", "I", "IN", "IT", "LOW", "NEW", "NEXT", "NOW", "ON",
    "ONE", "OR", "OUT", "POST", "REAL", "RH", "SO", "TA", "THE", "TO", "TV",
    "UP", "USA", "VERY", "WE", "WELL", "YOLO",
}


def _empty_social_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Ticker", "Source", "Community", "Published", "Title", "Url", "Author",
            "Engagement", "Sentiment", "Sentiment Label", "Age Hours", "Social Score",
        ]
    )


def _http_get_json(url: str, params: dict | None = None) -> dict:
    response = requests.get(
        url,
        params=params,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=12,
    )
    response.raise_for_status()
    return response.json()


def _clean_tickers(tickers: list[str]) -> list[str]:
    cleaned = []
    for ticker in tickers:
        ticker = str(ticker).strip().upper()
        if ticker and ticker not in COMMON_FALSE_TICKERS and ticker not in cleaned:
            cleaned.append(ticker)
    return cleaned


def _extract_tickers(text: str, valid_tickers: set[str]) -> list[str]:
    candidates = set(re.findall(r"\$([A-Z]{1,5})(?:\b|[._-])", text.upper()))
    candidates.update(re.findall(r"\b[A-Z]{2,5}\b", text.upper()))
    return sorted(ticker for ticker in candidates if ticker in valid_tickers and ticker not in COMMON_FALSE_TICKERS)


def _published_from_epoch(epoch_value) -> pd.Timestamp:
    try:
        return pd.Timestamp(datetime.fromtimestamp(float(epoch_value), tz=timezone.utc)).tz_convert(None)
    except Exception:
        return pd.Timestamp.utcnow().tz_convert(None)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_reddit_social_posts(
    subreddits: tuple[str, ...],
    tickers: tuple[str, ...],
    sort: str,
    limit_per_subreddit: int,
    lookback_hours: int,
) -> pd.DataFrame:
    valid_tickers = set(_clean_tickers(list(tickers)))
    if not valid_tickers:
        return _empty_social_frame()

    cutoff = pd.Timestamp.utcnow().tz_convert(None) - pd.Timedelta(hours=lookback_hours)
    rows = []

    for subreddit in subreddits:
        subreddit = str(subreddit).strip()
        if not subreddit:
            continue

        try:
            payload = _http_get_json(
                REDDIT_URL.format(subreddit=quote(subreddit), sort=sort),
                params={"limit": int(limit_per_subreddit), "raw_json": 1},
            )
        except Exception:
            continue

        children = payload.get("data", {}).get("children", [])
        for child in children:
            post = child.get("data", {})
            title = str(post.get("title") or "")
            body = str(post.get("selftext") or "")
            text = f"{title} {body}"
            matched = _extract_tickers(text, valid_tickers)
            if not matched:
                continue

            published = _published_from_epoch(post.get("created_utc"))
            if published < cutoff:
                continue

            engagement = int(post.get("score") or 0) + int(post.get("num_comments") or 0) * 2
            sentiment = score_sentiment(text[:1000])
            url = post.get("url") or f"https://www.reddit.com{post.get('permalink', '')}"
            for ticker in matched:
                rows.append(
                    {
                        "Ticker": ticker,
                        "Source": "Reddit",
                        "Community": f"r/{subreddit}",
                        "Published": published,
                        "Title": title,
                        "Url": url,
                        "Author": post.get("author") or "",
                        "Engagement": engagement,
                        "Sentiment": sentiment,
                        "Sentiment Label": sentiment_label(sentiment),
                    }
                )

    return _score_social_rows(pd.DataFrame(rows))


@st.cache_data(ttl=600, show_spinner=False)
def fetch_stocktwits_social_posts(
    tickers: tuple[str, ...],
    limit_per_ticker: int,
    lookback_hours: int,
) -> pd.DataFrame:
    cutoff = pd.Timestamp.utcnow().tz_convert(None) - pd.Timedelta(hours=lookback_hours)
    rows = []

    for ticker in _clean_tickers(list(tickers)):
        try:
            payload = _http_get_json(STOCKTWITS_URL.format(ticker=quote(ticker)), params={"limit": int(limit_per_ticker)})
        except Exception:
            continue

        for message in payload.get("messages", []):
            body = str(message.get("body") or "")
            created = message.get("created_at")
            try:
                published = pd.Timestamp(created).tz_convert(None)
            except Exception:
                published = pd.Timestamp.utcnow().tz_convert(None)
            if published < cutoff:
                continue

            likes = message.get("likes") or {}
            engagement = int(likes.get("total") or 0)
            sentiment = score_sentiment(body)
            user = message.get("user") or {}
            rows.append(
                {
                    "Ticker": ticker,
                    "Source": "Stocktwits",
                    "Community": f"${ticker}",
                    "Published": published,
                    "Title": body,
                    "Url": f"https://stocktwits.com/message/{message.get('id')}",
                    "Author": user.get("username") or "",
                    "Engagement": engagement,
                    "Sentiment": sentiment,
                    "Sentiment Label": sentiment_label(sentiment),
                }
            )

    return _score_social_rows(pd.DataFrame(rows))


def _score_social_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_social_frame()

    df = df.copy()
    now = pd.Timestamp.utcnow().tz_convert(None)
    df["Published"] = pd.to_datetime(df["Published"], errors="coerce")
    df["Age Hours"] = ((now - df["Published"]).dt.total_seconds() / 3600).clip(lower=0.05)
    recency = 1 / (1 + df["Age Hours"] / 12)
    engagement = pd.to_numeric(df["Engagement"], errors="coerce").fillna(0)
    df["Social Score"] = (
        engagement.clip(lower=0).pow(0.55) * 8
        + recency * 25
        + pd.to_numeric(df["Sentiment"], errors="coerce").fillna(0).abs() * 10
    ).round(1)
    return df.sort_values("Social Score", ascending=False).reset_index(drop=True)


def scan_social_stories(
    tickers: tuple[str, ...],
    subreddits: tuple[str, ...],
    reddit_sort: str,
    reddit_limit: int,
    stocktwits_limit: int,
    lookback_hours: int,
    include_reddit: bool,
    include_stocktwits: bool,
) -> pd.DataFrame:
    frames = []
    if include_reddit:
        frames.append(fetch_reddit_social_posts(subreddits, tickers, reddit_sort, reddit_limit, lookback_hours))
    if include_stocktwits:
        frames.append(fetch_stocktwits_social_posts(tickers, stocktwits_limit, lookback_hours))

    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return _empty_social_frame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["Ticker", "Source", "Title"], keep="first")
    return _score_social_rows(combined)


def _sentiment_color(value: str) -> str:
    colors = {
        "Positive": "background-color: rgba(58, 181, 74, 0.22); color: #b7f7c1; font-weight: 700;",
        "Neutral": "background-color: rgba(245, 158, 11, 0.18); color: #fde68a; font-weight: 700;",
        "Negative": "background-color: rgba(239, 68, 68, 0.22); color: #fecaca; font-weight: 700;",
    }
    return colors.get(str(value), "")


def _score_color(value) -> str:
    try:
        score = float(value)
    except Exception:
        return ""
    if score >= 100:
        return "background-color: rgba(58, 181, 74, 0.18); color: #b7f7c1;"
    if score >= 45:
        return "background-color: rgba(245, 158, 11, 0.16); color: #fde68a;"
    return "color: #e2e8f0;"


class SocialStockStoriesModule(FazDaneModule):
    MODULE_NAME = "Social Stock Stories"
    MODULE_ICON = "SS"
    MODULE_DESCRIPTION = "Find stock-leading social stories from Reddit and Stocktwits"
    TIER = 2
    SOURCE_NOTEBOOK = "Native Streamlit module"
    CACHE_TTL = 600
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["Reddit public JSON", "Stocktwits public symbol streams"]

    def render_sidebar(self):
        st.markdown("**Universe**")
        self.universe_name, tickers, _ = render_universe_manager(
            key_prefix="social_stories",
            show_benchmark=False,
            label="Ticker Universe:",
        )
        self.ticker_names = get_ticker_names(self.universe_name)
        self.tickers = _clean_tickers([ticker for ticker in tickers if not str(ticker).startswith("^")])
        st.caption(f"{len(self.tickers)} tickers selected from {self.universe_name}.")

        st.markdown("**Sources**")
        self.include_reddit = st.checkbox("Reddit", value=True, key="social_include_reddit")
        self.include_stocktwits = st.checkbox("Stocktwits", value=True, key="social_include_stocktwits")

        st.markdown("**Reddit Feeds**")
        subreddit_text = st.text_area(
            "Subreddits",
            value=", ".join(DEFAULT_SUBREDDITS),
            key="social_subreddits",
            height=86,
        )
        self.subreddits = tuple(s.strip().lstrip("r/") for s in subreddit_text.split(",") if s.strip())
        self.reddit_sort = st.selectbox("Reddit Sort", ["hot", "new", "top", "rising"], index=0, key="social_reddit_sort")

        st.markdown("**Limits**")
        self.lookback_hours = int(st.slider("Lookback Hours", 1, 168, 48, 1, key="social_lookback"))
        self.scan_limit = int(st.slider("Ticker Scan Limit", 3, 50, min(20, max(len(self.tickers), 3)), 1, key="social_ticker_limit"))
        self.reddit_limit = int(st.slider("Reddit Posts/Subreddit", 10, 100, 50, 5, key="social_reddit_limit"))
        self.stocktwits_limit = int(st.slider("Stocktwits Posts/Ticker", 10, 30, 20, 5, key="social_stocktwits_limit"))
        self.run_scan = st.button("Scan Social Stories", type="primary", width="stretch", key="social_scan")

    def render_main(self):
        self.render_section_header(
            "Social Stock Stories",
            "Public social feeds filtered to selected stock tickers"
        )

        if not self.tickers:
            st.warning("Select a universe with equity tickers to scan social stories.")
            return

        scan_tickers = tuple(self.tickers[: self.scan_limit])
        st.caption(
            f"Scanning {len(scan_tickers)} tickers: "
            + ", ".join(format_ticker_display(ticker, self.ticker_names) for ticker in scan_tickers[:10])
            + (f" +{len(scan_tickers) - 10} more" if len(scan_tickers) > 10 else "")
        )

        if self.run_scan or "social_stories_results" not in st.session_state:
            with st.spinner("Scanning public social feeds..."):
                st.session_state["social_stories_results"] = scan_social_stories(
                    scan_tickers,
                    self.subreddits,
                    self.reddit_sort,
                    self.reddit_limit,
                    self.stocktwits_limit,
                    self.lookback_hours,
                    self.include_reddit,
                    self.include_stocktwits,
                )

        stories = st.session_state.get("social_stories_results", _empty_social_frame())
        if stories.empty:
            st.info("No ticker-linked social stories found for the current filters.")
            return

        self._render_metrics(stories)
        tab1, tab2, tab3, tab4 = st.tabs(["Top Stories", "Ticker Momentum", "Source Mix", "Raw Feed"])
        with tab1:
            self._tab_top_stories(stories)
        with tab2:
            self._tab_ticker_momentum(stories)
        with tab3:
            self._tab_source_mix(stories)
        with tab4:
            self._tab_raw_feed(stories)

    def _render_metrics(self, stories: pd.DataFrame):
        top_ticker = stories.groupby("Ticker")["Social Score"].sum().sort_values(ascending=False).index[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stories", f"{len(stories):,}")
        c2.metric("Tickers", f"{stories['Ticker'].nunique():,}")
        c3.metric("Top Ticker", top_ticker)
        c4.metric("Avg Sentiment", f"{stories['Sentiment'].mean():+.2f}")

    def _tab_top_stories(self, stories: pd.DataFrame):
        top = stories.head(50).copy()
        display_cols = ["Ticker", "Source", "Community", "Published", "Title", "Engagement", "Sentiment Label", "Social Score"]
        styled_top = (
            top[display_cols]
            .style
            .map(_sentiment_color, subset=["Sentiment Label"])
            .map(_score_color, subset=["Social Score"])
        )
        st.dataframe(
            styled_top,
            width="stretch",
            hide_index=True,
            column_config={
                "Published": st.column_config.DatetimeColumn("Published", format="YYYY-MM-DD HH:mm"),
                "Social Score": st.column_config.NumberColumn("Score", format="%.1f"),
                "Engagement": st.column_config.NumberColumn("Engagement", format="%d"),
            },
        )
        self._render_sentiment_charts(stories)

    def _render_sentiment_charts(self, stories: pd.DataFrame):
        st.markdown("### Sentiment Segmentation")
        sentiment_by_ticker = (
            stories.groupby(["Ticker", "Sentiment Label"], as_index=False)
            .agg(Stories=("Title", "count"), SocialScore=("Social Score", "sum"))
        )
        ticker_order = (
            stories.groupby("Ticker")["Social Score"]
            .sum()
            .sort_values(ascending=False)
            .head(20)
            .index
            .tolist()
        )
        sentiment_by_ticker = sentiment_by_ticker[sentiment_by_ticker["Ticker"].isin(ticker_order)]

        fig = px.bar(
            sentiment_by_ticker,
            x="Ticker",
            y="Stories",
            color="Sentiment Label",
            barmode="stack",
            category_orders={"Ticker": ticker_order, "Sentiment Label": ["Positive", "Neutral", "Negative"]},
            color_discrete_map={"Positive": "#3ab54a", "Neutral": "#f59e0b", "Negative": "#ef4444"},
            hover_data=["SocialScore"],
            labels={"Stories": "Story Count"},
        )
        fig.update_layout(
            paper_bgcolor="#0d1b2e",
            plot_bgcolor="#152847",
            font=dict(color="#e2e8f0", family="Inter"),
            xaxis=dict(gridcolor="#1e3a5f"),
            yaxis=dict(gridcolor="#1e3a5f"),
            legend=dict(
                bgcolor="rgba(21,40,71,0.85)",
                bordercolor="#1e3a5f",
                borderwidth=1,
            ),
            margin=dict(l=0, r=0, t=20, b=0),
            height=360,
        )
        st.plotly_chart(fig, width="stretch", key="social_sentiment_by_ticker")

        scatter_df = stories.head(250).copy()
        fig2 = px.scatter(
            scatter_df,
            x="Sentiment",
            y="Social Score",
            color="Sentiment Label",
            size="Engagement",
            size_max=28,
            hover_data=["Ticker", "Source", "Community", "Title"],
            color_discrete_map={"Positive": "#3ab54a", "Neutral": "#f59e0b", "Negative": "#ef4444"},
            labels={"Sentiment": "Sentiment Score", "Social Score": "Social Score"},
        )
        fig2.update_layout(
            paper_bgcolor="#0d1b2e",
            plot_bgcolor="#152847",
            font=dict(color="#e2e8f0", family="Inter"),
            xaxis=dict(gridcolor="#1e3a5f", zeroline=True, zerolinecolor="#94a3b8"),
            yaxis=dict(gridcolor="#1e3a5f"),
            legend=dict(
                bgcolor="rgba(21,40,71,0.85)",
                bordercolor="#1e3a5f",
                borderwidth=1,
            ),
            margin=dict(l=0, r=0, t=20, b=0),
            height=360,
        )
        st.plotly_chart(fig2, width="stretch", key="social_sentiment_score_scatter")

    def _tab_ticker_momentum(self, stories: pd.DataFrame):
        summary = (
            stories.groupby("Ticker", as_index=False)
            .agg(
                Stories=("Title", "count"),
                SocialScore=("Social Score", "sum"),
                AvgSentiment=("Sentiment", "mean"),
                Engagement=("Engagement", "sum"),
            )
            .sort_values("SocialScore", ascending=False)
            .head(25)
        )
        fig = px.bar(
            summary,
            x="Ticker",
            y="SocialScore",
            color="AvgSentiment",
            hover_data=["Stories", "Engagement", "AvgSentiment"],
            labels={"SocialScore": "Social Score", "AvgSentiment": "Avg Sentiment"},
            color_continuous_scale=["#ef4444", "#f59e0b", "#3ab54a"],
        )
        fig.update_layout(
            paper_bgcolor="#0d1b2e",
            plot_bgcolor="#152847",
            font=dict(color="#e2e8f0", family="Inter"),
            xaxis=dict(gridcolor="#1e3a5f"),
            yaxis=dict(gridcolor="#1e3a5f"),
            margin=dict(l=0, r=0, t=20, b=0),
            height=420,
        )
        st.plotly_chart(fig, width="stretch")
        styled_summary = (
            summary.round(2)
            .style
            .background_gradient(subset=["AvgSentiment"], cmap="RdYlGn", vmin=-1, vmax=1)
            .background_gradient(subset=["SocialScore"], cmap="Greens")
        )
        st.dataframe(styled_summary, width="stretch", hide_index=True)

    def _tab_source_mix(self, stories: pd.DataFrame):
        source = stories.groupby(["Source", "Sentiment Label"], as_index=False).size()
        fig = px.bar(
            source,
            x="Source",
            y="size",
            color="Sentiment Label",
            barmode="stack",
            color_discrete_map={"Positive": "#3ab54a", "Neutral": "#f59e0b", "Negative": "#ef4444"},
            labels={"size": "Stories"},
        )
        fig.update_layout(
            paper_bgcolor="#0d1b2e",
            plot_bgcolor="#152847",
            font=dict(color="#e2e8f0", family="Inter"),
            xaxis=dict(gridcolor="#1e3a5f"),
            yaxis=dict(gridcolor="#1e3a5f"),
            margin=dict(l=0, r=0, t=20, b=0),
            height=360,
        )
        st.plotly_chart(fig, width="stretch")

    def _tab_raw_feed(self, stories: pd.DataFrame):
        styled_feed = (
            stories.sort_values("Published", ascending=False)
            .style
            .map(_sentiment_color, subset=["Sentiment Label"])
            .map(_score_color, subset=["Social Score"])
        )
        st.dataframe(
            styled_feed,
            width="stretch",
            hide_index=True,
            column_config={
                "Url": st.column_config.LinkColumn("Url"),
                "Published": st.column_config.DatetimeColumn("Published", format="YYYY-MM-DD HH:mm"),
                "Sentiment": st.column_config.NumberColumn("Sentiment", format="%.2f"),
                "Age Hours": st.column_config.NumberColumn("Age Hrs", format="%.1f"),
                "Social Score": st.column_config.NumberColumn("Score", format="%.1f"),
            },
        )
