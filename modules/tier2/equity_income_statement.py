"""
FazDane Analytics - Tier 2
Equity Income Statement
"""

from matplotlib.path import Path
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from modules.base_module import FazDaneModule
from utils.universe_manager import format_ticker_display, get_ticker_names, render_universe_manager


GREEN = (0.18, 0.72, 0.36)
RED = (0.86, 0.24, 0.30)
BLACK = (0, 0, 0)
TITLE_CLR = (0.05, 0.29, 0.44)
GRAY = (0.45, 0.45, 0.45)
GOLD = (0.85, 0.65, 0.10)

SOURCE_COLORS = [
    (0.22, 0.55, 0.82),
    (0.12, 0.62, 0.47),
    (0.85, 0.52, 0.10),
    (0.72, 0.28, 0.62),
    (0.35, 0.70, 0.82),
    (0.90, 0.75, 0.15),
    (0.60, 0.38, 0.78),
    (0.88, 0.38, 0.35),
]

ROW_ALIASES = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "cost": ["Cost Of Revenue", "Reconciled Cost Of Revenue"],
    "gross": ["Gross Profit"],
    "opex": ["Operating Expense", "Total Operating Expenses"],
    "total_expenses": ["Total Expenses"],
    "operating": ["Operating Income", "Operating Income Loss"],
    "tax": ["Tax Provision", "Income Tax Expense", "Income Tax Paid Supplemental Data"],
    "pretax": ["Pretax Income", "Income Before Tax", "Income Before Tax From Continuing Operations"],
    "other": ["Other Income Expense", "Other Non Operating Income Expenses"],
    "net": ["Net Income", "Net Income Common Stockholders"],
    "rd": ["Research And Development"],
    "sga": ["Selling General And Administration", "Selling And Marketing Expense", "General And Administrative Expense"],
}


def _financials(ticker: str) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    for attr in ["income_stmt", "financials"]:
        try:
            data = getattr(tk, attr)
            if data is not None and not data.empty:
                return data
        except Exception:
            continue
    return pd.DataFrame()


def _val(financials: pd.DataFrame, row_names: str | list[str], col_idx: int = 0) -> float:
    names = [row_names] if isinstance(row_names, str) else row_names
    for row_name in names:
        try:
            if row_name in financials.index:
                value = financials.loc[row_name].iloc[col_idx]
                if pd.notna(value):
                    return float(value) / 1e9
        except Exception:
            continue
    return 0.0


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_income_statement_payload(ticker: str, periods: int) -> dict:
    ticker = ticker.strip().upper()
    financials = _financials(ticker)
    if financials.empty:
        return {"ticker": ticker, "error": "No annual income statement data returned by yfinance."}

    revenue = _val(financials, ROW_ALIASES["revenue"])
    cost = abs(_val(financials, ROW_ALIASES["cost"]))
    gross = _val(financials, ROW_ALIASES["gross"])
    opex = abs(_val(financials, ROW_ALIASES["opex"]))
    total_expenses = abs(_val(financials, ROW_ALIASES["total_expenses"]))
    if opex == 0 and total_expenses and cost:
        opex = max(total_expenses - cost, 0.0)
    operating = _val(financials, ROW_ALIASES["operating"])
    tax = abs(_val(financials, ROW_ALIASES["tax"]))
    pretax = _val(financials, ROW_ALIASES["pretax"])
    net = _val(financials, ROW_ALIASES["net"])
    other = _val(financials, ROW_ALIASES["other"])
    if other == 0 and pretax:
        other = pretax - operating

    def row(name: str) -> list[float]:
        vals = []
        for idx in range(min(periods, len(financials.columns))):
            vals.append(_val(financials, ROW_ALIASES.get(name, name), idx))
        while len(vals) < periods:
            vals.append(0.0)
        return vals

    years = []
    for col in financials.columns[:periods]:
        try:
            years.append(str(pd.Timestamp(col).year))
        except Exception:
            years.append(str(col)[:4])
    while len(years) < periods:
        years.append("")

    sources, source_label = get_sources(ticker)
    seg_bar, seg_label = get_segments_for_bar(ticker, financials)
    revenues = row("revenue")
    grosses = row("gross")
    operatings = row("operating")
    nets = row("net")

    return {
        "ticker": ticker,
        "error": "",
        "latest_year": years[0] if years else "",
        "revenue": revenue,
        "cost": cost,
        "gross": gross,
        "opex": opex,
        "operating": operating,
        "other": other,
        "tax": tax,
        "pretax": pretax,
        "net": net,
        "years": years,
        "revenues": revenues,
        "grosses": grosses,
        "operatings": operatings,
        "nets": nets,
        "sources": sources,
        "source_label": source_label,
        "segment_bar": seg_bar,
        "segment_label": seg_label,
        "raw_rows": financials,
    }


def get_sources(ticker: str) -> tuple[list[tuple[str, float]], str]:
    tk = yf.Ticker(ticker)

    for attr in ["revenue_by_geography", "get_revenue_by_geography"]:
        try:
            geo = getattr(tk, attr)
            if callable(geo):
                geo = geo()
            if geo is not None and not geo.empty:
                latest = geo.iloc[:, 0].dropna()
                items = [(str(i), float(v) / 1e9) for i, v in latest.items() if float(v) > 0]
                if items:
                    return sorted(items, key=lambda x: x[1], reverse=True), "Revenue by Geography"
        except Exception:
            pass

    for attr in ["revenue_by_product", "get_revenue_by_product"]:
        try:
            seg = getattr(tk, attr)
            if callable(seg):
                seg = seg()
            if seg is not None and not seg.empty:
                latest = seg.iloc[:, 0].dropna()
                items = [(str(i), float(v) / 1e9) for i, v in latest.items() if float(v) > 0]
                if items:
                    return sorted(items, key=lambda x: x[1], reverse=True), "Revenue by Segment"
        except Exception:
            pass

    keywords = [
        "americas", "europe", "china", "japan", "asia", "pacific", "domestic",
        "international", "united states", "rest of world", "iphone", "mac", "ipad",
        "wearables", "services", "products", "cloud", "advertising", "gaming",
        "search", "youtube", "hardware", "software", "subscription", "data center",
        "automotive", "energy", "insurance",
    ]
    for stmt_attr in ["income_stmt", "financials", "quarterly_financials"]:
        try:
            financials = getattr(tk, stmt_attr)
            if financials is None or financials.empty:
                continue
            found = []
            seen = set()
            for idx in financials.index:
                idx_lower = str(idx).lower()
                if idx in seen or not any(keyword in idx_lower for keyword in keywords):
                    continue
                vals = financials.loc[idx].iloc[:4]
                value = float(vals.sum()) if stmt_attr == "quarterly_financials" else float(vals.iloc[0])
                if value > 1e8:
                    found.append((str(idx), value / 1e9))
                    seen.add(idx)
            if found:
                label = "Revenue by Segment (TTM)" if stmt_attr == "quarterly_financials" else "Revenue by Segment"
                return sorted(found, key=lambda x: x[1], reverse=True), label
        except Exception:
            pass

    return [], "No Geographic / Segment Data"


def get_segments_for_bar(ticker: str, financials: pd.DataFrame) -> tuple[list[tuple[str, float]], str]:
    sources, label = get_sources(ticker)
    if sources:
        return sources, label

    items = []
    cost = abs(_val(financials, ROW_ALIASES["cost"]))
    rd = abs(_val(financials, ROW_ALIASES["rd"]))
    sga = abs(_val(financials, ROW_ALIASES["sga"]))
    op_income = _val(financials, ROW_ALIASES["operating"])
    tax = abs(_val(financials, ROW_ALIASES["tax"]))
    net = _val(financials, ROW_ALIASES["net"])

    if cost > 0:
        items.append(("Cost of Revenue", cost))
    if rd > 0:
        items.append(("R&D", rd))
    if sga > 0:
        items.append(("SG&A", sga))
    if op_income > 0:
        items.append(("Operating Income", op_income))
    if tax > 0:
        items.append(("Tax", tax))
    if net > 0:
        items.append(("Net Income", net))

    return sorted(items, key=lambda x: x[1], reverse=True), "Cost & Profit Structure ($B)"


def compute_insights(payload: dict) -> list[tuple[str, str, tuple[float, float, float]]]:
    revenue = payload["revenue"]
    cost = payload["cost"]
    gross = payload["gross"]
    operating = payload["operating"]
    pretax = payload.get("pretax", 0)
    tax = payload["tax"]
    net = payload["net"]
    revenues = payload["revenues"]
    operatings = payload["operatings"]

    gross_margin = gross / revenue * 100 if revenue else 0
    op_margin = operating / revenue * 100 if revenue else 0
    net_margin = net / revenue * 100 if revenue else 0
    tax_base = pretax if pretax else operating + payload.get("other", 0)
    tax_rate = tax / tax_base * 100 if tax_base else 0
    cost_pct = cost / revenue * 100 if revenue else 0

    insights = [
        ("Gross Margin", f"{gross_margin:.1f}%", GREEN if gross_margin > 40 else GOLD if gross_margin > 20 else RED),
        ("Operating Margin", f"{op_margin:.1f}%", GREEN if op_margin > 20 else GOLD if op_margin > 10 else RED),
        ("Net Margin", f"{net_margin:.1f}%", GREEN if net_margin > 15 else GOLD if net_margin > 5 else RED),
        ("Cost of Rev %", f"{cost_pct:.1f}%", GREEN if cost_pct < 40 else GOLD if cost_pct < 60 else RED),
        ("Effective Tax", f"{tax_rate:.1f}%", GRAY),
    ]
    if len(revenues) >= 2 and revenues[1]:
        yoy = (revenues[0] - revenues[1]) / abs(revenues[1]) * 100
        insights.append(("Rev Growth YoY", f"{yoy:+.1f}%", GREEN if yoy > 0 else RED))
    if len(revenues) >= 2 and revenues[1] and operatings[1]:
        rev_g = (revenues[0] - revenues[1]) / abs(revenues[1]) * 100
        op_g = (operatings[0] - operatings[1]) / abs(operatings[1]) * 100
        leverage = op_g - rev_g
        insights.append(("Op Leverage", f"{leverage:+.1f}pp", GREEN if leverage > 0 else RED))
    return insights


def ribbon(ax, x0, x1, y0t, y0b, y1t, y1b, color, alpha=0.38):
    cx0 = x0 + (x1 - x0) * 0.45
    cx1 = x1 - (x1 - x0) * 0.45
    verts = [
        (x0, y0t), (cx0, y0t), (cx1, y1t), (x1, y1t),
        (x1, y1b), (cx1, y1b), (cx0, y0b), (x0, y0b), (x0, y0t),
    ]
    codes = [
        Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY,
    ]
    ax.add_patch(patches.PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", alpha=alpha))


def draw_sankey(ax, payload: dict):
    ticker = payload["ticker"]
    revenue = max(payload["revenue"], 0.0001)
    cost = max(payload["cost"], 0)
    gross = payload["gross"]
    opex = max(payload["opex"], 0)
    operating = payload["operating"]
    net = payload["net"]
    sources = payload["sources"]

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("white")

    has_sources = len(sources) > 0
    x_src = 0.08 if has_sources else None
    x_rev = 0.30 if has_sources else 0.18
    x_gross, x_op, x_net = 0.52, 0.72, 0.92
    scale = 0.52 / revenue
    t = lambda v: abs(v) * scale
    y_rev, y_op = 0.47, 0.38

    rev_top, rev_bot = y_rev + t(revenue) / 2, y_rev - t(revenue) / 2
    gross_top, gross_bot = rev_top, rev_top - t(max(gross, 0))
    op_top, op_bot = y_op + t(max(operating, 0)) / 2, y_op - t(max(operating, 0)) / 2
    net_top, net_bot = y_op + t(abs(net)) / 2, y_op - t(abs(net)) / 2
    net_color = GREEN if net >= 0 else RED
    net_label = "Net profit" if net >= 0 else "Net loss"

    if has_sources and x_src is not None:
        total_src = sum(v for _, v in sources)
        heights = [t(v * revenue / total_src) for _, v in sources] if total_src else []
        rev_span = max(rev_top - rev_bot, 0.01)
        gaps = 0.003 * max(len(sources) - 1, 0)
        usable = max(rev_span - gaps, 0.01)
        h_total = sum(heights)
        heights = [h / h_total * usable for h in heights] if h_total else heights
        rev_bands, cursor = [], rev_top
        for h in heights:
            rev_bands.append((cursor, cursor - h))
            cursor = cursor - h - 0.003
        src_bands, src_cursor = [], rev_top + 0.04
        for h in heights:
            src_h = max(h, 0.025)
            src_bands.append((src_cursor, src_cursor - src_h))
            src_cursor = src_cursor - src_h - 0.008
        for i, ((name, value), (rt, rb), (st, sb)) in enumerate(zip(sources, rev_bands, src_bands)):
            color = SOURCE_COLORS[i % len(SOURCE_COLORS)]
            pct = value / total_src * 100 if total_src else 0
            ribbon(ax, x_src + 0.012, x_rev - 0.01, st, sb, rt, rb, color, alpha=0.40)
            ax.add_patch(patches.Rectangle((x_src, sb), 0.012, st - sb, color=color, alpha=0.90, zorder=3))
            label = name if len(name) <= 22 else name[:20] + "..."
            ax.text(x_src - 0.008, (st + sb) / 2, f"{label}\n${value:.1f}B  {pct:.0f}%",
                    ha="right", va="center", fontsize=6.5, color=color, fontweight="bold", linespacing=1.4)
        ax.text(x_src + 0.006, rev_top + 0.10, "Revenue Sources", ha="center", fontsize=7.5, color=GRAY, fontstyle="italic")

    ribbon(ax, x_rev, x_gross, gross_top, gross_bot, gross_top, gross_bot, GREEN)
    ribbon(ax, x_rev, x_gross, gross_bot, rev_bot, gross_bot, rev_bot, RED)
    ribbon(ax, x_gross, x_op, gross_top, gross_bot, op_top, op_bot, GREEN)
    ribbon(ax, x_gross, x_op, gross_bot, rev_bot, op_bot, op_bot - t(max(opex - cost, 0)) * 0.5, RED)
    ribbon(ax, x_op, x_net, op_top, op_bot, net_top, net_bot, net_color)

    for x, bottom, top, color in [
        (x_rev, rev_bot, rev_top, BLACK),
        (x_gross, gross_bot, gross_top, GREEN),
        (x_op, op_bot, op_top, GREEN),
        (x_net, net_bot, net_top, net_color),
    ]:
        ax.add_patch(patches.Rectangle((x - 0.010, bottom), 0.012, top - bottom, color=color, zorder=4))

    ax.text(x_rev, rev_top + 0.055, f"Revenue\n${payload['revenue']:.1f}B", ha="center", va="bottom", fontsize=9, fontweight="bold", color=BLACK)
    ax.text(x_gross, gross_top + 0.04, f"Gross profit\n${gross:.1f}B", ha="center", fontsize=9, fontweight="bold", color=GREEN)
    ax.text(x_op, op_top + 0.04, f"Operating profit\n${operating:.1f}B", ha="center", fontsize=9, fontweight="bold", color=GREEN)
    ax.text(x_net + 0.015, (net_top + net_bot) / 2, f"{net_label}\n${abs(net):.1f}B", ha="left", va="center", fontsize=9, fontweight="bold", color=net_color)
    ax.text((x_rev + x_gross) / 2, max(0.07, (rev_bot + gross_bot) / 2 - 0.07), f"Cost of revenue\n(${cost:.1f}B)", ha="center", fontsize=8.5, fontweight="bold", color=RED)
    ax.text((x_gross + x_op) / 2, gross_bot - 0.08, f"Operating expenses\n(${opex:.1f}B)", ha="center", fontsize=8.5, fontweight="bold", color=RED)
    ax.text(0.99, 0.01, f"Source data: {payload['source_label']}", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color=GRAY, fontstyle="italic")
    ax.text(0.02, 0.02, ticker, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, color=GRAY, fontweight="bold")


def draw_segments(ax, segments, title):
    ax.set_facecolor("white")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(left=False)
    if not segments:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes, color=GRAY)
        ax.set_title(title, fontsize=10, fontweight="bold", color=TITLE_CLR, pad=6)
        return
    labels = [item[0] for item in segments]
    values = [item[1] for item in segments]
    total = sum(values)
    y_pos = np.arange(len(labels))
    colors = [SOURCE_COLORS[i % len(SOURCE_COLORS)] for i in range(len(segments))]
    bars = ax.barh(y_pos, values, color=colors, height=0.55, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.1f}B"))
    ax.tick_params(axis="x", labelsize=7.5, color=GRAY)
    for bar, value in zip(bars, values):
        pct = value / total * 100 if total else 0
        ax.text(bar.get_width() + max(values) * 0.015, bar.get_y() + bar.get_height() / 2,
                f"${value:.1f}B  ({pct:.0f}%)", va="center", fontsize=8, color=GRAY)
    ax.set_xlim(0, max(values) * 1.40)
    ax.set_title(title, fontsize=10, fontweight="bold", color=TITLE_CLR, pad=6)


def draw_margin_trends(ax, years, revenues, grosses, operatings, nets):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    yrs, rev = years[::-1], revenues[::-1]
    gro, op, net = grosses[::-1], operatings[::-1], nets[::-1]
    margins = lambda n, d: [ni / di * 100 if di else 0 for ni, di in zip(n, d)]
    x = range(len(yrs))
    ax.plot(x, margins(gro, rev), "o-", color=(*GREEN, 1), lw=2, ms=5, label="Gross Margin")
    ax.plot(x, margins(op, rev), "s-", color=(*GOLD, 1), lw=2, ms=5, label="Op Margin")
    ax.plot(x, margins(net, rev), "^-", color=(0.2, 0.5, 0.9), lw=2, ms=5, label="Net Margin")
    ax.set_xticks(list(x))
    ax.set_xticklabels(yrs, fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.tick_params(axis="y", labelsize=8)
    ax.axhline(0, color=GRAY, lw=0.6, ls="--")
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title("Margin Trends (Annual)", fontsize=10, fontweight="bold", color=TITLE_CLR, pad=6)


def draw_revenue_bars(ax, years, revenues, nets):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    yrs, rev, net = years[::-1], revenues[::-1], nets[::-1]
    x = np.arange(len(yrs))
    width = 0.38
    ax.bar(x - width / 2, rev, width=width, color=(0.2, 0.55, 0.80), label="Revenue", edgecolor="none")
    ax.bar(x + width / 2, net, width=width, color=[GREEN if v >= 0 else RED for v in net], label="Net Income", edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(yrs, fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.0f}B"))
    ax.tick_params(axis="y", labelsize=8)
    ax.axhline(0, color=GRAY, lw=0.6)
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title("Revenue vs Net Income ($B)", fontsize=10, fontweight="bold", color=TITLE_CLR, pad=6)


def draw_insights(ax, insights):
    ax.set_facecolor((0.97, 0.97, 0.97))
    ax.axis("off")
    ax.set_title("Key Metrics & Insights", fontsize=10, fontweight="bold", color=TITLE_CLR, pad=6)
    cols = 2
    rows = max((len(insights) + 1) // cols, 1)
    for i, (label, value, color) in enumerate(insights):
        col, row = i % cols, i // cols
        cx = 0.05 + col * 0.50
        cy = 0.88 - row * (0.88 / rows)
        ax.add_patch(patches.FancyBboxPatch(
            (cx, cy - 0.10), 0.43, 0.13,
            boxstyle="round,pad=0.01", facecolor="white", edgecolor=(*color, 0.5),
            linewidth=1.2, transform=ax.transAxes, clip_on=False,
        ))
        ax.text(cx + 0.215, cy + 0.01, value, ha="center", va="center", fontsize=13, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(cx + 0.215, cy - 0.065, label, ha="center", va="center", fontsize=7.5, color=GRAY, transform=ax.transAxes)


def render_income_figure(payload: dict):
    fig = plt.figure(figsize=(20, 15), dpi=140)
    fig.patch.set_facecolor("white")
    grid = gridspec.GridSpec(
        3, 2, figure=fig, hspace=0.42, wspace=0.30,
        left=0.03, right=0.97, top=0.90, bottom=0.04,
        height_ratios=[1.20, 1.0, 1.0],
    )
    ax_sankey = fig.add_subplot(grid[0, :])
    ax_segments = fig.add_subplot(grid[1, 0])
    ax_insights = fig.add_subplot(grid[1, 1])
    ax_margins = fig.add_subplot(grid[2, 0])
    ax_bars = fig.add_subplot(grid[2, 1])

    draw_sankey(ax_sankey, payload)
    draw_segments(ax_segments, payload["segment_bar"], payload["segment_label"])
    draw_insights(ax_insights, compute_insights(payload))
    draw_margin_trends(ax_margins, payload["years"], payload["revenues"], payload["grosses"], payload["operatings"], payload["nets"])
    draw_revenue_bars(ax_bars, payload["years"], payload["revenues"], payload["nets"])

    fig.text(0.02, 0.955, f"{payload['ticker']} | Annual Income Statement & Analytics",
             fontsize=20, fontweight="bold", color=TITLE_CLR, va="top")
    fig.text(0.02, 0.922, "Revenue sources -> flow -> Gross -> Operating -> Net  |  Key metrics  |  Margin trends  |  YoY comparison",
             fontsize=9.5, color=GRAY, va="top")
    fig.text(0.98, 0.012, "Research & Trading Intelligence Platform", ha="right", va="bottom", fontsize=10, color=GRAY, alpha=0.85)
    return fig


class EquityIncomeStatementModule(FazDaneModule):
    MODULE_NAME = "Equity Income Statement"
    MODULE_ICON = "📄"
    MODULE_DESCRIPTION = "Annual income statement flow, revenue mix, margins, and profitability analytics"
    TIER = 2
    SOURCE_NOTEBOOK = "Colab Equity Income Statement"
    CACHE_TTL = 21600
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["yfinance"]

    def render_sidebar(self):
        st.markdown("**Equity Universe**")
        self.universe_name, tickers, _ = render_universe_manager(
            key_prefix="income_stmt",
            show_benchmark=False,
            label="Ticker Universe:",
        )
        ticker_names = get_ticker_names(self.universe_name)
        self.tickers = tickers

        if self.tickers:
            default_index = self.tickers.index("NVDA") if "NVDA" in self.tickers else 0
            self.ticker = st.selectbox(
                "Select Ticker:",
                self.tickers,
                index=default_index,
                key="income_stmt_ticker",
                format_func=lambda ticker: format_ticker_display(ticker, ticker_names),
            )
        else:
            self.ticker = ""
        self.periods = int(st.slider("Annual Periods:", min_value=3, max_value=6, value=4, key="income_stmt_periods"))
        self.show_raw = st.checkbox("Show raw income statement rows", value=False, key="income_stmt_raw")
        st.caption(f"{len(self.tickers)} tickers selected from {self.universe_name}.")

        if st.button("Refresh Income Statement", use_container_width=True, type="primary", key="income_stmt_refresh"):
            fetch_income_statement_payload.clear()
            st.rerun()

    def render_main(self):
        self.render_section_header(
            "📄 Equity Income Statement",
            "Annual income statement flow and profitability analytics from the selected ticker universe",
        )

        if not self.ticker:
            st.warning("Select or create a ticker universe, then choose one ticker.")
            return

        with st.spinner(f"Fetching annual income statement for {self.ticker}..."):
            payload = fetch_income_statement_payload(self.ticker, self.periods)

        if payload.get("error"):
            st.warning(payload["error"])
            return

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Ticker", payload["ticker"])
        col2.metric("Latest Year", payload["latest_year"])
        col3.metric("Revenue", f"${payload['revenue']:.1f}B")
        col4.metric("Net Income", f"${payload['net']:.1f}B")

        fig = render_income_figure(payload)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        if self.show_raw:
            st.markdown("### Raw Annual Income Statement")
            st.dataframe(payload["raw_rows"], use_container_width=True)
