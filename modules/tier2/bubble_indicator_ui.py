"""
Bubble Indicator Dashboard UI (v2)
==================================
Renders the Grantham-inspired Bubble Indicator using live data from
bubble_data_engine. Layout follows the approved mockup:
header / master gauge + history / component mini-gauges / valuation snapshot /
breadth & momentum / liquidity & credit / asset flows / US vs world /
allocation + commentary / interpretation footer.
"""

import math
import re
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.tier2.bubble_data_engine import fetch_bubble_data


def _md(html, **kwargs):
    """
    Render HTML via st.markdown with indentation stripped.
    Streamlit's markdown parser treats lines indented by 4+ spaces as code
    blocks, so multi-line HTML must be collapsed to a single line first.
    """
    st.markdown(re.sub(r"\s*\n\s*", " ", str(html)).strip(),
                unsafe_allow_html=True)

BG_COLOR = "#0B1220"
PANEL_COLOR = "#111827"
BORDER_COLOR = "#293241"
TEXT_PRIMARY = "#FFFFFF"
TEXT_SECONDARY = "#9CA3AF"

GREEN = "#10B981"
YELLOW = "#F59E0B"
ORANGE = "#F97316"
RED = "#EF4444"
PURPLE = "#8B5CF6"
BLUE = "#3B82F6"

BUBBLE_EVENTS = [
    ("2000-03-01", "DOT-COM\nBUBBLE"),
    ("2007-10-01", "GLOBAL\nFINANCIAL CRISIS"),
    ("2021-11-01", "COVID\nBUBBLE"),
]


def get_risk_color_and_label(score: float):
    if score <= 20:
        return GREEN, "LOW RISK"
    if score <= 40:
        return GREEN, "MODERATE"
    if score <= 60:
        return YELLOW, "ELEVATED"
    if score <= 80:
        return ORANGE, "HIGH"
    return RED, "EXTREME RISK"


def _panel(title: str, inner_html: str, extra: str = "") -> str:
    return f"""
    <div style="background-color:{PANEL_COLOR}; border:1px solid {BORDER_COLOR};
                border-radius:10px; padding:16px; margin-bottom:14px;">
        <div style="color:{TEXT_PRIMARY}; font-size:13px; font-weight:bold;
                    text-transform:uppercase; letter-spacing:1px;
                    margin-bottom:12px; display:flex;
                    justify-content:space-between;">
            <span>{title}</span><span style="color:{TEXT_SECONDARY};
            font-weight:normal;">{extra}</span>
        </div>
        {inner_html}
    </div>"""


# ---------------------------------------------------------------------------
# Gauges & charts
# ---------------------------------------------------------------------------

def render_master_gauge(score: float):
    color, _ = get_risk_color_and_label(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 46, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": TEXT_SECONDARY,
                     "tickvals": [0, 25, 50, 75, 100],
                     "tickfont": {"size": 11, "color": TEXT_SECONDARY}},
            "bar": {"color": "rgba(0,0,0,0)"},
            "bgcolor": PANEL_COLOR,
            "borderwidth": 0,
            "steps": [
                {"range": [0, 20], "color": "#0e7a54"},
                {"range": [20, 40], "color": GREEN},
                {"range": [40, 60], "color": YELLOW},
                {"range": [60, 80], "color": ORANGE},
                {"range": [80, 100], "color": RED},
            ],
            "threshold": {"line": {"color": TEXT_PRIMARY, "width": 5},
                          "thickness": 0.85, "value": score},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=25, r=25, t=25, b=10), height=230,
        font={"color": TEXT_PRIMARY},
    )
    return fig


def render_history_chart(history: pd.Series):
    fig = go.Figure()

    # Danger-zone bands.
    zones = [(0, 20, GREEN, 0.05), (20, 40, GREEN, 0.03),
             (40, 60, YELLOW, 0.05), (60, 80, ORANGE, 0.06),
             (80, 100, RED, 0.08)]
    for lo, hi, col, op in zones:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=col, opacity=op, line_width=0)
    for y, col, lbl in [(80, RED, "EXTREME"), (60, ORANGE, "HIGH"),
                        (40, YELLOW, "ELEVATED"), (20, GREEN, "MODERATE")]:
        fig.add_hline(y=y, line_dash="dash", line_color=col, line_width=1,
                      opacity=0.6, annotation_text=lbl,
                      annotation_position="right",
                      annotation_font=dict(size=9, color=col))

    fig.add_trace(go.Scatter(
        x=history.index, y=history.values, mode="lines", name="Bubble Score",
        line=dict(color=ORANGE, width=1.6),
        hovertemplate="%{x|%b %Y}: %{y:.0f}<extra></extra>",
    ))

    # Historical bubble annotations (only where data exists).
    for date_str, label in BUBBLE_EVENTS:
        dt = pd.Timestamp(date_str)
        if history.index.min() <= dt <= history.index.max():
            i = history.index.get_indexer([dt], method="nearest")[0]
            fig.add_annotation(
                x=history.index[i], y=min(float(history.iloc[i]) + 8, 99),
                text=label.replace("\n", "<br>"), showarrow=True,
                arrowhead=0, arrowcolor=TEXT_SECONDARY, ax=0, ay=-28,
                font=dict(size=9, color=TEXT_PRIMARY),
            )
    # NOW marker.
    fig.add_trace(go.Scatter(
        x=[history.index[-1]], y=[history.values[-1]], mode="markers+text",
        marker=dict(color=RED, size=9, line=dict(color=TEXT_PRIMARY, width=1)),
        text=["NOW"], textposition="top center",
        textfont=dict(size=10, color=TEXT_PRIMARY), showlegend=False,
        hoverinfo="skip",
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=70, t=25, b=10), height=250, showlegend=False,
        font=dict(color=TEXT_PRIMARY, size=10),
        xaxis=dict(showgrid=False, showline=True, linecolor=BORDER_COLOR),
        yaxis=dict(showgrid=True, gridcolor="rgba(41,50,65,0.5)",
                   range=[0, 105], tickvals=[0, 25, 50, 75, 100]),
    )
    return fig


def _mini_gauge_svg(score: float, color: str) -> str:
    """Semicircular SVG arc gauge with needle (matches mockup style)."""
    ang = math.radians(180 - score * 1.8)
    nx = 60 + 40 * math.cos(ang)
    ny = 58 - 40 * math.sin(ang)
    uid = f"g{abs(hash((round(score, 1), color))) % 99999}"
    return f"""
    <svg width="120" height="70" viewBox="0 0 120 70">
      <defs>
        <linearGradient id="{uid}" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="{GREEN}"/>
          <stop offset="45%" stop-color="{YELLOW}"/>
          <stop offset="72%" stop-color="{ORANGE}"/>
          <stop offset="100%" stop-color="{RED}"/>
        </linearGradient>
      </defs>
      <path d="M 14 58 A 46 46 0 0 1 106 58" fill="none"
            stroke="url(#{uid})" stroke-width="9" stroke-linecap="round"/>
      <line x1="60" y1="58" x2="{nx:.1f}" y2="{ny:.1f}"
            stroke="{TEXT_PRIMARY}" stroke-width="2.5" stroke-linecap="round"/>
      <circle cx="60" cy="58" r="4" fill="{TEXT_PRIMARY}"/>
    </svg>"""


def _trend_arrow(delta: float) -> str:
    if delta > 2:
        return f'<span style="color:{RED};">&#9650; +{delta:.0f}</span>'
    if delta < -2:
        return f'<span style="color:{GREEN};">&#9660; {delta:.0f}</span>'
    return f'<span style="color:{TEXT_SECONDARY};">&#9654; {delta:+.0f}</span>'


def _component_card(title: str, subtitle: str, score: float,
                    delta: float, icon: str) -> str:
    color, label = get_risk_color_and_label(score)
    return f"""
    <div style="background-color:{PANEL_COLOR}; border:1px solid {BORDER_COLOR};
                border-radius:10px; padding:12px 8px; text-align:center;">
      <div style="color:{TEXT_PRIMARY}; font-size:11px; font-weight:bold;
                  text-transform:uppercase;">{icon} {title}</div>
      <div style="color:{TEXT_SECONDARY}; font-size:9px;
                  margin-bottom:4px;">{subtitle}</div>
      {_mini_gauge_svg(score, color)}
      <div style="font-size:26px; font-weight:bold; color:{color};
                  line-height:1;">{int(round(score))}</div>
      <div style="font-size:10px; font-weight:bold; color:{color};
                  text-transform:uppercase; margin-top:2px;">{label}</div>
      <div style="font-size:10px; margin-top:3px;">{_trend_arrow(delta)}
        <span style="color:{TEXT_SECONDARY};">1M</span></div>
    </div>"""


def _sparkline_svg(values: list, color: str, w: int = 90, h: int = 22) -> str:
    if not values or len(values) < 2:
        return ""
    vmin, vmax = min(values), max(values)
    rng = (vmax - vmin) or 1.0
    step = w / (len(values) - 1)
    pts = " ".join(
        f"{i * step:.1f},{h - 2 - (v - vmin) / rng * (h - 4):.1f}"
        for i, v in enumerate(values)
    )
    return (f'<svg width="{w}" height="{h}"><polyline points="{pts}" '
            f'fill="none" stroke="{color}" stroke-width="1.5"/></svg>')


def render_breadth_chart(series: pd.Series):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values, mode="lines",
        line=dict(color=ORANGE, width=1.6),
        hovertemplate="%{x|%b %d, %Y}: %{y:.0f}%<extra></extra>",
    ))
    fig.add_hline(y=50, line_dash="dot", line_color=TEXT_SECONDARY,
                  line_width=1, opacity=0.5)
    fig.update_layout(
        title=dict(text="% STOCKS ABOVE 200-DMA (S&P 100)",
                   font=dict(size=10, color=TEXT_SECONDARY), x=0.5),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10), height=190,
        font=dict(color=TEXT_PRIMARY, size=9), showlegend=False,
        xaxis=dict(showgrid=False, linecolor=BORDER_COLOR),
        yaxis=dict(showgrid=True, gridcolor="rgba(41,50,65,0.5)",
                   range=[0, 100], ticksuffix="%"),
    )
    return fig


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------

def render_bubble_indicator_dashboard():
    with st.spinner("Computing bubble indicators from live market data..."):
        data = fetch_bubble_data()

    if "error" in data:
        st.error(data["error"])
        st.json(data.get("data_quality", {}))
        return

    master = data["master_score"]
    comps = data["components"]
    trends = data["trends"]
    ctx = data["context"]
    color, label = get_risk_color_and_label(master)

    # ------------------------------------------------------------- header
    vix_txt = "N/A" if pd.isna(ctx["vix"]) else f"{ctx['vix']:.1f}"
    _md(f"""
    <div style="display:flex; justify-content:space-between; align-items:end;
                margin-bottom:14px;">
      <div>
        <h1 style="margin:0; padding:0; color:{TEXT_PRIMARY}; font-size:28px;
                   letter-spacing:1px;">BUBBLE INDICATOR DASHBOARD</h1>
        <div style="color:{TEXT_SECONDARY}; font-size:14px;">
          A Grantham-Inspired Framework for Market Excess &amp; Risk</div>
      </div>
      <div style="text-align:right; color:{TEXT_SECONDARY}; font-size:12px;
                  line-height:1.6;">
        <div>&#128197; UPDATED: {data['as_of']} &nbsp;|&nbsp; FREQUENCY: Daily</div>
        <div>REGIME: <b style="color:{color};">{ctx['regime']}</b>
          &nbsp;|&nbsp; VIX: <b style="color:{TEXT_PRIMARY};">{vix_txt}</b>
          &nbsp;|&nbsp; FED: <b style="color:{TEXT_PRIMARY};">{ctx['fed_status']}</b></div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ---------------------------------------------------- PDF export
    pc1, pc2, _spacer = st.columns([1, 1.2, 3.8])
    with pc1:
        if st.button("Generate PDF Report", key="bubble_pdf_generate"):
            with st.spinner("Rendering PDF report..."):
                try:
                    from modules.tier2.bubble_pdf_generator import build_bubble_pdf
                    st.session_state["bubble_pdf_bytes"] = build_bubble_pdf(data)
                    st.session_state["bubble_pdf_stamp"] = datetime.now().strftime("%Y%m%d_%H%M")
                except Exception as exc:
                    st.session_state["bubble_pdf_bytes"] = None
                    st.error(f"PDF generation failed: {exc}")
    with pc2:
        if st.session_state.get("bubble_pdf_bytes"):
            st.download_button(
                "Download Bubble Report (PDF)",
                data=st.session_state["bubble_pdf_bytes"],
                file_name=f"bubble_indicator_report_{st.session_state.get('bubble_pdf_stamp', '')}.pdf",
                mime="application/pdf",
                key="bubble_pdf_download",
            )

    # ------------------------------------------- master gauge + history
    c1, c2 = st.columns([1, 1.5])
    with c1:
        _md(
            f'<div style="color:{TEXT_PRIMARY}; font-size:13px; font-weight:bold; '
            f'text-transform:uppercase; letter-spacing:1px;">OVERALL BUBBLE SCORE</div>',
            unsafe_allow_html=True)
        st.plotly_chart(render_master_gauge(master), use_container_width=True,
                        key="bubble_master_gauge")
        _md(f"""
        <div style="text-align:center; margin-top:-15px;">
          <div style="color:{color}; font-weight:bold; font-size:18px;">{label}</div>
          <div style="color:{TEXT_SECONDARY}; font-size:12px;">
            12M crash probability (heuristic): <b style="color:{color};">
            {data['crash_prob_12m']:.0f}%</b></div>
        </div>""", unsafe_allow_html=True)
    with c2:
        _md(
            f'<div style="color:{TEXT_PRIMARY}; font-size:13px; font-weight:bold; '
            f'text-transform:uppercase; letter-spacing:1px;">BUBBLE SCORE HISTORY '
            f'<span style="color:{TEXT_SECONDARY}; font-weight:normal; font-size:11px;">'
            f'(computed from component series)</span></div>',
            unsafe_allow_html=True)
        if not data["history"].empty:
            st.plotly_chart(render_history_chart(data["history"]),
                            use_container_width=True, key="bubble_history")
        else:
            st.info("History unavailable.")

    # --------------------------------------------------- component cards
    _md(
        f'<div style="color:{TEXT_PRIMARY}; font-size:13px; font-weight:bold; '
        f'text-transform:uppercase; letter-spacing:1px; margin:6px 0 8px;">'
        f'BUBBLE SCORE COMPONENTS</div>', unsafe_allow_html=True)
    cards = [
        ("Valuation", "vs. History", "Valuation", "&#128269;"),
        ("Price Momentum", "vs. Fundamentals", "Momentum", "&#128200;"),
        ("Market Excitement", "Sentiment & Narratives", "Market Excitement", "&#128172;"),
        ("Credit Expansion", "Leverage & Liquidity", "Credit", "&#127974;"),
        ("Liquidity", "Rates, M2, Fed B/S", "Liquidity", "&#128167;"),
        ("Profit Margins", "vs. History", "Profit Margins", "&#128176;"),
        ("Concentration", "Mega-cap Weight", "Concentration", "&#128101;"),
        ("AI Bubble", "SMH & NVDA vs SPY", "AI Bubble", "&#129504;"),
    ]
    row1, row2 = cards[:4], cards[4:]
    for row in (row1, row2):
        cols = st.columns(4)
        for col, (title, subtitle, key, icon) in zip(cols, row):
            with col:
                if key in comps:
                    _md(
                        _component_card(title, subtitle, comps[key],
                                        trends.get(key, 0.0), icon),
                        unsafe_allow_html=True)
                else:
                    _md(_panel(title, f'<div style="color:{TEXT_SECONDARY};'
                                              f'font-size:11px;">Unavailable</div>'),
                                unsafe_allow_html=True)

    _md("<div style='height:6px;'></div>", unsafe_allow_html=True)

    # ------------------------------- valuation snapshot | breadth panel
    c3, c4 = st.columns([1.4, 1])
    with c3:
        if data["val_snapshot"]:
            rows_html = ""
            for row in data["val_snapshot"]:
                p = row["Percentile"]
                bar = RED if p > 80 else ORANGE if p > 60 else YELLOW if p > 40 else GREEN
                rows_html += f"""
                <tr>
                  <td style="padding:7px 4px; border-bottom:1px solid {BORDER_COLOR};
                             color:{TEXT_PRIMARY};">{row['Metric']}</td>
                  <td style="padding:7px 4px; border-bottom:1px solid {BORDER_COLOR};
                             color:{bar}; font-weight:bold;">{row['Current']}</td>
                  <td style="padding:7px 4px; border-bottom:1px solid {BORDER_COLOR};
                             color:{TEXT_SECONDARY};">{row['Historical Avg']}</td>
                  <td style="padding:7px 4px; border-bottom:1px solid {BORDER_COLOR};">
                    <div style="display:flex; align-items:center; gap:8px;">
                      <span style="color:{bar}; width:34px; font-weight:bold;">{p}%</span>
                      <div style="flex-grow:1; background:{BG_COLOR}; height:8px;
                                  border-radius:4px; overflow:hidden;">
                        <div style="width:{p}%; background:{bar}; height:100%;"></div>
                      </div>
                    </div>
                  </td>
                </tr>"""
            table = f"""
            <table style="width:100%; border-collapse:collapse; font-size:12px;">
              <tr>
                <th style="text-align:left; color:{TEXT_SECONDARY}; padding:6px 4px;
                           border-bottom:1px solid {BORDER_COLOR}; font-weight:normal;">METRIC</th>
                <th style="text-align:left; color:{TEXT_SECONDARY}; padding:6px 4px;
                           border-bottom:1px solid {BORDER_COLOR}; font-weight:normal;">CURRENT</th>
                <th style="text-align:left; color:{TEXT_SECONDARY}; padding:6px 4px;
                           border-bottom:1px solid {BORDER_COLOR}; font-weight:normal;">HIST AVG</th>
                <th style="text-align:left; color:{TEXT_SECONDARY}; padding:6px 4px;
                           border-bottom:1px solid {BORDER_COLOR}; font-weight:normal;">PERCENTILE (full history)</th>
              </tr>{rows_html}
            </table>"""
            _md(_panel("VALUATION SNAPSHOT", table,
                               "live: multpl.com monthly history"),
                        unsafe_allow_html=True)
        else:
            _md(_panel("VALUATION SNAPSHOT",
                               f'<div style="color:{TEXT_SECONDARY}; font-size:12px;">'
                               f'multpl.com unreachable - valuation table unavailable.</div>'),
                        unsafe_allow_html=True)

    with c4:
        br = data["breadth"]
        if br.get("available"):
            pct200 = br["pct_above_200"]
            pcol = GREEN if pct200 > 60 else YELLOW if pct200 > 45 else RED
            stats = f"""
            <table style="width:100%; border-collapse:collapse; font-size:12px;">
              <tr><td style="color:{TEXT_SECONDARY}; padding:5px 0;">% Stocks Above 200-DMA</td>
                  <td style="color:{pcol}; text-align:right; font-weight:bold;
                             font-size:15px;">{pct200:.0f}%</td></tr>
              <tr><td style="color:{TEXT_SECONDARY}; padding:5px 0;">New 52W Highs / Lows</td>
                  <td style="color:{TEXT_PRIMARY}; text-align:right; font-weight:bold;
                             font-size:15px;">{br['new_highs']} / {br['new_lows']}</td></tr>
              <tr><td style="color:{TEXT_SECONDARY}; padding:5px 0;">Advance / Decline (today)</td>
                  <td style="color:{TEXT_PRIMARY}; text-align:right; font-weight:bold;
                             font-size:15px;">{br['adv_dec']}</td></tr>
              <tr><td style="color:{TEXT_SECONDARY}; padding:5px 0;">S&amp;P Momentum (12M)</td>
                  <td style="color:{ORANGE}; text-align:right; font-weight:bold;
                             font-size:15px;">{br.get('momentum_12m', float('nan')):.1f}%</td></tr>
            </table>"""
            _md(_panel("MARKET BREADTH & MOMENTUM", stats,
                               f"universe: {br['universe']} stocks"),
                        unsafe_allow_html=True)
            st.plotly_chart(render_breadth_chart(br["pct_above_series"]),
                            use_container_width=True, key="breadth_chart")
            narrow = comps.get("Breadth", 50)
            if narrow > 60:
                _md(f"""
                <div style="border:1px solid {RED}; padding:9px; border-radius:6px;
                            color:{RED}; font-size:12px;">&#9888;&#65039;
                  Participation is narrowing - fewer stocks are driving index gains.
                </div>""", unsafe_allow_html=True)
        else:
            _md(_panel("MARKET BREADTH & MOMENTUM",
                               f'<div style="color:{TEXT_SECONDARY}; font-size:12px;">'
                               f'Constituent data unavailable.</div>'),
                        unsafe_allow_html=True)

    # ---------------------- liquidity | asset YTD | US vs world
    c5, c6, c7 = st.columns(3)
    with c5:
        liq = data["liquidity_snapshot"]
        if liq:
            rows_html = ""
            for row in liq:
                sc = {"TIGHT": RED, "NEUTRAL": YELLOW, "LOOSE": GREEN}[row["Status"]]
                rows_html += f"""
                <tr>
                  <td style="padding:6px 2px; border-bottom:1px solid {BORDER_COLOR};
                             color:{TEXT_PRIMARY};">{row['Indicator']}</td>
                  <td style="padding:6px 2px; border-bottom:1px solid {BORDER_COLOR};
                             font-weight:bold; color:{TEXT_PRIMARY};">{row['Level']}</td>
                  <td style="padding:6px 2px; border-bottom:1px solid {BORDER_COLOR};">
                    {_sparkline_svg(row['Spark'], sc)}</td>
                  <td style="padding:6px 2px; border-bottom:1px solid {BORDER_COLOR};
                             color:{sc}; font-weight:bold; font-size:11px;">{row['Status']}</td>
                </tr>"""
            table = f"""
            <table style="width:100%; border-collapse:collapse; font-size:11px;">
              <tr>
                <th style="text-align:left; color:{TEXT_SECONDARY}; font-weight:normal;
                           padding:4px 2px;">INDICATOR</th>
                <th style="text-align:left; color:{TEXT_SECONDARY}; font-weight:normal;
                           padding:4px 2px;">LEVEL</th>
                <th style="text-align:left; color:{TEXT_SECONDARY}; font-weight:normal;
                           padding:4px 2px;">TREND (6M)</th>
                <th style="text-align:left; color:{TEXT_SECONDARY}; font-weight:normal;
                           padding:4px 2px;">STATUS</th>
              </tr>{rows_html}
            </table>"""
            _md(_panel("LIQUIDITY & CREDIT CONDITIONS", table, "live: FRED"),
                        unsafe_allow_html=True)
        else:
            _md(_panel("LIQUIDITY & CREDIT CONDITIONS",
                               f'<div style="color:{TEXT_SECONDARY}; font-size:12px;">'
                               f'Add FRED_API_KEY to .streamlit/secrets.toml for live '
                               f'rates, spreads and money-supply data.</div>'),
                        unsafe_allow_html=True)

    with c6:
        ytd = data["asset_ytd"]
        if ytd:
            max_abs = max(abs(r["YTD"]) for r in ytd) or 1
            rows_html = ""
            for r in ytd:
                v = r["YTD"]
                bc = GREEN if v >= 0 else RED
                width = abs(v) / max_abs * 100
                rows_html += f"""
                <tr>
                  <td style="padding:6px 2px; color:{TEXT_PRIMARY};
                             border-bottom:1px solid {BORDER_COLOR};">{r['Icon']} {r['Asset']}</td>
                  <td style="padding:6px 2px; color:{bc}; font-weight:bold; width:52px;
                             border-bottom:1px solid {BORDER_COLOR};">{v:+.1f}%</td>
                  <td style="padding:6px 2px; border-bottom:1px solid {BORDER_COLOR}; width:40%;">
                    <div style="background:{BG_COLOR}; height:8px; border-radius:4px;">
                      <div style="width:{width:.0f}%; background:{bc}; height:100%;
                                  border-radius:4px;"></div></div></td>
                </tr>"""
            table = (f'<table style="width:100%; border-collapse:collapse; '
                     f'font-size:12px;">{rows_html}</table>'
                     f'<div style="margin-top:10px; color:{TEXT_SECONDARY}; '
                     f'font-size:10px;">Total-return YTD by asset class '
                     f'(performance proxy for capital flows).</div>')
            _md(_panel("ASSET CLASS PERFORMANCE (YTD)", table, "live"),
                        unsafe_allow_html=True)

    with c7:
        vw = data["us_vs_world"]
        if vw:
            rows_html = ""
            for r in vw:
                us, world = r["US"], r["World"]
                mx = max(us, world) or 1
                rows_html += f"""
                <div style="margin-bottom:10px;">
                  <div style="color:{TEXT_SECONDARY}; font-size:11px;
                              margin-bottom:3px;">{r['Metric']}
                    <span style="float:right; color:{TEXT_PRIMARY};
                                 font-weight:bold;">{r['Ratio']}x</span></div>
                  <div style="background:{BG_COLOR}; border-radius:3px; margin-bottom:2px;">
                    <div style="width:{us / mx * 100:.0f}%; background:{RED}; height:9px;
                                border-radius:3px; color:white; font-size:8px;
                                padding-left:4px; line-height:9px;">{us}</div></div>
                  <div style="background:{BG_COLOR}; border-radius:3px;">
                    <div style="width:{world / mx * 100:.0f}%; background:{BLUE}; height:9px;
                                border-radius:3px; color:white; font-size:8px;
                                padding-left:4px; line-height:9px;">{world}</div></div>
                </div>"""
            legend = f"""
            <div style="font-size:10px; color:{TEXT_SECONDARY}; margin-bottom:8px;">
              <span style="display:inline-block; width:9px; height:9px;
                           background:{RED}; margin-right:4px;"></span>US (SPY)
              <span style="display:inline-block; width:9px; height:9px;
                           background:{BLUE}; margin:0 4px 0 12px;"></span>World ex-US (EFA/VWO)
            </div>"""
            note = ""
            pe_row = next((r for r in vw if "P/E" in r["Metric"] and r["Ratio"]), None)
            if pe_row and pe_row["Ratio"] > 1.3:
                note = (f'<div style="margin-top:8px; border:1px solid {BLUE}; padding:8px; '
                        f'border-radius:6px; color:{BLUE}; font-size:11px;">&#127757; '
                        f'US equities trade at {pe_row["Ratio"]}x world multiples.</div>')
            _md(_panel("VALUATION: US VS. WORLD", legend + rows_html + note,
                               "live: fund fundamentals"),
                        unsafe_allow_html=True)
        else:
            _md(_panel("VALUATION: US VS. WORLD",
                               f'<div style="color:{TEXT_SECONDARY}; font-size:12px;">'
                               f'Fundamental data unavailable right now.</div>'),
                        unsafe_allow_html=True)

    # ------------------------------------ commentary + allocation
    c8, c9 = st.columns([1.6, 1])
    with c8:
        _md(_panel("FRAMEWORK COMMENTARY",
                           f'<div style="color:{TEXT_PRIMARY}; font-size:13px; '
                           f'line-height:1.7;">{data["commentary"]}</div>',
                           "auto-generated"), unsafe_allow_html=True)
    with c9:
        alloc = data["allocation"]
        colors = {"Equities": ORANGE, "International": BLUE, "Treasuries": GREEN,
                  "Gold": YELLOW, "Cash": TEXT_SECONDARY}
        bars = ""
        for k, v in alloc.items():
            bars += f"""
            <div style="margin-bottom:7px;">
              <span style="color:{TEXT_SECONDARY}; font-size:11px;">{k}
                <span style="float:right; color:{TEXT_PRIMARY};
                             font-weight:bold;">{v}%</span></span>
              <div style="background:{BG_COLOR}; height:8px; border-radius:4px;
                          margin-top:2px;">
                <div style="width:{v}%; background:{colors.get(k, BLUE)};
                            height:100%; border-radius:4px;"></div></div>
            </div>"""
        bars += (f'<div style="color:{TEXT_SECONDARY}; font-size:10px; '
                 f'margin-top:8px;">Rule-based moderate profile mapped to the '
                 f'current score band. Not investment advice.</div>')
        _md(_panel("SUGGESTED ALLOCATION (MODERATE)", bars),
                    unsafe_allow_html=True)

    # -------------------------------------------------- footer
    dq = data["data_quality"]
    dq_html = " &nbsp;&bull;&nbsp; ".join(
        f"<b>{k}</b>: {v}" for k, v in dq.items())
    _md(f"""
    <div style="display:flex; gap:20px; font-size:11px; color:{TEXT_SECONDARY};
                border-top:1px solid {BORDER_COLOR}; padding-top:14px; margin-top:8px;">
      <div style="flex:1;">
        <strong style="color:{TEXT_PRIMARY};">HOW TO INTERPRET</strong><br>
        <span style="color:{GREEN};">0-20 LOW RISK:</span> Attractive valuations, normal risk.<br>
        <span style="color:{GREEN};">21-40 MODERATE:</span> Caution warranted, monitor closely.<br>
        <span style="color:{YELLOW};">41-60 ELEVATED:</span> Risk building, be selective.<br>
        <span style="color:{ORANGE};">61-80 HIGH:</span> High risk of mean reversion.<br>
        <span style="color:{RED};">81-100 EXTREME:</span> Extremely overvalued, high crash risk.
      </div>
      <div style="flex:1;">
        <strong style="color:{TEXT_PRIMARY};">METHODOLOGY</strong><br>
        &bull; Scores are true percentile ranks vs. all available history.<br>
        &bull; Trending ratios are de-trended vs. a 3-year rolling mean.<br>
        &bull; History is computed daily from weighted component series.<br>
        &bull; Weights: Val 20 / Mom 15 / Sent 15 / Liq, Credit, Breadth, Conc, AI 10 each.<br>
        &bull; Risk-management tool, not a timing signal.
      </div>
      <div style="flex:1;">
        <strong style="color:{TEXT_PRIMARY};">DATA SOURCES</strong><br>
        <div style="line-height:1.7;">{dq_html}</div>
      </div>
    </div>""", unsafe_allow_html=True)
