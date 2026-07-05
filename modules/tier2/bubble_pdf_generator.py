"""
Bubble Indicator PDF Report Generator
=====================================
Builds a dark-themed 2-page A4 portrait PDF of the Bubble Indicator
dashboard from the data dict produced by bubble_data_engine.fetch_bubble_data.

Follows the platform conventions of utils/volatility_pdf_generator.py:
- fpdf2 layout, matplotlib (Agg) chart rendering, no kaleido dependency
- returns bytes for st.download_button
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF

LOGO_PATH = Path("assets") / "logo.png"

# Dashboard theme
BG = (11, 18, 32)          # #0B1220
PANEL = (17, 24, 39)       # #111827
BORDER = (41, 50, 65)      # #293241
TXT = (255, 255, 255)
TXT2 = (156, 163, 175)     # #9CA3AF
GREEN = (16, 185, 129)
YELLOW = (245, 158, 11)
ORANGE = (249, 115, 22)
RED = (239, 68, 68)
BLUE = (59, 130, 246)

BG_HEX = "#0B1220"
PANEL_HEX = "#111827"
BORDER_HEX = "#293241"


def sanitize_text(text):
    """Strip all non-ASCII chars that crash fpdf2 default fonts."""
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        "—": "-", "–": "-",
        "“": '"', "”": '"',
        "‘": "'", "’": "'",
        "…": "...",
        "±": "+/-",
        "≈": "~",
        "≥": ">=", "≤": "<=",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("ascii", "ignore").decode("ascii").strip()


def _risk_color_label(score):
    if score <= 20:
        return GREEN, "LOW RISK"
    if score <= 40:
        return GREEN, "MODERATE"
    if score <= 60:
        return YELLOW, "ELEVATED"
    if score <= 80:
        return ORANGE, "HIGH"
    return RED, "EXTREME RISK"


# ---------------------------------------------------------------------------
# Matplotlib chart renderers (dark theme, temp PNG files)
# ---------------------------------------------------------------------------

def _render_history_png(history, path, spx=None):
    if history is None or len(history) < 2:
        return False
    fig, ax = plt.subplots(figsize=(7.6, 2.05), dpi=160)
    fig.patch.set_facecolor(PANEL_HEX)
    ax.set_facecolor(PANEL_HEX)
    for lo, hi, col, alpha in [(80, 100, "#EF4444", 0.10), (60, 80, "#F97316", 0.08),
                               (40, 60, "#F59E0B", 0.06), (0, 40, "#10B981", 0.04)]:
        ax.axhspan(lo, hi, color=col, alpha=alpha)
    for y, col in [(80, "#EF4444"), (60, "#F97316"), (40, "#F59E0B"), (20, "#10B981")]:
        ax.axhline(y, color=col, linewidth=0.7, linestyle="--", alpha=0.6)
    if spx is not None and len(spx) > 1:
        import numpy as np
        log_px = np.log(spx.values.astype(float))
        rng = log_px.max() - log_px.min()
        norm = (log_px - log_px.min()) / (rng if rng else 1.0) * 100.0
        ax.plot(spx.index, norm, color="#3B82F6", linewidth=0.9, alpha=0.65)
    ax.plot(history.index, history.values, color="#F97316", linewidth=1.2,
            zorder=5)
    ax.plot(history.index[-1], history.values[-1], "o", color="#EF4444",
            markersize=5, markeredgecolor="white", markeredgewidth=0.8,
            zorder=6)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="both", labelsize=6, colors="#9CA3AF")
    ax.grid(True, color="#293241", linewidth=0.5, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color("#293241")
    if spx is not None and len(spx) > 1:
        ax.plot([], [], color="#F97316", linewidth=1.2, label="Bubble Score")
        ax.plot([], [], color="#3B82F6", linewidth=0.9,
                label="S&P 500 (log, normalized 0-100)")
        ax.legend(fontsize=5.5, loc="upper left", framealpha=0.3,
                  facecolor=PANEL_HEX, edgecolor="#293241",
                  labelcolor="#9CA3AF")
    fig.tight_layout(pad=0.6)
    fig.savefig(path, format="png", facecolor=PANEL_HEX, bbox_inches="tight")
    plt.close(fig)
    return os.path.exists(path) and os.path.getsize(path) > 0


def _render_breadth_png(series, path):
    if series is None or len(series) < 2:
        return False
    fig, ax = plt.subplots(figsize=(4.6, 1.9), dpi=160)
    fig.patch.set_facecolor(PANEL_HEX)
    ax.set_facecolor(PANEL_HEX)
    ax.plot(series.index, series.values, color="#F97316", linewidth=1.1)
    ax.axhline(50, color="#9CA3AF", linewidth=0.6, linestyle=":", alpha=0.6)
    ax.set_ylim(0, 100)
    ax.set_title("% STOCKS ABOVE 200-DMA (S&P 100)", fontsize=6,
                 color="#9CA3AF", fontweight="bold")
    ax.tick_params(axis="both", labelsize=5.5, colors="#9CA3AF")
    ax.grid(True, color="#293241", linewidth=0.5, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color("#293241")
    fig.tight_layout(pad=0.5)
    fig.savefig(path, format="png", facecolor=PANEL_HEX, bbox_inches="tight")
    plt.close(fig)
    return os.path.exists(path) and os.path.getsize(path) > 0


def _render_gauge_png(score, path):
    """Semicircular gauge with needle, matching the dashboard master gauge."""
    import numpy as np

    fig, ax = plt.subplots(figsize=(2.9, 1.9), dpi=160,
                           subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor(PANEL_HEX)
    ax.set_facecolor(PANEL_HEX)
    bands = [(0, 20, "#0e7a54"), (20, 40, "#10B981"), (40, 60, "#F59E0B"),
             (60, 80, "#F97316"), (80, 100, "#EF4444")]
    for lo, hi, col in bands:
        theta = np.linspace(np.pi * (1 - lo / 100), np.pi * (1 - hi / 100), 30)
        ax.plot(theta, [1.0] * len(theta), color=col, linewidth=10,
                solid_capstyle="butt")
    ang = np.pi * (1 - score / 100)
    ax.plot([ang, ang], [0, 0.82], color="white", linewidth=2)
    ax.plot(0, 0, "o", color="white", markersize=6)
    ax.set_ylim(0, 1.15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, format="png", facecolor=PANEL_HEX, bbox_inches="tight")
    plt.close(fig)
    # Polar axes save as a square canvas; crop to the top semicircle.
    try:
        from PIL import Image
        im = Image.open(path)
        im.crop((0, 0, im.size[0], int(im.size[1] * 0.62))).save(path)
    except Exception:
        pass
    return os.path.exists(path) and os.path.getsize(path) > 0


# ---------------------------------------------------------------------------
# PDF document
# ---------------------------------------------------------------------------

class BubblePDF(FPDF):
    def __init__(self, as_of):
        super().__init__(orientation="P", format="A4")  # 210 x 297 mm
        self.as_of = as_of
        self.set_margins(10, 10, 10)
        self.set_auto_page_break(auto=False)

    def header(self):
        # Full dark page background
        self.set_fill_color(*BG)
        self.rect(0, 0, 210, 297, style="F")
        # Top brand bar
        self.set_fill_color(*PANEL)
        self.rect(0, 0, 210, 14, style="F")
        self.set_draw_color(*BORDER)
        self.line(0, 14, 210, 14)
        if LOGO_PATH.exists():
            self.image(str(LOGO_PATH), x=188, y=2, w=14)
        self.set_font("Helvetica", style="B", size=12)
        self.set_text_color(*TXT)
        self.set_xy(10, 2)
        self.cell(0, 6, "BUBBLE INDICATOR DASHBOARD", align="L")
        self.set_font("Helvetica", size=7)
        self.set_text_color(*TXT2)
        self.set_xy(10, 8)
        self.cell(0, 4, sanitize_text(
            f"A Grantham-Inspired Framework for Market Excess & Risk  |  "
            f"Data as of {self.as_of}  |  Generated {datetime.now():%b %d, %Y %H:%M}"),
            align="L")
        self.set_y(18)

    def footer(self):
        self.set_y(-8)
        self.set_font("Helvetica", style="I", size=6.5)
        self.set_text_color(120, 128, 140)
        self.cell(0, 5, "Copyright (c) FazDane Analytics | Research & Trading "
                        "Intelligence Platform - Confidential | Risk-management "
                        "tool, not financial advice.", align="C")

    # -- drawing helpers ---------------------------------------------------

    def panel_title(self, x, y, w, title, extra=""):
        self.set_xy(x, y)
        self.set_font("Helvetica", style="B", size=8)
        self.set_text_color(*TXT)
        self.cell(w * 0.6, 5, sanitize_text(title), align="L")
        if extra:
            self.set_font("Helvetica", size=6)
            self.set_text_color(*TXT2)
            self.cell(w * 0.4, 5, sanitize_text(extra), align="R")
        self.set_draw_color(*BORDER)
        self.line(x, y + 5.5, x + w, y + 5.5)

    def bar(self, x, y, w, h, pct, color):
        self.set_fill_color(*BG)
        self.rect(x, y, w, h, style="F")
        self.set_fill_color(*color)
        self.rect(x, y, max(w * pct / 100.0, 0.5), h, style="F")


def build_bubble_pdf(data) -> bytes:
    """Render the bubble dashboard data dict into a 2-page PDF. Returns bytes."""
    master = float(data["master_score"])
    color, label = _risk_color_label(master)
    comps = data.get("components", {})
    trends = data.get("trends", {})
    ctx = data.get("context", {})

    pdf = BubblePDF(as_of=data.get("as_of", ""))
    tmpdir = tempfile.mkdtemp(prefix="bubble_pdf_")

    # ======================================================== PAGE 1
    pdf.add_page()
    y = 18

    # -- master score block (left) + gauge (right of it)
    pdf.set_draw_color(*BORDER)
    pdf.set_fill_color(*PANEL)
    pdf.rect(10, y, 92, 40, style="DF")
    pdf.set_xy(14, y + 3)
    pdf.set_font("Helvetica", style="B", size=8)
    pdf.set_text_color(*TXT)
    pdf.cell(60, 4, "OVERALL BUBBLE SCORE")
    pdf.set_xy(14, y + 9)
    pdf.set_font("Helvetica", style="B", size=30)
    pdf.set_text_color(*color)
    pdf.cell(40, 14, f"{master:.0f}")
    pdf.set_font("Helvetica", size=10)
    pdf.set_xy(14 + pdf.get_string_width(f"{master:.0f}") + 14, y + 16)
    pdf.set_text_color(*TXT2)
    pdf.cell(20, 6, "/100")
    pdf.set_xy(14, y + 25)
    pdf.set_font("Helvetica", style="B", size=11)
    pdf.set_text_color(*color)
    pdf.cell(60, 5, label)
    pdf.set_xy(14, y + 31)
    pdf.set_font("Helvetica", size=7)
    pdf.set_text_color(*TXT2)
    vix = ctx.get("vix")
    vix_txt = "N/A" if vix is None or vix != vix else f"{vix:.1f}"
    pdf.cell(84, 4, sanitize_text(
        f"Regime: {ctx.get('regime', 'N/A')}  |  VIX: {vix_txt}  |  "
        f"Fed: {ctx.get('fed_status', 'N/A')}"))
    pdf.set_xy(14, y + 35)
    pdf.cell(84, 4, sanitize_text(
        f"12M crash probability (heuristic): {data.get('crash_prob_12m', 0):.0f}%"))

    gauge_png = os.path.join(tmpdir, "gauge.png")
    pdf.rect(106, y, 94, 40, style="DF")
    if _render_gauge_png(master, gauge_png):
        pdf.image(gauge_png, x=126, y=y + 3, w=54)

    y += 46

    # -- history chart
    pdf.panel_title(10, y, 190, "BUBBLE SCORE HISTORY",
                    "computed daily from weighted component series")
    hist_png = os.path.join(tmpdir, "history.png")
    pdf.set_fill_color(*PANEL)
    pdf.set_draw_color(*BORDER)
    pdf.rect(10, y + 7, 190, 58, style="DF")
    if _render_history_png(data.get("history"), hist_png,
                           data.get("spx_history")):
        pdf.image(hist_png, x=12, y=y + 9, w=186)
    y += 71

    # -- components table
    pdf.panel_title(10, y, 190, "BUBBLE SCORE COMPONENTS",
                    "percentile ranks vs. all available history")
    y += 8
    order = [
        ("Valuation", "Valuation"), ("Momentum", "Price Momentum"),
        ("Market Excitement", "Market Excitement"), ("Credit", "Credit Expansion"),
        ("Liquidity", "Liquidity"), ("Profit Margins", "Profit Margins"),
        ("Concentration", "Concentration"), ("AI Bubble", "AI Bubble"),
    ]
    pdf.set_font("Helvetica", style="B", size=6.5)
    pdf.set_text_color(*TXT2)
    pdf.set_xy(10, y)
    for txt, w in [("COMPONENT", 48), ("SCORE", 16), ("LEVEL", 28),
                   ("1M TREND", 22), ("", 76)]:
        pdf.cell(w, 5, txt)
    y += 6
    for key, title in order:
        if key not in comps:
            continue
        score = comps[key]
        ccol, clabel = _risk_color_label(score)
        delta = trends.get(key, 0.0)
        tcol = RED if delta > 2 else GREEN if delta < -2 else TXT2
        pdf.set_xy(10, y)
        pdf.set_font("Helvetica", size=7.5)
        pdf.set_text_color(*TXT)
        pdf.cell(48, 6, sanitize_text(title))
        pdf.set_font("Helvetica", style="B", size=8)
        pdf.set_text_color(*ccol)
        pdf.cell(16, 6, f"{score:.0f}")
        pdf.set_font("Helvetica", style="B", size=6.5)
        pdf.cell(28, 6, clabel)
        pdf.set_text_color(*tcol)
        pdf.cell(22, 6, f"{delta:+.0f}")
        pdf.bar(126, y + 1.5, 72, 3, score, ccol)
        pdf.set_draw_color(*BORDER)
        pdf.line(10, y + 6, 200, y + 6)
        y += 6.5

    # ======================================================== PAGE 2
    pdf.add_page()
    y = 18

    # -- valuation snapshot (left) | breadth (right)
    pdf.panel_title(10, y, 110, "VALUATION SNAPSHOT", "live percentiles")
    pdf.panel_title(126, y, 74, "MARKET BREADTH & MOMENTUM")
    yy = y + 8
    for row in data.get("val_snapshot", [])[:8]:
        p = row["Percentile"]
        bcol = RED if p > 80 else ORANGE if p > 60 else YELLOW if p > 40 else GREEN
        pdf.set_xy(10, yy)
        pdf.set_font("Helvetica", size=6.8)
        pdf.set_text_color(*TXT)
        pdf.cell(44, 5, sanitize_text(row["Metric"])[:34])
        pdf.set_font("Helvetica", style="B", size=6.8)
        pdf.set_text_color(*bcol)
        pdf.cell(13, 5, str(row["Current"]))
        pdf.set_font("Helvetica", size=6.8)
        pdf.set_text_color(*TXT2)
        pdf.cell(13, 5, str(row["Historical Avg"]))
        pdf.set_text_color(*bcol)
        pdf.cell(9, 5, f"{p}%")
        pdf.bar(91, yy + 1.2, 28, 2.6, p, bcol)
        yy += 5.4

    br = data.get("breadth", {})
    by = y + 8
    if br.get("available"):
        stats = [
            ("% Above 200-DMA", f"{br.get('pct_above_200', 0):.0f}%"),
            ("New 52W Highs / Lows", f"{br.get('new_highs', 0)} / {br.get('new_lows', 0)}"),
            ("Advance / Decline", f"{br.get('adv_dec', 'N/A')}"),
            ("S&P Momentum (12M)", f"{br.get('momentum_12m', 0):.1f}%"),
        ]
        for name, val in stats:
            pdf.set_xy(126, by)
            pdf.set_font("Helvetica", size=6.8)
            pdf.set_text_color(*TXT2)
            pdf.cell(46, 5, name)
            pdf.set_font("Helvetica", style="B", size=6.8)
            pdf.set_text_color(*TXT)
            pdf.cell(28, 5, val, align="R")
            by += 5
        breadth_png = os.path.join(tmpdir, "breadth.png")
        if _render_breadth_png(br.get("pct_above_series"), breadth_png):
            pdf.image(breadth_png, x=126, y=by + 1, w=74)
            by += 32

    y = max(yy, by) + 6

    # -- liquidity (left) | US vs world (right)
    pdf.panel_title(10, y, 110, "LIQUIDITY & CREDIT CONDITIONS", "live: FRED")
    pdf.panel_title(126, y, 74, "VALUATION: US VS. WORLD")
    yy = y + 8
    status_col = {"TIGHT": RED, "NEUTRAL": YELLOW, "LOOSE": GREEN}
    for row in data.get("liquidity_snapshot", []):
        sc = status_col.get(row["Status"], TXT2)
        pdf.set_xy(10, yy)
        pdf.set_font("Helvetica", size=6.8)
        pdf.set_text_color(*TXT)
        pdf.cell(52, 5, sanitize_text(row["Indicator"]))
        pdf.set_font("Helvetica", style="B", size=6.8)
        pdf.cell(26, 5, sanitize_text(row["Level"]))
        pdf.set_text_color(*sc)
        pdf.cell(20, 5, row["Status"])
        yy += 5.2
    if not data.get("liquidity_snapshot"):
        pdf.set_xy(10, yy)
        pdf.set_font("Helvetica", size=6.8)
        pdf.set_text_color(*TXT2)
        pdf.cell(100, 5, "FRED data unavailable - add FRED_API_KEY.")
        yy += 6

    by = y + 8
    for r in data.get("us_vs_world", []):
        us, world = r["US"], r["World"]
        mx = max(us, world) or 1
        pdf.set_xy(126, by)
        pdf.set_font("Helvetica", size=6.5)
        pdf.set_text_color(*TXT2)
        pdf.cell(50, 4, sanitize_text(r["Metric"]))
        pdf.set_font("Helvetica", style="B", size=6.5)
        pdf.set_text_color(*TXT)
        pdf.cell(24, 4, f"{r['Ratio']}x", align="R")
        pdf.bar(126, by + 4.4, 74 * us / mx / 1.0, 2, 100, RED)
        pdf.bar(126, by + 7, 74 * world / mx / 1.0, 2, 100, BLUE)
        by += 11
    if data.get("us_vs_world"):
        pdf.set_xy(126, by)
        pdf.set_font("Helvetica", size=5.8)
        pdf.set_text_color(*TXT2)
        pdf.cell(74, 4, "Red = US (SPY)   Blue = World ex-US (EFA/VWO)")
        by += 5

    y = max(yy, by) + 6

    # -- asset class YTD (left) | allocation (right)
    pdf.panel_title(10, y, 110, "ASSET CLASS PERFORMANCE (YTD)")
    pdf.panel_title(126, y, 74, "SUGGESTED ALLOCATION (MODERATE)")
    yy = y + 8
    ytd_rows = data.get("asset_ytd", [])
    max_abs = max((abs(r["YTD"]) for r in ytd_rows), default=1) or 1
    for r in ytd_rows:
        v = r["YTD"]
        vcol = GREEN if v >= 0 else RED
        pdf.set_xy(10, yy)
        pdf.set_font("Helvetica", size=6.8)
        pdf.set_text_color(*TXT)
        pdf.cell(34, 5, sanitize_text(r["Asset"]))
        pdf.set_font("Helvetica", style="B", size=6.8)
        pdf.set_text_color(*vcol)
        pdf.cell(14, 5, f"{v:+.1f}%")
        pdf.bar(62, yy + 1.2, 56, 2.6, abs(v) / max_abs * 100, vcol)
        yy += 5.2

    by = y + 8
    alloc_colors = {"Equities": ORANGE, "International": BLUE,
                    "Treasuries": GREEN, "Gold": YELLOW, "Cash": TXT2}
    for k, v in data.get("allocation", {}).items():
        pdf.set_xy(126, by)
        pdf.set_font("Helvetica", size=6.8)
        pdf.set_text_color(*TXT2)
        pdf.cell(30, 5, k)
        pdf.set_font("Helvetica", style="B", size=6.8)
        pdf.set_text_color(*TXT)
        pdf.cell(10, 5, f"{v}%")
        pdf.bar(170, by + 1.2, 30, 2.6, v, alloc_colors.get(k, BLUE))
        by += 5.2

    y = max(yy, by) + 6

    # -- commentary
    pdf.panel_title(10, y, 190, "FRAMEWORK COMMENTARY", "auto-generated")
    pdf.set_xy(10, y + 7)
    pdf.set_font("Helvetica", size=7.5)
    pdf.set_text_color(*TXT)
    pdf.multi_cell(190, 4.2, sanitize_text(data.get("commentary", "")))
    y = pdf.get_y() + 5

    # -- interpretation legend
    pdf.panel_title(10, y, 190, "HOW TO INTERPRET")
    ly = y + 7
    legend = [("0-20 LOW RISK", GREEN, "Attractive valuations, normal risk."),
              ("21-40 MODERATE", GREEN, "Caution warranted, monitor closely."),
              ("41-60 ELEVATED", YELLOW, "Risk building, be selective."),
              ("61-80 HIGH", ORANGE, "High risk of mean reversion."),
              ("81-100 EXTREME", RED, "Extremely overvalued, high crash risk.")]
    for name, lcol, desc in legend:
        pdf.set_xy(10, ly)
        pdf.set_font("Helvetica", style="B", size=6.5)
        pdf.set_text_color(*lcol)
        pdf.cell(30, 4, name)
        pdf.set_font("Helvetica", size=6.5)
        pdf.set_text_color(*TXT2)
        pdf.cell(120, 4, desc)
        ly += 4.4

    # -- data sources
    dq = data.get("data_quality", {})
    if dq:
        ly += 3
        pdf.set_xy(10, ly)
        pdf.set_font("Helvetica", style="B", size=6.5)
        pdf.set_text_color(*TXT)
        pdf.cell(60, 4, "DATA SOURCES")
        ly += 4.5
        pdf.set_font("Helvetica", size=6)
        pdf.set_text_color(*TXT2)
        src_txt = "   |   ".join(f"{k}: {v}" for k, v in dq.items())
        pdf.set_xy(10, ly)
        pdf.multi_cell(190, 3.6, sanitize_text(src_txt))

    # cleanup temp pngs
    for f in os.listdir(tmpdir):
        try:
            os.remove(os.path.join(tmpdir, f))
        except OSError:
            pass
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass

    return bytes(pdf.output())
