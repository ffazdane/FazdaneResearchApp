import os
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF


LOGO_PATH = Path("assets") / "logo.png"


def _series_values(values):
    if values is None:
        return []
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def _render_plot_png(fig, output_path, title):
    """Render enough of a Plotly figure for the PDF without relying on Kaleido."""
    if fig is None:
        return False

    traces = list(getattr(fig, "data", []) or [])
    if not traces:
        return False

    plt.style.use("default")
    canvas, ax = plt.subplots(figsize=(7.8, 4.4), dpi=160)
    ax.set_facecolor("white")
    canvas.patch.set_facecolor("white")

    rendered = False
    for trace in traces:
        trace_type = getattr(trace, "type", "")
        name = getattr(trace, "name", "") or trace_type.title()
        x_values = _series_values(getattr(trace, "x", None))

        if trace_type == "candlestick":
            close_values = _series_values(getattr(trace, "close", None))
            if x_values and close_values:
                ax.plot(x_values, close_values, color="#1a3a8f", linewidth=1.6, label="Close")
                rendered = True
            continue

        y_values = _series_values(getattr(trace, "y", None))
        if x_values and y_values:
            mode = getattr(trace, "mode", "") or ""
            marker = "o" if "markers" in mode else None
            ax.plot(x_values, y_values, linewidth=1.7, marker=marker, markersize=3, label=name)
            rendered = True

    if not rendered:
        plt.close(canvas)
        return False

    try:
        layout_title = getattr(getattr(fig, "layout", None), "title", None)
        plot_title = getattr(layout_title, "text", None) or title
    except Exception:
        plot_title = title
    ax.set_title(sanitize_text(plot_title), fontsize=10, fontweight="bold")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.tick_params(axis="both", labelsize=7, colors="#111111")
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
    if len(ax.get_legend_handles_labels()[0]) > 1:
        ax.legend(fontsize=7, loc="best")
    canvas.autofmt_xdate(rotation=20)
    canvas.tight_layout()
    canvas.savefig(output_path, format="png", bbox_inches="tight")
    plt.close(canvas)
    return os.path.exists(output_path) and os.path.getsize(output_path) > 0


def sanitize_text(text):
    """Strip all non-ASCII chars that crash fpdf2 default fonts."""
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        "\u2014": "-", "\u2013": "-",
        "\u201c": '"', "\u201d": '"',
        "\u2018": "'", "\u2019": "'",
        "\u2026": "...",
        "\u00b1": "+/-",
        "\u2248": "~",
        "\u2265": ">=", "\u2264": "<=",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Final pass: drop anything outside printable ASCII
    return text.encode("ascii", "ignore").decode("ascii").strip()


class PDFReport(FPDF):
    def __init__(self, ticker, report_date):
        # Landscape A4: 297mm wide x 210mm tall
        super().__init__(orientation="L", format="A4")
        self.ticker = ticker
        self.report_date = report_date
        self.set_margins(10, 10, 10)
        self.set_auto_page_break(auto=False)

    def header(self):
        # Teal top bar
        self.set_fill_color(0, 173, 181)
        self.rect(0, 0, 297, 12, style="F")

        # Logo (if exists)
        if LOGO_PATH.exists():
            self.image(str(LOGO_PATH), x=275, y=1, w=18)

        # Company name in header bar
        self.set_font("Helvetica", style="B", size=14)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 1)
        self.cell(0, 10, "FazDane Analytics  |  Volatility Strategy Report", align="L")

        # Sub-header bar
        self.set_fill_color(20, 25, 35)
        self.rect(0, 12, 297, 7, style="F")
        self.set_font("Helvetica", size=8)
        self.set_text_color(160, 200, 210)
        self.set_xy(10, 13)
        self.cell(
            0, 5,
            f"Ticker: {self.ticker}   |   Generated: {self.report_date}   |   "
            "Question: Is this a good time to sell premium?",
            align="L"
        )
        self.ln(10)

    def footer(self):
        self.set_y(-8)
        self.set_font("Helvetica", style="I", size=7)
        self.set_text_color(150, 150, 150)
        self.cell(
            0, 5,
            "FazDane Analytics - Confidential | For informational purposes only. Not financial advice.",
            align="C"
        )


def generate_pdf_report(ticker, company_name, current_price, result, fig_vol, fig_term, table_rows=None):
    """
    Generates a landscape A4 1-pager PDF with:
      - Header bar with branding
      - LEFT column: Strategy recommendation box + key metrics table
      - RIGHT column: Two charts side by side
    Returns bytes.
    """
    report_date = datetime.now().strftime("%Y-%m-%d")
    pdf = PDFReport(ticker=ticker, report_date=report_date)
    pdf.add_page()

    # ── Layout constants (landscape A4 = 297 x 210 mm, margins 10mm each) ──
    # Usable area starts at y=22 (after header bars)
    PAGE_W = 277   # 297 - 10 - 10
    Y_START = 22
    LEFT_W = 120   # Left column width
    RIGHT_W = 153  # Right column width (297 - 10 - 120 - 4 gap - 10)
    LEFT_X = 10
    RIGHT_X = 134  # 10 + 120 + 4

    # ── LEFT COLUMN ──────────────────────────────────────────────────────────
    pdf.set_xy(LEFT_X, Y_START)

    # Ticker title
    pdf.set_font("Helvetica", style="B", size=13)
    pdf.set_text_color(15, 15, 15)
    pdf.cell(LEFT_W, 7, sanitize_text(f"{ticker}  -  {company_name}"), new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(100, 100, 100)
    pdf.set_x(LEFT_X)
    pdf.cell(LEFT_W, 5, sanitize_text(f"Current Price: ${current_price:,.2f}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # ── Strategy Decision Box ──
    pdf.set_x(LEFT_X)
    conf = sanitize_text(result.get("confidence", "N/A"))

    # Confidence color
    if conf == "High":
        box_color = (82, 214, 138)
    elif conf == "Medium":
        box_color = (220, 180, 50)
    else:
        box_color = (255, 107, 107)

    # Box background
    box_y = pdf.get_y()
    pdf.set_fill_color(240, 248, 252)
    pdf.set_draw_color(0, 173, 181)
    pdf.set_line_width(0.4)

    # Header row of box
    pdf.set_font("Helvetica", style="B", size=9)
    pdf.set_text_color(0, 130, 140)
    pdf.set_x(LEFT_X)
    pdf.cell(LEFT_W, 6, "STRATEGY RECOMMENDATION", border="TLR", fill=True, align="C", new_x="LMARGIN", new_y="NEXT")

    # Strategy name
    pdf.set_font("Helvetica", style="B", size=11)
    pdf.set_text_color(20, 20, 20)
    pdf.set_x(LEFT_X)
    pdf.cell(LEFT_W, 8, sanitize_text(result.get("strategy", "N/A")), border="LR", fill=True, align="C", new_x="LMARGIN", new_y="NEXT")

    # Confidence badge row
    pdf.set_x(LEFT_X)
    pdf.set_fill_color(*box_color)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", style="B", size=8)
    pdf.cell(LEFT_W // 2, 5, f"Confidence: {conf}", border="LB", fill=True, align="C")
    pdf.set_fill_color(240, 248, 252)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(LEFT_W - LEFT_W // 2, 5, sanitize_text(f"DTE: {result.get('dte_rec','N/A')}"), border="RB", fill=True, align="C", new_x="LMARGIN", new_y="NEXT")

    # Strike note row
    pdf.set_x(LEFT_X)
    pdf.set_font("Helvetica", size=8)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(LEFT_W, 5, sanitize_text(f"Strike: {result.get('strike_note','N/A')}"), border="LR", fill=True, align="C", new_x="LMARGIN", new_y="NEXT")

    # Reason / rationale
    pdf.set_x(LEFT_X)
    pdf.set_font("Helvetica", style="I", size=7.5)
    pdf.set_text_color(90, 90, 90)
    reason_text = sanitize_text(result.get("reason", ""))
    pdf.multi_cell(LEFT_W, 4, reason_text, border="BLR", fill=True, align="L")
    pdf.ln(3)

    # ── Key Metrics Table ──
    if table_rows:
        pdf.set_x(LEFT_X)
        pdf.set_font("Helvetica", style="B", size=8)
        pdf.set_text_color(0, 130, 140)
        pdf.cell(LEFT_W, 5, "KEY METRICS SUMMARY", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

        COL_METRIC = 42
        COL_VALUE = 30
        COL_INTERP = LEFT_W - COL_METRIC - COL_VALUE

        # Header
        pdf.set_x(LEFT_X)
        pdf.set_fill_color(220, 238, 242)
        pdf.set_draw_color(180, 200, 210)
        pdf.set_text_color(0, 110, 120)
        pdf.set_font("Helvetica", style="B", size=7)
        pdf.cell(COL_METRIC, 5, "Metric", border=1, fill=True)
        pdf.cell(COL_VALUE, 5, "Value", border=1, fill=True)
        pdf.cell(COL_INTERP, 5, "Signal", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

        # Rows — only show while we have vertical space
        for i, row in enumerate(table_rows):
            if pdf.get_y() > 195:
                break
            bg = (248, 252, 253) if i % 2 == 0 else (255, 255, 255)
            pdf.set_fill_color(*bg)
            pdf.set_draw_color(200, 215, 220)

            pdf.set_x(LEFT_X)
            pdf.set_text_color(70, 70, 70)
            pdf.set_font("Helvetica", size=6.5)
            pdf.cell(COL_METRIC, 4, sanitize_text(str(row[0])), border=1, fill=True)

            pdf.set_font("Helvetica", style="B", size=6.5)
            pdf.set_text_color(20, 20, 20)
            pdf.cell(COL_VALUE, 4, sanitize_text(str(row[1])), border=1, fill=True)

            pdf.set_font("Helvetica", size=6.5)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(COL_INTERP, 4, sanitize_text(str(row[2])), border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

    # ── RIGHT COLUMN: Two charts side by side ────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fv:
        vol_path = fv.name
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as ft:
        term_path = ft.name

    try:
        _render_plot_png(fig_vol, vol_path, "Volatility Snapshot")
        _render_plot_png(fig_term, term_path, "IV Term Structure")

        # Chart dimensions in the right column
        CHART_W = RIGHT_W // 2 - 1      # ~75mm each
        CHART_H = 80                     # 80mm tall

        # Labels
        pdf.set_font("Helvetica", style="B", size=8)
        pdf.set_text_color(0, 130, 140)
        pdf.set_xy(RIGHT_X, Y_START)
        pdf.cell(CHART_W, 5, "Volatility Cone", align="C")
        pdf.set_xy(RIGHT_X + CHART_W + 2, Y_START)
        pdf.cell(CHART_W, 5, "IV Term Structure", align="C")

        chart_y = Y_START + 5

        if os.path.exists(vol_path) and os.path.getsize(vol_path) > 0:
            pdf.image(vol_path, x=RIGHT_X, y=chart_y, w=CHART_W, h=CHART_H)

        if os.path.exists(term_path) and os.path.getsize(term_path) > 0:
            pdf.image(term_path, x=RIGHT_X + CHART_W + 2, y=chart_y, w=CHART_W, h=CHART_H)

    finally:
        for p in [vol_path, term_path]:
            if os.path.exists(p):
                os.remove(p)

    return bytes(pdf.output())
