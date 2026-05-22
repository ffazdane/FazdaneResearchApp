"""Excel export helpers for the Gamma Flip Line / GEX Engine."""

from __future__ import annotations

import io

import pandas as pd


def analysis_to_excel(analysis: dict) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        analysis["summary"].to_excel(writer, sheet_name="Summary", index=False)
        analysis["by_strike"].to_excel(writer, sheet_name="GEX by Strike", index=False)
        analysis["by_expiration"].to_excel(writer, sheet_name="GEX by Expiration", index=False)
        analysis["simulation"].to_excel(writer, sheet_name="Simulation", index=False)
        analysis["gex_rows"].to_excel(writer, sheet_name="Raw GEX Rows", index=False)
    return output.getvalue()

