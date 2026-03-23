"""HTML report generation for the Trading System Dashboard.

Generates a self-contained HTML file with embedded interactive Plotly charts,
metric cards, and tables. No external dependencies (kaleido/Chromium) needed.
"""
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


def _fig_to_html_div(fig: go.Figure, include_js: bool = False) -> str:
    """Convert a Plotly figure to an HTML <div> string.

    Set include_js=True for the first chart to embed plotly.js inline.
    """
    return pio.to_html(fig, full_html=False, include_plotlyjs=include_js)


def _metric_card(label: str, value: str) -> str:
    return (
        f'<div class="metric"><span class="metric-label">{label}</span>'
        f'<span class="metric-value">{value}</span></div>'
    )


def _html_table(df: pd.DataFrame, col_formats: dict | None = None) -> str:
    col_formats = col_formats or {}
    rows = "<tr>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>\n"
    for _, row in df.iterrows():
        cells = ""
        for col in df.columns:
            fmt = col_formats.get(col)
            text = fmt.format(row[col]) if fmt else str(row[col])
            cells += f"<td>{text}</td>"
        rows += f"<tr>{cells}</tr>\n"
    return f'<table class="data-table">{rows}</table>'


_CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px 40px;
       background: #fafafa; color: #333; }
h1 { text-align: center; margin-bottom: 4px; }
.subtitle { text-align: center; color: #888; margin-bottom: 30px; }
h2 { border-bottom: 2px solid #ddd; padding-bottom: 6px; margin-top: 40px; }
h3 { color: #555; margin-top: 24px; }
.metrics-row { display: flex; gap: 20px; margin: 12px 0 16px; }
.metric { background: #fff; border: 1px solid #e0e0e0; border-radius: 8px;
           padding: 12px 20px; flex: 1; text-align: center; }
.metric-label { display: block; font-size: 0.85em; color: #888; }
.metric-value { display: block; font-size: 1.3em; font-weight: 600; margin-top: 4px; }
.data-table { border-collapse: collapse; margin: 10px 0; }
.data-table th, .data-table td { border: 1px solid #ddd; padding: 6px 14px; text-align: center; }
.data-table th { background: #f0f0f0; font-weight: 600; }
.chart-block { margin: 16px 0; }
@media print { body { padding: 10px; } .chart-block { page-break-inside: avoid; } }
"""


def generate_report(
    eq_view: pd.DataFrame,
    eq_full: pd.DataFrame,
    tr_view: pd.DataFrame | None,
    live_date: pd.Timestamp,
    log_y: bool,
    dd_dist: pd.DataFrame,
    cur_dd: float,
    cur_dd_pct: float,
    pct_above_ceiling: float,
    pct_below_floor: float,
    x_col: str,
    equity_charts: list[go.Figure],
    trades_charts: list[go.Figure],
) -> str:
    """Generate a self-contained HTML report string.

    Parameters
    ----------
    equity_charts : list of 7 Plotly figures (equity curve through best fit)
    trades_charts : list of 2 Plotly figures (win pct, gain)
    """
    chart_names = [
        "System Equity Curve",
        "System Drawdown",
        "Rolling CAGR",
        "Equity Curve + Std Dev Bands",
        "Standard Deviation Bands (Consecutive Days)",
        "Rolling Volatility",
        "Best Fit Floor / Ceiling",
    ]

    date_min = eq_view["Date"].min().strftime("%Y-%m-%d")
    date_max = eq_view["Date"].max().strftime("%Y-%m-%d")
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    parts: list[str] = []
    parts.append(f"<h1>Trading System Performance Report</h1>")
    parts.append(
        f'<p class="subtitle">Period: {date_min} &mdash; {date_max} &nbsp;|&nbsp; '
        f"Generated: {generated}</p>"
    )

    # ── Equity Section ────────────────────────────────────────────────────────
    parts.append("<h2>Equity Curve Dashboard</h2>")

    first_chart = True
    for i, fig in enumerate(equity_charts):
        parts.append(f"<h3>{chart_names[i]}</h3>")
        # Embed plotly.js inline with the first chart so the file is self-contained
        parts.append(f'<div class="chart-block">{_fig_to_html_div(fig, include_js=first_chart)}</div>')
        first_chart = False

        # After drawdown chart — metrics + distribution table
        if i == 1:
            parts.append('<div class="metrics-row">')
            parts.append(_metric_card("Current Drawdown", f"{cur_dd:.2%}"))
            parts.append(_metric_card("Current DD Percentile", f"{cur_dd_pct:.1%}"))
            parts.append("</div>")
            parts.append(_html_table(dd_dist, {
                "percentile": "{:.0%}", "drawdown": "{:.4%}",
            }))

        # After best fit chart — ceiling / floor stats
        if i == 6:
            parts.append('<div class="metrics-row">')
            parts.append(_metric_card("% Time Above Ceiling", f"{pct_above_ceiling:.1%}"))
            parts.append(_metric_card("% Time Below Floor", f"{pct_below_floor:.1%}"))
            parts.append("</div>")

    # ── Trades Section ────────────────────────────────────────────────────────
    if trades_charts:
        trade_names = [
            "Rolling Win % + Standard Deviation",
            "Rolling Avg Gain % + Standard Deviation",
        ]
        parts.append("<h2>Trades Dashboard</h2>")
        for i, fig in enumerate(trades_charts):
            parts.append(f"<h3>{trade_names[i]}</h3>")
            parts.append(f'<div class="chart-block">{_fig_to_html_div(fig)}</div>')

    body = "\n".join(parts)

    html = (
        "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
        "<meta charset='utf-8'>\n"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
        "<title>Trading System Performance Report</title>\n"
        f"<style>{_CSS}</style>\n"
        "</head>\n<body>\n"
        f"{body}\n"
        "</body>\n</html>"
    )
    return html
