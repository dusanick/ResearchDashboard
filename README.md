# Trading System Performance Dashboard

A Streamlit-based research dashboard that replicates and extends the analysis from an Excel-based trading system performance tool. It provides interactive equity curve analysis, trade-level statistics, and exportable reports.

## Features

- **Equity Curve Dashboard** — 7 interactive Plotly charts covering equity growth, drawdowns, rolling CAGR, Bollinger bands, volatility, and best-fit channel analysis
- **Trades Dashboard** — Rolling win rate and average gain with configurable standard deviation bands
- **CSV Upload** — Drag-and-drop your own equity curve or trades list with automatic column validation
- **Date Range Zoom** — Slider to focus on any sub-period, with optional full recalculation
- **Report Export** — Two export modes:
  - **Interactive Report** (~11 MB) — Self-contained HTML with embedded Plotly charts (zoom, hover, pan)
  - **Static Report** (~1 MB) — Lightweight HTML with matplotlib PNG images, suitable for sharing

## Project Structure

```
TDashboard/
├── app.py                          # Streamlit entry point
├── requirements.txt                # Direct dependencies
├── .streamlit/config.toml          # Streamlit config (telemetry off)
├── calculations/
│   ├── equity_calcs.py             # Equity curve metrics (DD, CAGR, BB, volatility, best fit)
│   └── trades_calcs.py             # Trade-level metrics (win %, gain, rolling stats)
├── charts/
│   ├── equity_charts.py            # 7 Plotly equity charts
│   └── trades_charts.py            # 2 Plotly trades charts
├── reports/
│   ├── export_html.py              # Interactive HTML report (Plotly)
│   └── export_static.py            # Static HTML report (matplotlib)
├── data/
│   ├── equity_curve.csv            # Default equity curve data
│   └── trades_list.csv             # Default trades list data
├── tests/
│   ├── test_equity_calcs.py        # 32 tests
│   ├── test_trades_calcs.py        # 23 tests
│   ├── test_charts.py              # 26 tests
│   ├── test_export_html.py         # 28 tests
│   └── test_export_static.py       # 22 tests
└── Docu/
    ├── System_Performance_Tool.xlsx # Reference Excel template
    └── Your Trading System Might Be Broken.._.pdf
```

## Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
python -m pip install -r requirements.txt
```

## Usage

```bash
python -m streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

### Sidebar Controls

| Section | Parameter | Description |
|---------|-----------|-------------|
| **Data Source** | Upload CSVs | Override default data with your own files |
| **Equity Params** | Live Start Date | Vertical marker on charts |
| | Drawdown Percentile | Band width for DD distribution (1–50) |
| | Bollinger Band Std Devs | Standard deviations for BB (0.5–10) |
| | Best Fit ± % | Channel width around regression line |
| | Log Scale | Toggle log/linear y-axis |
| | Date Range Slider | Zoom into a sub-period |
| | Recalculate | Recompute all metrics on the selected slice |
| **Trades Params** | Trade Std Devs | Band width for win % and gain charts |
| **Export** | Interactive Report | Full Plotly HTML export |
| | Static Report | Lightweight matplotlib PNG export |

### CSV Format

**Equity Curve CSV** — must contain columns:
- `Date` — date in ISO or Excel serial format
- `System Equity Curve` — equity value

**Trades List CSV** — must contain columns:
- `Trade`, `DateIn`, `DateOut`, `Bars`, `PctGain`

Extra columns are ignored. Column order does not matter.

## Tests

```bash
python -m pytest tests/ -v
```

131 tests covering calculations, charts, and both export modules.

## Dependencies

| Package | Purpose |
|---------|---------|
| streamlit | Web dashboard framework |
| pandas | Data manipulation |
| numpy | Numerical calculations |
| plotly | Interactive charts + HTML export |
| matplotlib | Static chart image rendering |
| openpyxl | Excel file reading |
| pytest | Unit testing |
