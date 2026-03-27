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

The uploader accepts multiple CSV flavours automatically:

| Feature | Supported formats |
|---------|-------------------|
| **Encoding** | UTF-8, CP-1252 (Windows/Euro sign €), Latin-1 — tried in order |
| **Date formats** | ISO (`2024-01-15`), `dd/mm/yyyy`, `yyyy/mm/dd`, Excel serial numbers |
| **Numeric values** | Plain (`12345.67`), currency-prefixed (`€12,345.67`, `$1,000`), parenthesised negatives (`(123.45)`) |
| **Percentage values** | Decimal (`0.21`) or string (`21.00%`), including parenthesised negatives (`(5.28%)`) |

**Equity Curve CSV** — must contain:
- A `Date` column (case-insensitive)
- One numeric equity column — `System Equity Curve` is detected by name; if the column has a different name the loader picks the first numeric column automatically

**Trades List CSV** — must contain columns (case-insensitive):
- `Trade` — trade sequence number (handles zero-padded strings like `001`)
- `DateIn` — entry date
- `DateOut` — exit date
- `Bars` — number of bars/days in the trade (may contain currency symbols in some exports — stripped automatically)
- `PctGain` — trade return as decimal (`0.21`), percentage string (`21.00%`), or parenthesised negative (`(5.28%)`)

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

## Calculation Reference

### Equity Curve Dashboard

#### Chart 1 — System Equity Curve

Plots the raw equity value over time. An optional vertical dashed line marks the **Live Start Date** selected in the sidebar. Supports linear or log Y-axis.

No additional calculations — the series is the `equity` column loaded from the CSV.

#### Chart 2 — System Drawdown

| Series | Formula | Notes |
|--------|---------|-------|
| **Drawdown** | `equity / cummax(equity) − 1` | Always ≤ 0; shaded red fill to zero |
| **Avg DD (Whole Period)** | `expanding mean` of drawdown | Running average since first date |
| **Upper 5Y DD Percentile** | Rolling 1260-day `quantile((100 − p) / 100)` | Shallower drawdowns; green dashed line. `p` = sidebar *Drawdown Percentile* (default 30) |
| **Lower 5Y DD Percentile** | Rolling 1260-day `quantile(p / 100)` | Deeper drawdowns; orange dashed line |

The 1260-day window corresponds to 5 years of trading days, matching the Excel reference tool. The window expands from day 1 until 1260 days of data are available (`min_periods=1`).

#### Drawdown Percentile Distribution (expander)

| Metric | Formula |
|--------|---------|
| **Percentile table** | `numpy.percentile` (linear interpolation) of the inverted drawdown series at 10 %, 15 %, … 95 % levels |
| **Current Drawdown** | Last value of the drawdown series |
| **Current DD Percentile** | Highest percentile level whose threshold drawdown is ≤ the current drawdown (XLOOKUP-style reverse lookup) |

#### Chart 3 — Rolling CAGR

| Series | Window (trading days) | Formula |
|--------|-----------------------|---------|
| 1Y CAGR | 252 | `(equity / equity[−252])^(1/1) − 1` |
| 3Y CAGR | 756 | `(equity / equity[−756])^(1/3) − 1` |
| 5Y CAGR | 1260 | `(equity / equity[−1260])^(1/5) − 1` |
| 7Y CAGR | 1764 | `(equity / equity[−1764])^(1/7) − 1` |
| 10Y CAGR | 2520 | `(equity / equity[−2520])^(1/10) − 1` |

Each series begins once enough history is available (NaN before that).

#### Chart 4 — Equity Curve + Std Dev Bands (Bollinger)

| Series | Formula | Notes |
|--------|---------|-------|
| **Equity Curve** | Raw equity | — |
| **MA Average** | `mean(MA25, MA50, MA100, MA200)` | Averages whichever MAs are available (`skipna=True`), so it starts at day 25 |
| **Upper / Lower Band** | `MA Average ± n × σ` | `σ` = population std dev of MA Average over a rolling 176-day window. `n` = sidebar *Bollinger Band Std Devs* |
| **Log-space Bands** | `exp(log(MA Average) ± n × σ_log)` | Used when the log-scale toggle is on, for symmetric visual display |

Moving averages: simple rolling means with windows of 25, 50, 100, and 200 days (`min_periods = window`).

#### Chart 5 — Standard Deviation Bands (Consecutive Days)

| Series | Formula |
|--------|---------|
| **Days Above Upper BB** | Consecutive count of days where `equity > bb_upper` (resets to 0 when condition breaks) |
| **Days Below Lower BB** | Consecutive count (negative) of days where `equity < bb_lower` |

Displayed as a stacked bar chart — green bars above zero, red bars below.

#### Chart 6 — Rolling Volatility (Annualized)

| Series | Window | Formula |
|--------|--------|---------|
| 25/50/100/200-Day Vol | 25/50/100/200 | `std(log_returns, window) × √252` |
| Long-term Median | Expanding from day 200 | `expanding median` of the 200-day vol series |

Log returns: `ln(equity_t / equity_{t−1})`.

#### Chart 7 — Best Fit Floor / Ceiling

| Series | Formula | Notes |
|--------|---------|-------|
| **Best Fit Line** | `exp(intercept + slope × days)` | OLS regression of `ln(equity)` on calendar day count (historical data only) |
| **Upper Channel** | `exp(intercept × (1 + p/1000) + slope × days)` | `p` = sidebar *Best Fit ± %* |
| **Lower Channel** | `exp(intercept × (1 − p/1000) + slope × days)` | — |

Statistics computed from historical data (≤ today): % of time above/below the best-fit line and above ceiling / below floor.

---

### Trades Dashboard

#### Chart 1 — Rolling Win % + Standard Deviation

| Series | Formula | Notes |
|--------|---------|-------|
| **Rolling 25 / 50 / 100 Win %** | `rolling sum(win) / window` | `win = 1` if `PctGain > 0`, else `0`. Min periods = window |
| **Average** | `mean(win_pct_25, win_pct_50, win_pct_100, win_pct_200)` | Requires all 4 windows to be non-NaN (`skipna=False`) |
| **Upper / Lower Band** | `Average ± n × σ` | `σ` = rolling 25-trade std dev of the Average series. `n` = sidebar *Trade Std Devs* |

#### Chart 2 — Rolling Avg Gain % + Standard Deviation

| Series | Formula | Notes |
|--------|---------|-------|
| **Rolling 25 / 50 / 100 / 200 Avg Gain** | `rolling mean(PctGain, window)` | Min periods = window |
| **Average** | `mean(gain_avg_25, …, gain_avg_200)` | Same `skipna=False` rule as Win % |
| **Upper / Lower Band** | `Average ± n × σ` | `σ` = rolling 25-trade std dev of the Average series |

---

### Data Sheet Tab

The **Data Sheet** tab exposes every intermediate calculation column for audit and verification. Columns are grouped by category with multi-select filters:

**Equity Calculations** groups: Core, Drawdown Bands, Rolling CAGR, Moving Averages, Bollinger Bands, Volatility, Best Fit Channel.

**Trades Calculations** groups: Core, Rolling Win %, Rolling Avg Gain.

Both sub-tabs include a CSV download button for the full computed dataset.
