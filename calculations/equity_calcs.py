import re

import numpy as np
import pandas as pd


EQUITY_REQUIRED_COLS = ["Date", "System Equity Curve"]

# Characters to strip from numeric values (currency symbols, whitespace)
_CURRENCY_RE = re.compile(r"[^\d.\-eE+]")


def _read_csv_flexible(path):
    """Try multiple encodings to read a CSV file."""
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            if hasattr(path, "seek"):
                path.seek(0)
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    if hasattr(path, "seek"):
        path.seek(0)
    return pd.read_csv(path, encoding="latin-1")


def _clean_numeric(series: pd.Series) -> pd.Series:
    """Strip currency symbols and parenthesised negatives, return float series."""
    s = series.astype(str).str.strip()
    # Handle parenthesised negatives: (123.45) -> -123.45
    neg_mask = s.str.startswith("(") & s.str.endswith(")")
    s = s.str.replace("(", "", regex=False).str.replace(")", "", regex=False)
    s = s.apply(lambda v: _CURRENCY_RE.sub("", v))
    result = pd.to_numeric(s, errors="coerce")
    result[neg_mask] = -result[neg_mask].abs()
    return result


def _detect_equity_column(df):
    """Find the equity value column by name or position."""
    if "System Equity Curve" in df.columns:
        return "System Equity Curve"
    # Accept any column (other than Date) that contains numeric-looking values
    non_date_cols = [c for c in df.columns if c.lower() != "date"]
    if len(non_date_cols) == 1:
        return non_date_cols[0]
    # Try each non-date column; pick the first that is mostly numeric after cleaning
    for col in non_date_cols:
        cleaned = _clean_numeric(df[col])
        if cleaned.notna().sum() > len(df) * 0.5:
            return col
    raise ValueError(
        "Cannot detect equity column. Ensure the CSV has a 'Date' column "
        "and one numeric equity column."
    )


def load_equity_curve(path):
    """Load equity curve CSV, validate required columns, and parse dates."""
    df = _read_csv_flexible(path)

    # Ensure a Date column exists (case-insensitive fallback)
    if "Date" not in df.columns:
        date_col = next((c for c in df.columns if c.lower() == "date"), None)
        if date_col:
            df = df.rename(columns={date_col: "Date"})
        else:
            raise ValueError("Missing required equity column: Date")

    eq_col = _detect_equity_column(df)
    df = df[["Date", eq_col]].copy()
    df = df.rename(columns={eq_col: "equity"})

    # Clean equity values (strip currency symbols etc.)
    df["equity"] = _clean_numeric(df["equity"])

    # Handle mixed date formats: ISO strings + Excel serial numbers
    parsed = pd.to_datetime(df["Date"], errors="coerce")
    mask = parsed.isna()
    if mask.any():
        serial = pd.to_numeric(df.loc[mask, "Date"], errors="coerce")
        excel_epoch = pd.Timestamp("1899-12-30")
        parsed.loc[mask] = excel_epoch + pd.to_timedelta(serial, unit="D")
    df["Date"] = parsed
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def compute_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    """Compute drawdown, running max, and drawdown averages."""
    df["highest_high"] = df["equity"].cummax()
    df["drawdown"] = df["equity"] / df["highest_high"] - 1
    df["dd_avg_whole"] = df["drawdown"].expanding().mean()
    df["dd_avg_252"] = df["drawdown"].rolling(window=252, min_periods=252).mean()
    return df


# Rolling window for DD percentile bands (5 years of trading days)
DD_PERCENTILE_WINDOW = 1260


def compute_dd_percentile_bands(df: pd.DataFrame, percentile: int) -> pd.DataFrame:
    """Compute rolling upper/lower drawdown percentile bands.

    Uses a 1260-day (5-year) rolling window, expanding until the window is full.
    Labels match Excel convention:
      dd_lower_pct = quantile(percentile/100) → deeper (more negative) drawdowns
      dd_upper_pct = quantile((100-percentile)/100) → shallower drawdowns
    """
    lower_q = percentile / 100.0
    upper_q = (100 - percentile) / 100.0
    dd = df["drawdown"]
    df["dd_lower_pct"] = dd.rolling(window=DD_PERCENTILE_WINDOW, min_periods=1).quantile(lower_q)
    df["dd_upper_pct"] = dd.rolling(window=DD_PERCENTILE_WINDOW, min_periods=1).quantile(upper_q)
    return df


def compute_dd_distribution(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """Compute drawdown percentile distribution table (10% to 100%).

    Returns (table_df, current_drawdown, current_dd_percentile).
    Current DD percentile is found by XLOOKUP-style lookup in the table:
    find the highest percentile whose drawdown threshold is <= current DD.
    """
    dd_series = df["drawdown"].dropna()
    dd_inverted = dd_series * -1
    pct_levels = [round(x / 100, 2) for x in range(10, 100, 5)]
    rows = []
    for p in pct_levels:
        val = np.percentile(dd_inverted, min(p * 100, 100), method="linear") * -1
        rows.append({"percentile": p, "drawdown": val})
    dist_df = pd.DataFrame(rows)

    # Current drawdown = last available value
    current_dd = dd_series.iloc[-1]
    # XLOOKUP match_mode=-1: find the highest percentile where dd_value <= current_dd
    current_dd_pct = dist_df.loc[dist_df["drawdown"] <= current_dd, "percentile"]
    current_dd_pct = current_dd_pct.iloc[0] if len(current_dd_pct) else 0.0
    return dist_df, current_dd, current_dd_pct


def compute_rolling_cagr(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling CAGR for 1, 3, 5, 7, 10 year windows (252 days/yr)."""
    for n in [1, 3, 5, 7, 10]:
        lookback = 252 * n
        shifted = df["equity"].shift(lookback)
        df[f"cagr_{n}y"] = (df["equity"] / shifted) ** (1 / n) - 1
    return df


def compute_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 25/50/100/200 day moving averages and their average.

    Excel uses AVERAGE(AD:AG) which skips blank cells, so the average
    starts from day 25 (just the 25-day MA), gradually adds 50/100/200
    as they become available.
    """
    for w in [25, 50, 100, 200]:
        df[f"ma_{w}"] = df["equity"].rolling(window=w, min_periods=w).mean()
    ma_cols = ["ma_25", "ma_50", "ma_100", "ma_200"]
    df["ma_avg"] = df[ma_cols].mean(axis=1, skipna=True)
    return df


def compute_bollinger_bands(df: pd.DataFrame, n_std: float) -> pd.DataFrame:
    """Compute Bollinger Bands on the MA average with consecutive tracking.

    Excel logic: STDEV.P of ma_avg over a rolling 176-period window.
    Bands: ma_avg +/- n_std * rolling_std.
    Also computes log-space bands for symmetric display on log-scale charts.
    Consecutive tracking: equity vs bands.
    """
    df["bb_std"] = df["ma_avg"].rolling(window=176, min_periods=176).std(ddof=0)
    df["bb_upper"] = df["ma_avg"] + n_std * df["bb_std"]
    df["bb_lower"] = df["ma_avg"] - n_std * df["bb_std"]

    # Log-space BB: symmetric bands when displayed on log-scale charts
    log_ma = np.log(df["ma_avg"])
    df["bb_std_log"] = log_ma.rolling(window=176, min_periods=176).std(ddof=0)
    df["bb_upper_log"] = np.exp(log_ma + n_std * df["bb_std_log"])
    df["bb_lower_log"] = np.exp(log_ma - n_std * df["bb_std_log"])

    # Consecutive days above upper / below lower band
    above = (df["equity"] > df["bb_upper"]).astype(int)
    below = (df["equity"] < df["bb_lower"]).astype(int) * -1

    consec_above = []
    consec_below = []
    ca = 0
    cb = 0
    for a, b in zip(above, below):
        ca = ca + 1 if a == 1 else 0
        cb = cb - 1 if b == -1 else 0
        consec_above.append(ca)
        consec_below.append(cb)

    df["days_above_bb"] = consec_above
    df["days_below_bb"] = consec_below
    return df


def compute_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns and rolling annualized volatility."""
    df["log_return"] = np.log(df["equity"] / df["equity"].shift(1))
    sqrt252 = np.sqrt(252)
    for w in [25, 50, 100, 200]:
        df[f"vol_{w}"] = df["log_return"].rolling(window=w, min_periods=w).std() * sqrt252
    df["vol_median"] = df["vol_200"].expanding(min_periods=200).median()
    return df


def compute_best_fit(df: pd.DataFrame, band_pct: float) -> pd.DataFrame:
    """Compute log-linear best fit line and upper/lower channels.

    band_pct: e.g., 33 means ±3.3% applied to the intercept multiplier.
    """
    today = pd.Timestamp.today().normalize()
    mask = df["Date"] <= today
    hist = df.loc[mask].copy()

    if len(hist) < 2:
        for col in ["best_fit", "best_fit_upper", "best_fit_lower"]:
            df[col] = np.nan
        return df

    days = (hist["Date"] - hist["Date"].iloc[0]).dt.days.values.astype(float)
    log_eq = np.log(hist["equity"].values)

    slope, intercept = np.polyfit(days, log_eq, 1)

    # Apply to full dataset
    all_days = (df["Date"] - df["Date"].iloc[0]).dt.days.values.astype(float)
    df["best_fit"] = np.exp(intercept + slope * all_days)

    # Upper/lower channels: shift intercept by ±band_pct/1000
    mult = band_pct / 1000.0
    df["best_fit_upper"] = np.exp(intercept * (1 + mult) + slope * all_days)
    df["best_fit_lower"] = np.exp(intercept * (1 - mult) + slope * all_days)

    # Stats: % of time above/below best fit line (historical only)
    df["above_fit"] = np.where(mask & (df["equity"] > df["best_fit"]), 1, 0)
    df["below_fit"] = np.where(mask & (df["equity"] < df["best_fit"]), 1, 0)

    # % of time above ceiling / below floor (historical only)
    hist_mask = mask.values
    n_hist = hist_mask.sum()
    df["above_ceiling"] = np.where(hist_mask & (df["equity"] > df["best_fit_upper"]), 1, 0)
    df["below_floor"] = np.where(hist_mask & (df["equity"] < df["best_fit_lower"]), 1, 0)
    df["pct_above_ceiling"] = df["above_ceiling"].cumsum() / np.maximum(np.cumsum(hist_mask), 1)
    df["pct_below_floor"] = df["below_floor"].cumsum() / np.maximum(np.cumsum(hist_mask), 1)

    return df


def compute_all(
    df: pd.DataFrame,
    dd_percentile: int = 30,
    bb_std: float = 2.0,
    band_pct: float = 33.0,
) -> pd.DataFrame:
    """Run all equity curve calculations in sequence."""
    df = compute_drawdown(df)
    df = compute_dd_percentile_bands(df, dd_percentile)
    df = compute_rolling_cagr(df)
    df = compute_moving_averages(df)
    df = compute_bollinger_bands(df, bb_std)
    df = compute_volatility(df)
    df = compute_best_fit(df, band_pct)
    return df
