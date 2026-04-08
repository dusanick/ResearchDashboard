import re

import pandas as pd

TRADES_REQUIRED_COLS = ["Trade", "DateIn", "DateOut", "Bars", "PctGain"]

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


def _parse_pct_gain(series: pd.Series) -> pd.Series:
    """Parse PctGain that may be decimal (0.21) or string percent (21.00%)."""
    s = series.astype(str).str.strip()
    # Handle parenthesised negatives: (5.28%) -> -5.28%
    neg_mask = s.str.startswith("(") & s.str.endswith(")")
    s = s.str.replace("(", "", regex=False).str.replace(")", "", regex=False)
    has_pct = s.str.contains("%", na=False)

    # Remove % sign, strip currency symbols
    s_clean = s.str.replace("%", "", regex=False)
    s_clean = s_clean.apply(lambda v: _CURRENCY_RE.sub("", v) if isinstance(v, str) else v)
    result = pd.to_numeric(s_clean, errors="coerce")

    # If value was a percentage string, divide by 100
    result[has_pct] = result[has_pct] / 100.0
    result[neg_mask] = -result[neg_mask].abs()
    return result


def load_trades(path):
    """Load trades list CSV, validate required columns, and parse dates."""
    df = _read_csv_flexible(path)

    # Case-insensitive column matching
    col_map = {c.lower(): c for c in df.columns}
    for req in TRADES_REQUIRED_COLS:
        if req not in df.columns and req.lower() in col_map:
            df = df.rename(columns={col_map[req.lower()]: req})

    missing = [c for c in TRADES_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required trades columns: {', '.join(missing)}")
    df = df[TRADES_REQUIRED_COLS].copy()

    # Parse PctGain (handles both decimal and percentage string formats)
    df["PctGain"] = _parse_pct_gain(df["PctGain"])

    # Parse Trade number (handles zero-padded strings)
    df["Trade"] = pd.to_numeric(df["Trade"], errors="coerce").astype(int)

    # Parse Bars (may have currency symbols in some exports)
    df["Bars"] = pd.to_numeric(df["Bars"], errors="coerce")

    # Robust date parsing (mixed formats + Excel serial numbers)
    _excel_epoch = pd.Timestamp("1899-12-30")
    for col in ("DateIn", "DateOut"):
        parsed = pd.to_datetime(df[col].astype(str).str.strip(),
                                format="mixed", dayfirst=False, errors="coerce")
        mask = parsed.isna()
        if mask.any():
            nums = pd.to_numeric(df.loc[mask, col], errors="coerce")
            valid = nums.notna()
            if valid.any():
                parsed.loc[nums[valid].index] = (
                    _excel_epoch + pd.to_timedelta(nums[valid], unit="D")
                )
        df[col] = parsed
    return df


def compute_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the core columns needed for calculations."""
    df = df.copy()
    df["trade_num"] = df["Trade"]
    df["pct_gain"] = df["PctGain"]
    df["bars"] = df["Bars"]
    df["win"] = (df["pct_gain"] > 0).astype(int)
    return df


def compute_rolling_win_pct(df: pd.DataFrame, n_std: float) -> pd.DataFrame:
    """Compute rolling win % over 25/50/100/200 trade windows + std dev bands."""
    wins = df["win"]
    for w in [25, 50, 100, 200]:
        rolling_sum = wins.rolling(window=w, min_periods=w).sum()
        rolling_count = wins.rolling(window=w, min_periods=w).count()
        df[f"win_pct_{w}"] = rolling_sum / rolling_count

    wp_cols = ["win_pct_25", "win_pct_50", "win_pct_100", "win_pct_200"]
    df["win_pct_avg"] = df[wp_cols].mean(axis=1, skipna=False)

    # Std dev of the average over a 25-trade rolling window
    df["win_pct_std"] = df["win_pct_avg"].rolling(window=25, min_periods=25).std()
    df["win_pct_upper"] = df["win_pct_avg"] + n_std * df["win_pct_std"]
    df["win_pct_lower"] = df["win_pct_avg"] - n_std * df["win_pct_std"]
    return df


def compute_rolling_gain(df: pd.DataFrame, n_std: float) -> pd.DataFrame:
    """Compute rolling avg gain % over 25/50/100/200 trade windows + std dev bands."""
    gains = df["pct_gain"]
    for w in [25, 50, 100, 200]:
        df[f"gain_avg_{w}"] = gains.rolling(window=w, min_periods=w).mean()

    ga_cols = ["gain_avg_25", "gain_avg_50", "gain_avg_100", "gain_avg_200"]
    df["gain_avg"] = df[ga_cols].mean(axis=1, skipna=False)

    # Std dev of the average over a 25-trade rolling window
    df["gain_std"] = df["gain_avg"].rolling(window=25, min_periods=25).std()
    df["gain_upper"] = df["gain_avg"] + n_std * df["gain_std"]
    df["gain_lower"] = df["gain_avg"] - n_std * df["gain_std"]
    return df


def compute_all(df: pd.DataFrame, trade_std: float = 4.0) -> pd.DataFrame:
    """Run all trades calculations in sequence."""
    df = compute_base_columns(df)
    df = compute_rolling_win_pct(df, trade_std)
    df = compute_rolling_gain(df, trade_std)
    return df
