import pandas as pd

TRADES_REQUIRED_COLS = ["Trade", "DateIn", "DateOut", "Bars", "PctGain"]


def load_trades(path):
    """Load trades list CSV, validate required columns, and parse dates."""
    df = pd.read_csv(path)
    missing = [c for c in TRADES_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required trades columns: {', '.join(missing)}")
    df = df[TRADES_REQUIRED_COLS].copy()
    df["DateIn"] = pd.to_datetime(df["DateIn"])
    df["DateOut"] = pd.to_datetime(df["DateOut"])
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
