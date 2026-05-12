"""Statistical analysis calculations for trade list data.

Implements two analysis scopes:
1. Statistical Health Check — significance tests, distribution, dependency, bootstrap
2. Fragility & Robustness — outlier stress, fat-tail, temporal, equity dip, significance decay
"""
import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ── Helper: Profit parsing ────────────────────────────────────────────────────

def _parse_profit(series: pd.Series) -> pd.Series:
    """Parse Profit column: strip currency symbols, handle parenthesised negatives."""
    import re
    s = series.astype(str).str.strip()
    neg = s.str.startswith("(") & s.str.endswith(")")
    s = s.str.replace("(", "", regex=False).str.replace(")", "", regex=False)
    s = s.apply(lambda v: re.sub(r"[^\d.\-eE+]", "", v) if isinstance(v, str) else v)
    result = pd.to_numeric(s, errors="coerce")
    result[neg] = -result[neg].abs()
    return result


# ── Load extended trades (includes Profit, PctMAE, PctMFE) ────────────────────

STATS_EXTRA_COLS = ["Profit", "PctMAE", "PctMFE"]


def load_trades_extended(path) -> pd.DataFrame:
    """Load trades CSV with extra columns needed for statistical analysis.

    Expects the base trades_calcs.load_trades() columns plus Profit, PctMAE, PctMFE.
    Returns a DataFrame with cleaned numeric columns.
    """
    from calculations.trades_calcs import _read_csv_flexible, _parse_pct_gain, TRADES_REQUIRED_COLS

    df = _read_csv_flexible(path)

    # Case-insensitive column matching
    col_map = {c.lower(): c for c in df.columns}
    all_needed = TRADES_REQUIRED_COLS + STATS_EXTRA_COLS
    for req in all_needed:
        if req not in df.columns and req.lower() in col_map:
            df = df.rename(columns={col_map[req.lower()]: req})

    # Only require PctGain; Profit/MAE/MFE are optional
    missing_base = [c for c in TRADES_REQUIRED_COLS if c not in df.columns]
    if missing_base:
        raise ValueError(f"Missing required columns: {', '.join(missing_base)}")

    df["PctGain"] = _parse_pct_gain(df["PctGain"])

    if "Profit" in df.columns:
        df["Profit"] = _parse_profit(df["Profit"])
    if "PctMAE" in df.columns:
        df["PctMAE"] = pd.to_numeric(df["PctMAE"], errors="coerce")
    if "PctMFE" in df.columns:
        df["PctMFE"] = pd.to_numeric(df["PctMFE"], errors="coerce")

    # Parse dates
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

    df["Trade"] = pd.to_numeric(df["Trade"], errors="coerce").astype(int)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SCOPE 1: Statistical Health Check
# ══════════════════════════════════════════════════════════════════════════════

def summary_metrics(returns: pd.Series) -> dict:
    """Core summary: N, win rate, mean, median, profit factor, skew, kurtosis."""
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf
    return {
        "n_trades": len(returns),
        "win_rate": (returns > 0).mean(),
        "mean_return": returns.mean(),
        "median_return": returns.median(),
        "std_return": returns.std(),
        "profit_factor": pf,
        "skewness": sp_stats.skew(returns, nan_policy="omit"),
        "kurtosis": sp_stats.kurtosis(returns, nan_policy="omit"),  # Fisher
        "sem": sp_stats.sem(returns, nan_policy="omit"),
    }


def significance_tests(returns: pd.Series) -> dict:
    """One-sample t-test, Wilcoxon signed-rank, binomial test."""
    t_stat, t_pval = sp_stats.ttest_1samp(returns.dropna(), 0)
    try:
        w_stat, w_pval = sp_stats.wilcoxon(returns.dropna(), zero_method="wilcox")
    except ValueError:
        w_stat, w_pval = np.nan, np.nan
    n_wins = int((returns > 0).sum())
    n_total = len(returns.dropna())
    binom_pval = sp_stats.binomtest(n_wins, n_total, 0.5).pvalue if n_total > 0 else np.nan
    return {
        "t_stat": t_stat, "t_pval": t_pval,
        "wilcoxon_stat": w_stat, "wilcoxon_pval": w_pval,
        "binom_pval": binom_pval,
    }


def jarque_bera_test(returns: pd.Series) -> dict:
    """Jarque-Bera normality test."""
    jb_stat, jb_pval = sp_stats.jarque_bera(returns.dropna())
    return {"jb_stat": jb_stat, "jb_pval": jb_pval}


def runs_test(returns: pd.Series) -> dict:
    """Wald-Wolfowitz runs test for independence of wins/losses."""
    binary = (returns > 0).astype(int).values
    n = len(binary)
    if n < 2:
        return {"runs": np.nan, "runs_z": np.nan, "runs_pval": np.nan}

    n1 = binary.sum()
    n0 = n - n1
    runs = 1 + np.sum(binary[1:] != binary[:-1])

    if n1 == 0 or n0 == 0:
        return {"runs": int(runs), "runs_z": np.nan, "runs_pval": np.nan}

    mu = 1 + (2 * n1 * n0) / n
    denom = n * n * (n - 1)
    var = (2 * n1 * n0 * (2 * n1 * n0 - n)) / denom
    if var <= 0:
        return {"runs": int(runs), "runs_z": np.nan, "runs_pval": np.nan}

    z = (runs - mu) / np.sqrt(var)
    pval = 2 * sp_stats.norm.sf(abs(z))
    return {"runs": int(runs), "runs_z": z, "runs_pval": pval}


def autocorrelation(returns: pd.Series, lags: list[int] | None = None) -> dict:
    """Autocorrelation at specified lags."""
    lags = lags or [1, 2, 5]
    r = returns.dropna().values
    n = len(r)
    r_mean = r.mean()
    denom = np.sum((r - r_mean) ** 2)
    result = {}
    for lag in lags:
        if lag >= n:
            result[f"acf_lag{lag}"] = np.nan
        else:
            num = np.sum((r[lag:] - r_mean) * (r[:-lag] - r_mean))
            result[f"acf_lag{lag}"] = num / denom if denom != 0 else np.nan
    return result


def bootstrap_mean_ci(returns: pd.Series, n_iter: int = 10_000,
                       ci: float = 0.95, seed: int = 42) -> dict:
    """Bootstrap 95% CI for the mean return."""
    rng = np.random.default_rng(seed)
    r = returns.dropna().values
    means = np.array([rng.choice(r, size=len(r), replace=True).mean()
                      for _ in range(n_iter)])
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(means, [100 * alpha, 100 * (1 - alpha)])
    return {"bootstrap_mean": means.mean(), "ci_lower": lo, "ci_upper": hi}


def compute_health_check(returns: pd.Series) -> dict:
    """Run all health-check analyses, return combined dict."""
    result = {}
    result.update(summary_metrics(returns))
    result.update(significance_tests(returns))
    result.update(jarque_bera_test(returns))
    result.update(runs_test(returns))
    result.update(autocorrelation(returns))
    result.update(bootstrap_mean_ci(returns))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SCOPE 2: Fragility & Robustness Analysis
# ══════════════════════════════════════════════════════════════════════════════

def _trades_per_year(df: pd.DataFrame) -> float:
    """Average number of trades per year."""
    if df.empty:
        return 0.0
    start = df["DateIn"].min()
    end = df["DateOut"].max()
    days = (end - start).days
    if days <= 0:
        return 0.0
    years = days / 365.25
    return len(df) / years


def _ann_return_from_trades(df: pd.DataFrame) -> float:
    """Annualized return = mean per-trade return × trades per year."""
    if df.empty:
        return 0.0
    tpy = _trades_per_year(df)
    return df["PctGain"].mean() * tpy


def _max_drawdown_from_profits(df: pd.DataFrame) -> float:
    """Max drawdown from cumulative Profit (dollar-based, additive).

    This avoids the compounding problem with per-position PctGain.
    """
    if "Profit" not in df.columns or df.empty:
        return np.nan
    cum_profit = df["Profit"].cumsum()
    peak = cum_profit.cummax()
    # Drawdown as fraction of peak equity (assuming starting equity)
    starting_equity = 100_000  # reasonable default
    equity = starting_equity + cum_profit
    eq_peak = equity.cummax()
    dd = (equity - eq_peak) / eq_peak
    return dd.min()


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    """Max drawdown from a series of per-trade returns.

    WARNING: Only valid if returns are non-overlapping sequential returns.
    For overlapping positions, use _max_drawdown_from_profits or equity curve.
    """
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()


def _cagr_from_equity(equity_df: pd.DataFrame) -> float:
    """CAGR from equity curve DataFrame."""
    if equity_df is None or equity_df.empty:
        return np.nan
    start_val = equity_df["equity"].iloc[0]
    end_val = equity_df["equity"].iloc[-1]
    days = (equity_df["Date"].iloc[-1] - equity_df["Date"].iloc[0]).days
    if days <= 0 or start_val <= 0:
        return np.nan
    years = days / 365.25
    return (end_val / start_val) ** (1 / years) - 1


def _cagr_from_profits(df: pd.DataFrame) -> float:
    """CAGR approximation from cumulative Profit column."""
    if "Profit" not in df.columns or df.empty:
        return np.nan
    starting_equity = 100_000
    total_profit = df["Profit"].sum()
    end_val = starting_equity + total_profit
    days = (df["DateOut"].max() - df["DateIn"].min()).days
    if days <= 0 or end_val <= 0:
        return np.nan
    years = days / 365.25
    return (end_val / starting_equity) ** (1 / years) - 1


def _sharpe_from_returns(returns: pd.Series,
                         trades_per_year: float = 252) -> float:
    """Annualised Sharpe ratio (risk-free = 0).

    Annualisation uses sqrt(trades_per_year) rather than
    sqrt(252) because we are working with per-trade returns,
    not daily returns.
    """
    if returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(trades_per_year)


def _profit_factor(returns: pd.Series) -> float:
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns <= 0].sum())
    return gross_profit / gross_loss if gross_loss > 0 else np.inf


def outlier_stress_test(df: pd.DataFrame, pct: float = 0.05,
                       equity_df: pd.DataFrame | None = None) -> dict:
    """Remove top pct% trades by Profit, compare pre/post metrics.

    Uses equity curve for max drawdown if available.
    """
    profit_col = "Profit" if "Profit" in df.columns else "PctGain"
    threshold = df[profit_col].quantile(1 - pct)
    df_post = df[df[profit_col] <= threshold].copy()

    # Max DD from equity curve (ground truth)
    if equity_df is not None and "equity" in equity_df.columns:
        eq = equity_df["equity"]
        eq_peak = eq.cummax()
        max_dd_pre = ((eq - eq_peak) / eq_peak).min()
    else:
        max_dd_pre = _max_drawdown_from_profits(df)

    # CAGR: always from Profit for apples-to-apples pre/post comparison
    cagr_pre = _cagr_from_profits(df)

    tpy = _trades_per_year(df)
    pre = {
        "cagr": cagr_pre,
        "mean_return": df["PctGain"].mean(),
        "max_dd": max_dd_pre,
        "sharpe": _sharpe_from_returns(df["PctGain"], tpy),
        "profit_factor": _profit_factor(df["PctGain"]),
        "n_trades": len(df),
    }
    tpy_post = _trades_per_year(df_post) if not df_post.empty else tpy
    post = {
        "cagr": _cagr_from_profits(df_post),
        "mean_return": df_post["PctGain"].mean() if not df_post.empty else 0.0,
        "max_dd": _max_drawdown_from_profits(df_post),
        "sharpe": _sharpe_from_returns(df_post["PctGain"], tpy_post),
        "profit_factor": _profit_factor(df_post["PctGain"]),
        "n_trades": len(df_post),
    }
    return {"pre": pre, "post": post, "removed": len(df) - len(df_post),
            "threshold": threshold}


def annual_returns(df: pd.DataFrame,
                   equity_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Net return by calendar year.

    If equity_df is provided (with Date and equity columns), computes true
    portfolio returns from the equity curve. Otherwise falls back to summing
    PctGain * Fraction as an approximation.
    """
    if equity_df is not None and "Date" in equity_df.columns and "equity" in equity_df.columns:
        eq = equity_df.copy()
        eq["year"] = eq["Date"].dt.year
        yearly_eq = eq.groupby("year").agg(
            start=("equity", "first"),
            end=("equity", "last"),
        ).reset_index()
        yearly_eq["net_return"] = yearly_eq["end"] / yearly_eq["start"] - 1

        # Count trades per year from trades df
        df_c = df.copy()
        df_c["year"] = df_c["DateOut"].dt.year
        trade_counts = df_c.groupby("year").agg(
            n_trades=("PctGain", "count"),
        ).reset_index()

        yearly = yearly_eq.merge(trade_counts, on="year", how="left")
        yearly["n_trades"] = yearly["n_trades"].fillna(0).astype(int)

        if "Profit" in df.columns:
            df_c2 = df.copy()
            df_c2["year"] = df_c2["DateOut"].dt.year
            profit_yr = df_c2.groupby("year")["Profit"].sum().reset_index()
            profit_yr.columns = ["year", "net_profit"]
            yearly = yearly.merge(profit_yr, on="year", how="left")
            yearly["net_profit"] = yearly["net_profit"].fillna(0)
        else:
            yearly["net_profit"] = yearly["end"] - yearly["start"]

        yearly.drop(columns=["start", "end"], inplace=True)
    else:
        # Fallback: approximate from trade data
        df = df.copy()
        df["year"] = df["DateOut"].dt.year

        if "Fraction" in df.columns:
            df["_port_return"] = df["PctGain"] * df["Fraction"]
        else:
            df["_port_return"] = df["PctGain"]

        agg_dict = {
            "net_return": ("_port_return", "sum"),
            "n_trades": ("PctGain", "count"),
        }
        if "Profit" in df.columns:
            agg_dict["net_profit"] = ("Profit", "sum")
        else:
            agg_dict["net_profit"] = ("_port_return", "sum")

        yearly = df.groupby("year").agg(**agg_dict).reset_index()

    mu = yearly["net_return"].mean()
    sigma = yearly["net_return"].std()
    yearly["z_score"] = (yearly["net_return"] - mu) / sigma if sigma > 0 else 0.0
    yearly["outlier_year"] = yearly["z_score"].abs() > 2
    return yearly


_CRISIS_PERIODS = {
    "GFC 2008": ("2007-10-01", "2009-03-31"),
    "COVID 2020": ("2020-02-01", "2020-06-30"),
    "Bear 2022": ("2022-01-01", "2022-10-31"),
}


def regime_performance(df: pd.DataFrame,
                       periods: dict | None = None,
                       equity_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Performance during specified crisis/regime periods."""
    periods = periods or _CRISIS_PERIODS
    rows = []
    for name, (start, end) in periods.items():
        mask = (df["DateIn"] >= start) & (df["DateOut"] <= end)
        sub = df[mask]
        if sub.empty:
            rows.append({"regime": name, "n_trades": 0, "net_return": np.nan,
                          "win_rate": np.nan, "profit_factor": np.nan})
            continue

        # Use equity curve for true portfolio return if available
        if equity_df is not None and "Date" in equity_df.columns and "equity" in equity_df.columns:
            eq_mask = (equity_df["Date"] >= start) & (equity_df["Date"] <= end)
            eq_sub = equity_df[eq_mask]
            if len(eq_sub) >= 2:
                port_ret = eq_sub["equity"].iloc[-1] / eq_sub["equity"].iloc[0] - 1
            else:
                port_ret = np.nan
        elif "Fraction" in sub.columns:
            port_ret = (sub["PctGain"] * sub["Fraction"]).sum()
        else:
            port_ret = sub["PctGain"].sum()

        rows.append({
            "regime": name,
            "n_trades": len(sub),
            "net_return": port_ret,
            "win_rate": (sub["PctGain"] > 0).mean(),
            "profit_factor": _profit_factor(sub["PctGain"]),
        })
    return pd.DataFrame(rows)


def profit_concentration(yearly: pd.DataFrame) -> dict:
    """Best year's contribution to total profit."""
    if yearly.empty or "net_profit" not in yearly.columns:
        return {"best_year": None, "contribution_pct": 0.0}
    total = yearly["net_profit"].sum()
    best_idx = yearly["net_profit"].idxmax()
    best = yearly.loc[best_idx]
    pct = best["net_profit"] / total if total != 0 else 0.0
    return {"best_year": int(best["year"]), "contribution_pct": pct}


def equity_dip_analysis(df: pd.DataFrame,
                       equity_df: pd.DataFrame | None = None) -> dict:
    """Compare trades taken in drawdown vs at equity peak.

    Uses actual equity curve if provided, otherwise falls back
    to Profit-based cumulative equity.
    """
    if equity_df is not None and "equity" in equity_df.columns:
        eq = equity_df.copy()
        cum = eq["equity"]
        peak = cum.cummax()
        in_dd = cum < peak

        # Map each trade to whether the portfolio was in DD at trade exit
        trade_dd_status = []
        for _, trade in df.iterrows():
            exit_date = trade["DateOut"]
            # Find closest equity date <= exit_date
            mask = eq["Date"] <= exit_date
            if mask.any():
                idx = eq.loc[mask].index[-1]
                trade_dd_status.append(in_dd.iloc[idx])
            else:
                trade_dd_status.append(False)
        trade_in_dd = pd.Series(trade_dd_status, index=df.index)
    elif "Profit" in df.columns:
        cum = df["Profit"].cumsum() + 100000  # starting equity
        peak = cum.cummax()
        in_dd = cum < peak
        trade_in_dd = in_dd
    else:
        # Last resort: approximate (flag but still compute)
        cum = (1 + df["PctGain"] * df.get("Fraction", pd.Series(1, index=df.index))).cumprod()
        peak = cum.cummax()
        in_dd = cum < peak
        trade_in_dd = in_dd

    dd_trades = df[trade_in_dd.values].copy()
    peak_trades = df[~trade_in_dd.values].copy()

    def _stats(sub):
        if sub.empty:
            return {"n": 0, "win_rate": np.nan, "avg_profit": np.nan,
                    "profit_factor": np.nan, "avg_mae": np.nan}
        result = {
            "n": len(sub),
            "win_rate": (sub["PctGain"] > 0).mean(),
            "avg_profit": sub["PctGain"].mean(),
            "profit_factor": _profit_factor(sub["PctGain"]),
        }
        if "PctMAE" in sub.columns:
            result["avg_mae"] = sub["PctMAE"].mean()
        else:
            result["avg_mae"] = np.nan
        return result

    # For charting, use actual equity curve if available
    if equity_df is not None and "equity" in equity_df.columns:
        chart_cum = equity_df["equity"]
        chart_peak = chart_cum.cummax()
        chart_in_dd = chart_cum < chart_peak
        chart_dates = equity_df["Date"]
    elif "Profit" in df.columns:
        chart_cum = cum
        chart_peak = peak
        chart_in_dd = in_dd
        chart_dates = df["DateOut"]
    else:
        chart_cum = cum
        chart_peak = peak
        chart_in_dd = in_dd
        chart_dates = df["DateOut"]

    return {
        "in_drawdown": _stats(dd_trades),
        "at_peak": _stats(peak_trades),
        "cum_equity": chart_cum.reset_index(drop=True),
        "peak_equity": chart_peak.reset_index(drop=True),
        "in_dd_mask": chart_in_dd.reset_index(drop=True),
        "chart_dates": chart_dates.reset_index(drop=True),
    }


def significance_decay(returns: pd.Series, pct: float = 0.05) -> dict:
    """T-test pre/post removing top pct% of trades."""
    r = returns.dropna()
    threshold = r.quantile(1 - pct)
    r_post = r[r <= threshold]

    t_pre, p_pre = sp_stats.ttest_1samp(r, 0)
    t_post, p_post = sp_stats.ttest_1samp(r_post, 0) if len(r_post) > 1 else (np.nan, np.nan)

    return {
        "pre_t": t_pre, "pre_p": p_pre,
        "post_t": t_post, "post_p": p_post,
        "significant_pre": p_pre < 0.05,
        "significant_post": p_post < 0.05 if not np.isnan(p_post) else False,
    }


def fragility_verdict(stress: dict, sig_decay: dict, dip: dict,
                      conc: dict) -> str:
    """Return FRAGILE or ROBUST with reasoning."""
    flags = []
    # Outlier dependency: if removing top 5% drops profit factor by >50%
    pre_pf = stress["pre"]["profit_factor"]
    post_pf = stress["post"]["profit_factor"]
    if pre_pf != np.inf and pre_pf > 0:
        pf_drop = (pre_pf - post_pf) / pre_pf
        if pf_drop > 0.5:
            flags.append(f"Profit factor drops {pf_drop:.0%} after removing top 5% outliers")

    # Significance decay
    if sig_decay["significant_pre"] and not sig_decay["significant_post"]:
        flags.append("Statistical significance lost after removing top 5% trades")

    # Concentration
    if conc.get("contribution_pct", 0) > 0.5:
        flags.append(f"Best year contributes {conc['contribution_pct']:.0%} of total profit")

    # DD performance collapse
    dd_wr = dip["in_drawdown"].get("win_rate", 0)
    pk_wr = dip["at_peak"].get("win_rate", 0)
    if not np.isnan(dd_wr) and not np.isnan(pk_wr) and (pk_wr - dd_wr) > 0.15:
        flags.append(f"Win rate drops {pk_wr - dd_wr:.0%} during drawdowns")

    if flags:
        verdict = "FRAGILE"
        detail = "Strategy shows fragility: " + "; ".join(flags) + "."
    else:
        verdict = "ROBUST"
        detail = ("Strategy shows consistent expectancy across distributions, "
                  "time periods, and equity states.")
    return verdict, detail
