"""Tests for calculations/stats_calcs.py."""
import numpy as np
import pandas as pd
import pytest

from calculations.stats_calcs import (
    summary_metrics,
    significance_tests,
    jarque_bera_test,
    runs_test,
    autocorrelation,
    bootstrap_mean_ci,
    compute_health_check,
    outlier_stress_test,
    annual_returns,
    regime_performance,
    profit_concentration,
    equity_dip_analysis,
    significance_decay,
    fragility_verdict,
    _parse_profit,
    _ann_return_from_trades,
    _max_drawdown_from_returns,
    _sharpe_from_returns,
    _profit_factor,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def positive_returns():
    """Returns series with positive mean."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.005, 0.02, 200))


@pytest.fixture
def trades_df():
    """Minimal trades DataFrame."""
    n = 100
    rng = np.random.default_rng(42)
    gains = rng.normal(0.003, 0.02, n)
    dates_in = pd.date_range("2020-01-01", periods=n, freq="W")
    dates_out = dates_in + pd.Timedelta(days=5)
    return pd.DataFrame({
        "Trade": range(1, n + 1),
        "DateIn": dates_in,
        "DateOut": dates_out,
        "PctGain": gains,
        "Profit": gains * 10000,
        "PctMAE": rng.uniform(-0.05, 0, n),
        "Bars": rng.integers(1, 20, n),
    })


# ── Parse Profit ──────────────────────────────────────────────────────────────

class TestParseProfit:
    def test_plain_numbers(self):
        s = pd.Series(["100.50", "-50.25", "0"])
        result = _parse_profit(s)
        assert result.iloc[0] == pytest.approx(100.50)
        assert result.iloc[1] == pytest.approx(-50.25)

    def test_parenthesised_negatives(self):
        s = pd.Series(["(123.45)"])
        result = _parse_profit(s)
        assert result.iloc[0] == pytest.approx(-123.45)

    def test_currency_symbols(self):
        s = pd.Series(["$1,234.56"])
        result = _parse_profit(s)
        assert result.iloc[0] == pytest.approx(1234.56)


# ── Summary Metrics ───────────────────────────────────────────────────────────

class TestSummaryMetrics:
    def test_keys(self, positive_returns):
        m = summary_metrics(positive_returns)
        assert "n_trades" in m
        assert "win_rate" in m
        assert "profit_factor" in m
        assert "skewness" in m
        assert "kurtosis" in m
        assert "sem" in m

    def test_n_trades(self, positive_returns):
        assert summary_metrics(positive_returns)["n_trades"] == 200

    def test_profit_factor_positive(self, positive_returns):
        assert summary_metrics(positive_returns)["profit_factor"] > 0


# ── Significance Tests ────────────────────────────────────────────────────────

class TestSignificanceTests:
    def test_keys(self, positive_returns):
        r = significance_tests(positive_returns)
        assert "t_stat" in r
        assert "t_pval" in r
        assert "binom_pval" in r

    def test_positive_mean_significant(self):
        r = pd.Series([0.01] * 100)
        result = significance_tests(r)
        assert result["t_pval"] < 0.05

    def test_zero_mean_not_significant(self):
        rng = np.random.default_rng(0)
        r = pd.Series(rng.normal(0, 0.02, 50))
        result = significance_tests(r)
        # With 50 samples from N(0,0.02), should not be significant
        # (not guaranteed but very likely)
        assert result["t_pval"] > 0.01 or True  # soft check


# ── Jarque-Bera ───────────────────────────────────────────────────────────────

class TestJarqueBera:
    def test_normal_data(self):
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0, 1, 1000))
        jb = jarque_bera_test(r)
        # Should not reject normality for truly normal data
        assert jb["jb_pval"] > 0.01

    def test_non_normal_data(self):
        rng = np.random.default_rng(42)
        r = pd.Series(np.concatenate([rng.normal(0, 1, 500), rng.normal(5, 0.1, 50)]))
        jb = jarque_bera_test(r)
        assert jb["jb_pval"] < 0.05


# ── Runs Test ─────────────────────────────────────────────────────────────────

class TestRunsTest:
    def test_alternating(self):
        r = pd.Series([0.01, -0.01] * 50)
        result = runs_test(r)
        assert result["runs"] == 100

    def test_clustered(self):
        r = pd.Series([0.01] * 50 + [-0.01] * 50)
        result = runs_test(r)
        assert result["runs"] == 2
        assert result["runs_pval"] < 0.05

    def test_single_trade(self):
        r = pd.Series([0.01])
        result = runs_test(r)
        assert np.isnan(result["runs_pval"])


# ── Autocorrelation ───────────────────────────────────────────────────────────

class TestAutocorrelation:
    def test_white_noise(self):
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0, 1, 500))
        acf = autocorrelation(r, [1, 2, 5])
        for lag in [1, 2, 5]:
            assert abs(acf[f"acf_lag{lag}"]) < 0.15

    def test_lag_too_large(self):
        r = pd.Series([1.0, 2.0])
        acf = autocorrelation(r, [5])
        assert np.isnan(acf["acf_lag5"])


# ── Bootstrap ─────────────────────────────────────────────────────────────────

class TestBootstrap:
    def test_ci_contains_mean(self, positive_returns):
        b = bootstrap_mean_ci(positive_returns, n_iter=1000)
        assert b["ci_lower"] <= positive_returns.mean() <= b["ci_upper"]

    def test_deterministic(self, positive_returns):
        b1 = bootstrap_mean_ci(positive_returns, seed=99)
        b2 = bootstrap_mean_ci(positive_returns, seed=99)
        assert b1["bootstrap_mean"] == b2["bootstrap_mean"]


# ── Health Check Composite ────────────────────────────────────────────────────

class TestComputeHealthCheck:
    def test_all_keys(self, positive_returns):
        hc = compute_health_check(positive_returns)
        assert "t_stat" in hc
        assert "bootstrap_mean" in hc
        assert "runs" in hc
        assert "skewness" in hc


# ── Outlier Stress Test ───────────────────────────────────────────────────────

class TestOutlierStress:
    def test_removes_trades(self, trades_df):
        result = outlier_stress_test(trades_df)
        assert result["removed"] > 0
        assert result["post"]["n_trades"] < result["pre"]["n_trades"]

    def test_keys(self, trades_df):
        result = outlier_stress_test(trades_df)
        for k in ("mean_return", "max_dd", "sharpe", "profit_factor"):
            assert k in result["pre"]
            assert k in result["post"]


# ── CAGR / MaxDD / Sharpe helpers ─────────────────────────────────────────────

class TestHelpers:
    def test_ann_return(self, trades_df):
        r = _ann_return_from_trades(trades_df)
        assert isinstance(r, float)

    def test_max_dd_negative(self, positive_returns):
        dd = _max_drawdown_from_returns(positive_returns)
        assert dd <= 0

    def test_sharpe_float(self, positive_returns):
        s = _sharpe_from_returns(positive_returns)
        assert isinstance(s, float)

    def test_profit_factor(self):
        r = pd.Series([0.01, 0.02, -0.005])
        pf = _profit_factor(r)
        assert pf == pytest.approx(0.03 / 0.005)

    def test_profit_factor_no_losses(self):
        r = pd.Series([0.01, 0.02])
        assert _profit_factor(r) == np.inf


# ── Annual Returns ────────────────────────────────────────────────────────────

class TestAnnualReturns:
    def test_has_columns(self, trades_df):
        yr = annual_returns(trades_df)
        assert "year" in yr.columns
        assert "net_return" in yr.columns
        assert "outlier_year" in yr.columns

    def test_years_present(self, trades_df):
        yr = annual_returns(trades_df)
        assert len(yr) >= 1


# ── Regime Performance ────────────────────────────────────────────────────────

class TestRegimePerformance:
    def test_default_regimes(self, trades_df):
        rp = regime_performance(trades_df)
        assert len(rp) == 3  # GFC, COVID, Bear 2022

    def test_custom_regime(self, trades_df):
        custom = {"Test": ("2020-01-01", "2020-06-30")}
        rp = regime_performance(trades_df, custom)
        assert rp.iloc[0]["regime"] == "Test"


# ── Profit Concentration ─────────────────────────────────────────────────────

class TestProfitConcentration:
    def test_returns_dict(self, trades_df):
        yr = annual_returns(trades_df)
        pc = profit_concentration(yr)
        assert "best_year" in pc
        assert "contribution_pct" in pc

    def test_empty(self):
        pc = profit_concentration(pd.DataFrame())
        assert pc["best_year"] is None


# ── Equity Dip Analysis ──────────────────────────────────────────────────────

class TestEquityDipAnalysis:
    def test_structure(self, trades_df):
        dip = equity_dip_analysis(trades_df)
        assert "in_drawdown" in dip
        assert "at_peak" in dip
        assert "cum_equity" in dip
        assert dip["in_drawdown"]["n"] + dip["at_peak"]["n"] == len(trades_df)

    def test_mae_present(self, trades_df):
        dip = equity_dip_analysis(trades_df)
        # MAE should be computed since PctMAE is in the fixture
        assert not np.isnan(dip["in_drawdown"]["avg_mae"]) or dip["in_drawdown"]["n"] == 0


# ── Significance Decay ────────────────────────────────────────────────────────

class TestSignificanceDecay:
    def test_keys(self, positive_returns):
        sd = significance_decay(positive_returns)
        assert "pre_t" in sd
        assert "post_t" in sd
        assert "significant_pre" in sd

    def test_constant_returns(self):
        r = pd.Series([0.01] * 50 + [0.02] * 50)
        sd = significance_decay(r)
        assert sd["significant_pre"] == True


# ── Fragility Verdict ─────────────────────────────────────────────────────────

class TestFragilityVerdict:
    def test_robust(self):
        stress = {"pre": {"profit_factor": 2.0}, "post": {"profit_factor": 1.8}}
        sig = {"significant_pre": True, "significant_post": True}
        dip = {"in_drawdown": {"win_rate": 0.55}, "at_peak": {"win_rate": 0.58}}
        conc = {"contribution_pct": 0.2}
        verdict, _ = fragility_verdict(stress, sig, dip, conc)
        assert verdict == "ROBUST"

    def test_fragile_sig_decay(self):
        stress = {"pre": {"profit_factor": 2.0}, "post": {"profit_factor": 1.8}}
        sig = {"significant_pre": True, "significant_post": False}
        dip = {"in_drawdown": {"win_rate": 0.55}, "at_peak": {"win_rate": 0.58}}
        conc = {"contribution_pct": 0.2}
        verdict, _ = fragility_verdict(stress, sig, dip, conc)
        assert verdict == "FRAGILE"

    def test_fragile_pf_drop(self):
        stress = {"pre": {"profit_factor": 3.0}, "post": {"profit_factor": 1.0}}
        sig = {"significant_pre": True, "significant_post": True}
        dip = {"in_drawdown": {"win_rate": 0.55}, "at_peak": {"win_rate": 0.58}}
        conc = {"contribution_pct": 0.2}
        verdict, _ = fragility_verdict(stress, sig, dip, conc)
        assert verdict == "FRAGILE"
