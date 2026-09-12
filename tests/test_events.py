"""Analytical fixtures test actual economic definitions and missing-data behavior."""
import numpy as np
import pandas as pd
import pytest

from src.events import assign_day0, build_event_panel, trading_sessions


@pytest.mark.parametrize("filing,accepted,expected,moved", [
    ("2021-07-02", "2021-07-02T19:59:59Z", "2021-07-02", False),
    ("2021-07-02", "2021-07-02T20:00:00Z", "2021-07-06", True),
    ("2021-07-03", "2021-07-03T20:01:00Z", "2021-07-06", False),
    ("2021-07-06", "2021-07-02T20:01:00Z", "2021-07-06", False),
    # Midnight UTC still belongs to the previous Eastern calendar date.
    ("2021-07-02", "2021-07-03T00:30:00Z", "2021-07-06", True),
    # Winter uses EST: 20:59 UTC is still before the fixed 16:00 cutoff.
    ("2021-01-15", "2021-01-15T20:59:59Z", "2021-01-15", False),
    ("2021-01-15", "2021-01-15T21:00:00Z", "2021-01-19", True),
    # Assignment expressly uses 16:00, even on the 13:00 early close.
    ("2021-11-26", "2021-11-26T19:00:00Z", "2021-11-26", False),
])
def test_eastern_cutoff_holidays_and_later_filing(filing, accepted, expected, moved):
    calendar = trading_sessions("2021-01-01", "2021-12-31")
    row = assign_day0(filing, accepted, calendar)
    assert row["day0"] == pd.Timestamp(expected)
    assert row["day0_moved"] is moved


def test_missing_acceptance_is_excluded_without_guessing():
    row = assign_day0("2021-07-02", None, trading_sessions("2021-07-01", "2021-07-10"))
    assert pd.isna(row["day0"])
    assert row["day0_reason"] == "missing_or_invalid_acceptance_timestamp"


@pytest.fixture
def panel_inputs():
    sessions = trading_sessions("2021-01-01", "2021-12-31")
    pos = 100
    daily = np.where(np.arange(len(sessions)) % 2, .012, -.007)
    benchmark_daily = np.repeat(.001, len(sessions))
    prices = pd.DataFrame({"FIRM": 20 * np.cumprod(1 + daily),
                           "SPY": 300 * np.cumprod(1 + benchmark_daily)}, index=sessions)
    # Nominal prices deliberately differ from total-return adjusted prices.
    raw = pd.DataFrame({"FIRM": np.repeat(5., len(sessions))}, index=sessions)
    volume = pd.DataFrame({"FIRM": np.arange(len(sessions)) + 1000.}, index=sessions)
    day = sessions[pos]
    meta = pd.DataFrame([{"accession": "a", "cik": "0000000001", "ticker": "FIRM", "form": "10-K",
        "filing_date": day, "acceptance_datetime": day.tz_localize("UTC") + pd.Timedelta(hours=15)}])
    shares = pd.DataFrame([{"accession": "a", "shares_outstanding": 100., "shares_as_of": day - pd.Timedelta(days=5),
        "shares_tag": "dei:EntityCommonStockSharesOutstanding", "shares_status": "ok", "shares_source_accession": "a"}])
    return dict(meta=meta, prices=prices, raw_prices=raw, volume=volume, shares=shares, sessions=sessions)


def test_exact_windows_buy_and_hold_and_nominal_controls(panel_inputs):
    row = build_event_panel(**panel_inputs).iloc[0]
    p = panel_inputs["prices"]
    returns = p.pct_change(fill_method=None)
    assert row["n_pre_vol_returns"] == 55
    assert row["n_post_vol_returns"] == 60
    assert row["n_pre_history_returns"] == row["n_post_history_returns"] == 60
    # Independent endpoint ratios test compounded excess returns, not summed differences.
    expected_event = p["FIRM"].iloc[103] / p["FIRM"].iloc[99] - p["SPY"].iloc[103] / p["SPY"].iloc[99]
    expected_prior = p["FIRM"].iloc[94] / p["FIRM"].iloc[39] - p["SPY"].iloc[94] / p["SPY"].iloc[39]
    assert row["event_return"] == pytest.approx(expected_event)
    assert row["prior_excess_return"] == pytest.approx(expected_prior)
    assert row["pre_volatility"] == pytest.approx(np.std(returns["FIRM"].iloc[40:95], ddof=1) * np.sqrt(252))
    assert row["post_volatility"] == pytest.approx(np.std(returns["FIRM"].iloc[104:164], ddof=1) * np.sqrt(252))
    assert row["price_day_minus1"] == 5
    assert row["log_size"] == pytest.approx(np.log(500))
    assert row["log_dollar_volume"] == pytest.approx(np.log(5 * np.mean(np.arange(40, 95) + 1000)))
    assert row["pass_day0_price"] and row["pass_return_history"] and row["pass_complete_windows"]
    assert row["pass_shares"] and row["pass_controls"]
    assert row["market_exclusion_reasons"] == ""


def test_missing_quote_is_not_filled_or_removed_from_offsets(panel_inputs):
    missing_day = panel_inputs["sessions"][102]  # event +2
    panel_inputs["prices"] = panel_inputs["prices"].drop(index=missing_day)
    row = build_event_panel(**panel_inputs).iloc[0]
    assert row["day0"] == panel_inputs["sessions"][100]
    assert row["n_event_returns"] == 2  # both +2 and +3 daily returns need the missing close
    assert pd.isna(row["event_return"])
    assert not row["pass_complete_windows"]
    assert not row["pass_return_history"]


def test_sixty_after_does_not_imply_complete_post_volatility_window(panel_inputs):
    panel_inputs["prices"].loc[panel_inputs["sessions"][163], "FIRM"] = np.nan
    row = build_event_panel(**panel_inputs).iloc[0]
    assert row["pass_return_history"]  # +1 through +60 all exist
    assert not row["pass_complete_windows"]  # +63 does not
    assert row["n_post_vol_returns"] == 59
    assert pd.isna(row["post_volatility"])


def test_pre_window_needs_day_minus61_close(panel_inputs):
    panel_inputs["prices"].loc[panel_inputs["sessions"][39], "FIRM"] = np.nan
    row = build_event_panel(**panel_inputs).iloc[0]
    assert row["n_pre_vol_returns"] == 54
    assert row["n_pre_history_returns"] == 59
    assert not row["pass_return_history"]
    assert pd.isna(row["prior_excess_return"])


def test_benchmark_gap_and_missing_volume_are_audited(panel_inputs):
    panel_inputs["prices"].loc[panel_inputs["sessions"][101], "SPY"] = np.nan
    panel_inputs["volume"].loc[panel_inputs["sessions"][50], "FIRM"] = np.nan
    row = build_event_panel(**panel_inputs).iloc[0]
    assert row["pass_return_history"]
    assert not row["pass_complete_windows"]
    assert row["n_dollar_volume_days"] == 54
    assert not row["pass_controls"]
    assert pd.isna(row["event_return"]) and pd.isna(row["log_dollar_volume"])


@pytest.mark.parametrize("column,value", [
    ("shares_tag", "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic"),
    ("shares_source_accession", "later-accession"),
    ("shares_as_of", "2022-01-01"),
    ("shares_outstanding", 0),
])
def test_ineligible_shares_do_not_enter_size(panel_inputs, column, value):
    panel_inputs["shares"][column] = value
    row = build_event_panel(**panel_inputs).iloc[0]
    assert not row["pass_shares"]
    assert pd.isna(row["log_size"])
    assert "missing_or_ineligible_filing_outstanding_shares" in row["market_exclusion_reasons"]


def test_nominal_three_dollar_rule_and_all_accessions_preserved(panel_inputs):
    other = panel_inputs["meta"].iloc[0].copy()
    other["accession"], other["ticker"] = "b", "UNAVAILABLE"
    panel_inputs["meta"] = pd.concat([panel_inputs["meta"], other.to_frame().T], ignore_index=True)
    panel_inputs["raw_prices"].loc[panel_inputs["sessions"][99], "FIRM"] = 2.99
    panel = build_event_panel(**panel_inputs).set_index("accession")
    assert set(panel.index) == {"a", "b"}
    assert not panel.loc["a", "pass_day0_price"]
    assert "missing_ticker_prices" in panel.loc["b", "market_exclusion_reasons"]
    assert not panel.loc["b", "pass_shares"]


def test_duplicate_accession_and_absent_benchmark_raise(panel_inputs):
    duplicate = {**panel_inputs, "meta": pd.concat([panel_inputs["meta"]] * 2)}
    with pytest.raises(ValueError, match="Duplicate accessions"):
        build_event_panel(**duplicate)
    panel_inputs["prices"] = panel_inputs["prices"].drop(columns="SPY")
    with pytest.raises(ValueError, match="Missing SPY"):
        build_event_panel(**panel_inputs)
