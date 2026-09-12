"""Assignment event calendar, explicit windows, controls and exclusion audit.

Returns are simple daily total returns, with no filling or deletion of missing
quotes. All offsets refer to a common NYSE calendar. Pre volatility uses 55
daily returns [-60,-6]; post uses 60 [+4,+63], both sample SD times sqrt(252).
The separate README history requirement uses 60 returns [-60,-1] and [+1,+60].
The filing-period excess buy-and-hold return is stock BH minus SPY BH [0,+3],
each measured from close -1. Prior excess return uses the same construction
over [-60,-6], measured from close -61. Log dollar volume is log of the mean
nominal price times nominal volume across the 55 pre-window sessions.

The fixed 16:00 Eastern cutoff is the assignment rule, including early-close
sessions. Day0 moved compares with the first session on/after filing_date, so
an acceptance delay that leaves the same eventual Monday is not counted twice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .market import SHARE_TAGS, _daily

COMMON_CONTROLS = ["log_size", "log_dollar_volume", "prior_excess_return", "pre_volatility", "is_10k"]


def trading_sessions(start, end) -> pd.DatetimeIndex:
    """NYSE exchange sessions (holidays included; no weekday approximation)."""
    import pandas_market_calendars as mcal
    return pd.DatetimeIndex(mcal.get_calendar("NYSE").valid_days(start_date=start, end_date=end)).tz_localize(None)


def _session_index(sessions) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(sessions)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    idx = idx.normalize()
    if idx.empty or idx.has_duplicates or not idx.is_monotonic_increasing:
        raise ValueError("Trading sessions must be nonempty, unique and ascending")
    return idx


def assign_day0(filing_date, acceptance_datetime, sessions) -> dict:
    """Convert acceptance UTC to New York, shift at >=16, then max with filing.

    A missing/invalid acceptance timestamp is not silently replaced by a filing
    date. Naive input timestamps are interpreted as UTC, matching EDGAR schema.
    """
    sessions = _session_index(sessions)
    filing = pd.to_datetime(filing_date, errors="coerce")
    accepted = pd.to_datetime(acceptance_datetime, errors="coerce", utc=True)
    empty = {"day0": pd.NaT, "filing_date_day0": pd.NaT, "day0_moved": False,
             "acceptance_eastern": None, "acceptance_after_close": False,
             "day0_candidate_date": pd.NaT, "day0_reason": ""}
    if pd.isna(filing):
        return {**empty, "day0_reason": "invalid_filing_date"}
    filing = pd.Timestamp(filing).tz_localize(None).normalize()
    if pd.isna(accepted):
        return {**empty, "day0_reason": "missing_or_invalid_acceptance_timestamp"}
    eastern = accepted.tz_convert("America/New_York")
    after_close = eastern.hour >= 16
    accepted_date = eastern.tz_localize(None).normalize() + pd.Timedelta(days=int(after_close))
    candidate = max(filing, accepted_date)
    baseline_pos, event_pos = sessions.searchsorted(filing), sessions.searchsorted(candidate)
    if event_pos >= len(sessions) or baseline_pos >= len(sessions):
        return {**empty, "day0_reason": "outside_calendar_coverage", "acceptance_eastern": eastern.isoformat(),
                "acceptance_after_close": after_close, "day0_candidate_date": candidate}
    day0, baseline = sessions[event_pos], sessions[baseline_pos]
    return {"day0": day0, "filing_date_day0": baseline, "day0_moved": bool(day0 != baseline),
            "acceptance_eastern": eastern.isoformat(), "acceptance_after_close": after_close,
            "day0_candidate_date": candidate, "day0_reason": "ok"}


def _window(series: pd.Series, position: int, start: int, end: int) -> pd.Series:
    """Return the exact offset window, or an empty series if out of coverage."""
    if position + start < 0 or position + end >= len(series):
        return pd.Series(dtype=float)
    return series.iloc[position + start:position + end + 1]


def _complete(values: pd.Series, expected: int) -> bool:
    return len(values) == expected and bool(np.isfinite(values).all())


def _bh(values: pd.Series) -> float:
    return float((1 + values).prod() - 1)


def build_event_panel(meta: pd.DataFrame, prices: pd.DataFrame,
                      raw_prices: pd.DataFrame, volume: pd.DataFrame,
                      shares: pd.DataFrame, sessions=None,
                      benchmark: str = "SPY") -> pd.DataFrame:
    """Preserve every accession and append event values plus explicit pass flags.

    Required meta: accession,cik,ticker,form,filing_date,acceptance_datetime.
    Price inputs are dates x tickers. Shares joins are one-to-one; supported tag,
    timely as-of date and positive count are required even for legacy CSVs.
    No sample filtering occurs here. The caller applies the prescribed waterfall
    before defining the tf.idf corpus. pass_complete_windows and pass_controls
    are additional exclusions, separately counted after the README filters.
    """
    required = {"accession", "cik", "ticker", "form", "filing_date", "acceptance_datetime"}
    if not required.issubset(meta.columns):
        raise ValueError(f"Metadata missing {sorted(required - set(meta.columns))}")
    if meta["accession"].duplicated().any() or shares["accession"].duplicated().any():
        raise ValueError("Duplicate accessions in metadata or shares")
    prices, raw_prices, volume = (_daily(f) for f in (prices, raw_prices, volume))
    if benchmark not in prices or prices[benchmark].dropna().empty:
        raise ValueError(f"Missing {benchmark} benchmark history")
    if sessions is None:
        filing_dates = pd.to_datetime(meta["filing_date"], errors="coerce")
        start = min(prices.index.min(), filing_dates.min() - pd.Timedelta(days=140))
        end = max(prices.index.max(), filing_dates.max() + pd.Timedelta(days=140))
        sessions = trading_sessions(start, end)
    sessions = _session_index(sessions)
    # Calendar reindexing before pct_change preserves missing-session offsets.
    prices = prices.reindex(sessions).apply(pd.to_numeric, errors="coerce")
    prices = prices.where(prices > 0)
    raw_prices = raw_prices.reindex(sessions).apply(pd.to_numeric, errors="coerce")
    raw_prices = raw_prices.where(raw_prices > 0)
    volume = volume.reindex(sessions).apply(pd.to_numeric, errors="coerce")
    volume = volume.where(volume >= 0)
    returns = prices.pct_change(fill_method=None)
    spy = returns[benchmark]
    share_cols = [c for c in shares if c != "cik" and c not in meta.columns] + ["accession"]
    merged = meta.merge(shares[list(dict.fromkeys(share_cols))], on="accession", how="left", validate="one_to_one")
    allowed_tags = {f"{ns}:{tag}" for ns, tag in SHARE_TAGS}
    rows = []
    numeric = ["price_day_minus1", "event_return", "prior_excess_return", "pre_volatility", "post_volatility",
               "log_size", "log_dollar_volume", "mean_dollar_volume"]
    flags = ["pass_day0_price", "pass_return_history", "pass_complete_windows", "pass_shares", "pass_controls"]
    for _, row in merged.iterrows():
        out = {**row.to_dict(), **{name: np.nan for name in numeric}, **{name: False for name in flags}}
        out.update(assign_day0(row["filing_date"], row["acceptance_datetime"], sessions))
        filing = pd.to_datetime(row["filing_date"], errors="coerce")
        out["quarter"] = str(filing.to_period("Q")) if pd.notna(filing) else None
        out["is_10k"] = int(row["form"] == "10-K")
        out.update(n_pre_history_returns=0, n_post_history_returns=0, n_pre_vol_returns=0,
                   n_post_vol_returns=0, n_event_returns=0, n_pre_benchmark_returns=0,
                   n_event_benchmark_returns=0, n_dollar_volume_days=0)
        count = pd.to_numeric(row.get("shares_outstanding"), errors="coerce")
        as_of = pd.to_datetime(row.get("shares_as_of"), errors="coerce")
        out["pass_shares"] = bool(pd.notna(count) and np.isfinite(count) and count > 0
            and row.get("shares_tag") in allowed_tags and pd.notna(as_of) and pd.notna(filing)
            and as_of <= filing and row.get("shares_status", "ok") == "ok"
            and row.get("shares_source_accession", row["accession"]) == row["accession"])
        reasons = []
        ticker = row["ticker"]
        if pd.isna(out["day0"]):
            reasons.append(out["day0_reason"])
        elif ticker not in prices or ticker not in raw_prices:
            reasons.append("missing_ticker_prices")
        else:
            pos = sessions.get_loc(out["day0"])
            stock = returns[ticker]
            price_minus1 = _window(raw_prices[ticker], pos, -1, -1)
            if _complete(price_minus1, 1):
                out["price_day_minus1"] = float(price_minus1.iloc[0])
                out["pass_day0_price"] = out["price_day_minus1"] >= 3
            before, after = _window(stock, pos, -60, -1), _window(stock, pos, 1, 60)
            pre, post = _window(stock, pos, -60, -6), _window(stock, pos, 4, 63)
            event, spy_pre, spy_event = _window(stock, pos, 0, 3), _window(spy, pos, -60, -6), _window(spy, pos, 0, 3)
            for name, values in [("pre_history", before), ("post_history", after), ("pre_vol", pre),
                                 ("post_vol", post), ("event", event), ("pre_benchmark", spy_pre),
                                 ("event_benchmark", spy_event)]:
                out[f"n_{name}_returns"] = int(np.isfinite(values).sum())
            out["pass_return_history"] = _complete(before, 60) and _complete(after, 60)
            out["pass_complete_windows"] = (_complete(pre, 55) and _complete(post, 60)
                and _complete(event, 4) and _complete(spy_pre, 55) and _complete(spy_event, 4))
            if _complete(pre, 55):
                out["pre_volatility"] = float(pre.std(ddof=1) * np.sqrt(252))
            if _complete(post, 60):
                out["post_volatility"] = float(post.std(ddof=1) * np.sqrt(252))
            if _complete(pre, 55) and _complete(spy_pre, 55):
                out["prior_excess_return"] = _bh(pre) - _bh(spy_pre)
            if _complete(event, 4) and _complete(spy_event, 4):
                out["event_return"] = _bh(event) - _bh(spy_event)
            if out["pass_shares"] and np.isfinite(out["price_day_minus1"]):
                out["log_size"] = float(np.log(out["price_day_minus1"] * count))
            if ticker in volume:
                dollars = _window(raw_prices[ticker] * volume[ticker], pos, -60, -6)
                out["n_dollar_volume_days"] = int(np.isfinite(dollars).sum())
                if _complete(dollars, 55) and dollars.mean() > 0:
                    out["mean_dollar_volume"] = float(dollars.mean())
                    out["log_dollar_volume"] = float(np.log(dollars.mean()))
        out["pass_controls"] = all(np.isfinite(out[c]) for c in COMMON_CONTROLS)
        for flag, reason in [("pass_day0_price", "unusable_day0_or_nominal_price_below_3"),
            ("pass_return_history", "incomplete_60_before_or_60_after_returns"),
            ("pass_complete_windows", "incomplete_assigned_event_or_volatility_or_benchmark_window"),
            ("pass_shares", "missing_or_ineligible_filing_outstanding_shares"),
            ("pass_controls", "missing_common_controls")]:
            if not out[flag]:
                reasons.append(reason)
        out["market_exclusion_reasons"] = ";".join(reasons)
        rows.append(out)
    return pd.DataFrame(rows)
