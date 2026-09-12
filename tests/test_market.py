"""No-network checks of split units, exact-filing shares and failure persistence."""
from types import SimpleNamespace
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.market import (download_market_data, get_shares, nominal_history,
                        select_companyfacts_shares, select_inline_shares)


def test_forward_and_reverse_splits_preserve_nominal_dollar_volume():
    history = pd.DataFrame({"Close": [10., 11., 12., 13.], "Adj Close": [9., 10., 11., 13.],
        "Volume": [100., 110., 120., 130.], "Stock Splits": [0., 2., 0., .2]},
        index=pd.date_range("2021-01-01", periods=4))
    converted = nominal_history(history)
    np.testing.assert_allclose(converted["split_factors"], [.4, .2, .2, 1.])
    np.testing.assert_allclose(converted["raw_prices"], [4., 2.2, 2.4, 13.])
    np.testing.assert_allclose(converted["volume"], [250., 550., 600., 130.])
    np.testing.assert_allclose(converted["raw_prices"] * converted["volume"], history["Close"] * history["Volume"])
    np.testing.assert_allclose(converted["prices"], history["Adj Close"])


def test_market_cache_future_splits_and_failures(tmp_path, monkeypatch):
    import yfinance as yf
    calls = []
    history = pd.DataFrame({"Close": [5., 6., 7.], "Adj Close": [4., 5., 7.],
        "Volume": [100., 120., 140.], "Stock Splits": [0., 0., 2.]},
        index=pd.to_datetime(["2021-01-04", "2021-01-05", "2025-01-06"]))

    class FakeTicker:
        def __init__(self, ticker):
            self.ticker = ticker

        def history(self, **kwargs):
            calls.append((self.ticker, kwargs))
            if self.ticker == "MISSING":
                raise RuntimeError("simulated unavailable history")
            return history.copy()

    monkeypatch.setattr(yf, "Ticker", FakeTicker)
    result = download_market_data(["OK", "MISSING"], "2021-01-01", "2021-02-01", tmp_path)
    assert result["raw_prices"].loc["2021-01-04", "OK"] == 10  # undo split after requested end
    assert result["volume"].loc["2021-01-04", "OK"] == 50
    assert result["prices"].loc["2021-01-04", "OK"] == 4
    assert result["prices"]["MISSING"].isna().all()
    log = pd.read_csv(tmp_path / "market_download_log.csv").set_index("ticker")
    assert log.loc["MISSING", "status"] == "failed"
    assert "simulated unavailable history" in log.loc["MISSING", "reason"]
    assert calls[1][1]["auto_adjust"] is False and calls[1][1]["actions"] is True
    assert pd.Timestamp(calls[1][1]["end"]) > pd.Timestamp("2025-01-06")
    calls.clear()
    cached = download_market_data(["OK"], "2021-01-01", "2021-02-01", tmp_path)
    assert calls == [] and cached["diagnostics"].iloc[0]["cache_hit"]
    download_market_data(["OK"], "2021-01-01", "2021-02-01", tmp_path, refresh=True)
    assert len(calls) == 1


def facts_payload(facts, extra=None):
    return {"facts": {"dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": facts}}},
                      "us-gaap": extra or {}}}


def test_share_selection_exact_accession_latest_eligible_instant():
    base = {"accn": "a", "filed": "2021-05-10", "end": "2021-05-03", "val": 100}
    facts = [base, {**base, "end": "2021-03-31", "val": 80},
        {**base, "accn": "later-filing", "val": 900},
        {**base, "end": "2021-05-11", "val": 1000},
        {**base, "filed": "2021-05-11", "val": 2000},
        {**base, "start": "2021-01-01", "val": 3000},
        {**base, "end": "2021-05-09", "val": -1}]
    result = select_companyfacts_shares(facts_payload(facts), "a", "2021-05-10")
    assert result["shares_status"] == "ok"
    assert result["shares_outstanding"] == 100
    assert result["shares_as_of"] == "2021-05-03"
    assert result["shares_source"] == "sec_companyfacts"


def test_weighted_shares_and_nonshare_units_are_rejected():
    fact = {"accn": "a", "end": "2021-03-31", "filed": "2021-05-10", "val": 123}
    payload = facts_payload([], {"WeightedAverageNumberOfSharesOutstandingBasic": {"units": {"shares": [fact]}}})
    payload["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"]["USD"] = [fact]
    result = select_companyfacts_shares(payload, "a", "2021-05-10")
    assert result["shares_status"] == "missing"
    assert "shares_outstanding" not in result


def test_actual_common_outstanding_fallback_and_conflicts():
    fact = {"accn": "a", "end": "2021-03-31", "filed": "2021-05-10", "val": 123}
    payload = facts_payload([], {"CommonStockSharesOutstanding": {"units": {"shares": [fact]}}})
    result = select_companyfacts_shares(payload, "a", "2021-05-10")
    assert result["shares_outstanding"] == 123
    assert result["shares_tag"] == "us-gaap:CommonStockSharesOutstanding"
    result = select_companyfacts_shares(facts_payload([fact, {**fact, "val": 124}]), "a", "2021-05-10")
    assert result["shares_reason"] == "conflicting_counts_same_as_of"


def inline_document(dimensions="", value="12,345", scale="3"):
    return f'''<html><body><xbrli:context id="c"><xbrli:entity>{dimensions}</xbrli:entity>
    <xbrli:period><xbrli:instant>2021-05-03</xbrli:instant></xbrli:period></xbrli:context>
    <xbrli:unit id="u"><xbrli:measure>xbrli:shares</xbrli:measure></xbrli:unit>
    <ix:nonFraction name="dei:EntityCommonStockSharesOutstanding" contextRef="c" unitRef="u"
      scale="{scale}" format="ixt:num-dot-decimal">{value}</ix:nonFraction></body></html>'''


def test_inline_cover_shares_honor_context_scale_and_units():
    result = select_inline_shares(inline_document(), "2021-05-10")
    assert result["shares_outstanding"] == 12_345_000
    assert result["shares_context"] == "c"
    assert result["shares_source"] == "filing_inline_xbrl"
    invalid = select_inline_shares(inline_document().replace("xbrli:shares", "iso4217:USD"), "2021-05-10")
    assert invalid["shares_status"] == "missing"


def test_multiclass_counts_are_not_guessed_or_summed():
    dimensions = '<xbrli:segment><xbrldi:explicitMember dimension="us-gaap:StatementClassOfStockAxis">company:ClassA</xbrldi:explicitMember></xbrli:segment>'
    result = select_inline_shares(inline_document(dimensions), "2021-05-10")
    assert result["shares_reason"] == "dimensional_or_multiclass_count"
    assert "shares_outstanding" not in result


def test_multiclass_cover_can_use_explicit_dimensionless_aggregate():
    dimensions = '<xbrli:segment><xbrldi:explicitMember dimension="us-gaap:StatementClassOfStockAxis">company:ClassA</xbrldi:explicitMember></xbrli:segment>'
    raw = inline_document(dimensions)
    aggregate = inline_document(value="25", scale="6").replace("dei:EntityCommonStockSharesOutstanding", "us-gaap:CommonStockSharesOutstanding")
    aggregate = aggregate.replace('id="c"', 'id="total"').replace('contextRef="c"', 'contextRef="total"')
    aggregate_body = aggregate.split("<body>")[1].split("</body>")[0]
    result = select_inline_shares(raw.replace("</body>", aggregate_body + "</body>"), "2021-05-10")
    assert result["shares_outstanding"] == 25_000_000
    assert result["shares_tag"] == "us-gaap:CommonStockSharesOutstanding"


def test_every_filing_retained_when_companyfacts_fails(tmp_path):
    class FailedClient:
        def _get(self, url):
            raise RuntimeError("simulated SEC request failure")

    html_dir = tmp_path / "html"
    html_dir.mkdir()
    (html_dir / "a.html").write_text(inline_document(), encoding="utf-8")
    meta = pd.DataFrame([{"cik": "1", "accession": a, "filing_date": "2021-05-10"} for a in ["a", "b"]])
    result = get_shares(FailedClient(), meta, tmp_path / "facts", html_dir).set_index("accession")
    assert set(result.index) == {"a", "b"}
    assert result.loc["a", "shares_status"] == "ok"
    assert result.loc["b", "shares_status"] == "missing"
    assert pd.isna(result.loc["b", "shares_outstanding"])
    assert result["companyfacts_error"].str.contains("simulated SEC request failure").all()
    assert result.loc["a", "shares_source_accession"] == "a"


def test_companyfacts_cache_is_reused_without_network(tmp_path):
    fact = {"accn": "a", "end": "2021-05-03", "filed": "2021-05-10", "val": 100}
    calls = []

    class Client:
        def _get(self, url):
            calls.append(url)
            return SimpleNamespace(json=lambda: facts_payload([fact]))

    meta = pd.DataFrame([{"cik": "1", "accession": "a", "filing_date": "2021-05-10"}])
    result = get_shares(Client(), meta, tmp_path, tmp_path / "html")
    assert result.loc[0, "shares_outstanding"] == 100
    cached = get_shares(Client(), meta, tmp_path, tmp_path / "html")
    assert len(calls) == 1
    assert cached.loc[0, "companyfacts_retrieved_at_utc"] == result.loc[0, "companyfacts_retrieved_at_utc"]


def test_missing_cik_retains_row_and_avoids_a_request(tmp_path):
    class Client:
        def _get(self, url):
            pytest.fail("A missing CIK must not cause a request")

    meta = pd.DataFrame([{"cik": None, "accession": "a", "filing_date": "2021-05-10"}])
    result = get_shares(Client(), meta, tmp_path, tmp_path / "html")
    assert len(result) == 1
    assert result.loc[0, "shares_status"] == "missing"
    assert "missing_cik" in result.loc[0, "companyfacts_error"]


def test_coverpage_api_has_priority_over_raw_balance_sheet_total(tmp_path):
    fact = {"accn": "a", "end": "2021-05-03", "filed": "2021-05-10", "val": 100}
    client = SimpleNamespace(_get=lambda url: SimpleNamespace(json=lambda: facts_payload([fact])))
    html_dir = tmp_path / "html"
    html_dir.mkdir()
    raw = inline_document().replace("dei:EntityCommonStockSharesOutstanding", "us-gaap:CommonStockSharesOutstanding")
    (html_dir / "a.html").write_text(raw, encoding="utf-8")
    meta = pd.DataFrame([{"cik": "1", "accession": "a", "filing_date": "2021-05-10"}])
    result = get_shares(client, meta, tmp_path / "facts", html_dir)
    assert result.loc[0, "shares_outstanding"] == 100
    assert result.loc[0, "shares_source"] == "sec_companyfacts"


def test_conflicting_coverpage_values_not_hidden_by_api_fallback(tmp_path):
    fact = {"accn": "a", "end": "2021-05-03", "filed": "2021-05-10", "val": 100}
    client = SimpleNamespace(_get=lambda url: SimpleNamespace(json=lambda: facts_payload([fact])))
    html_dir = tmp_path / "html"
    html_dir.mkdir()
    raw = inline_document(value="100", scale="0")
    second_body = inline_document(value="101", scale="0").split("<body>")[1].split("</body>")[0]
    raw = raw.replace("</body>", second_body + "</body>")
    (html_dir / "a.html").write_text(raw, encoding="utf-8")
    meta = pd.DataFrame([{"cik": "1", "accession": "a", "filing_date": "2021-05-10"}])
    result = get_shares(client, meta, tmp_path / "facts", html_dir)
    assert result.loc[0, "shares_reason"] == "conflicting_counts_same_as_of"
    assert pd.isna(result.loc[0, "shares_outstanding"])


def test_market_provenance_rejects_partial_shares_and_changed_metadata(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "03_get_market_data.py"
    spec = importlib.util.spec_from_file_location("market_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata_path = tmp_path / "filings_meta.csv"
    pd.DataFrame([{"accession": "a", "cik": "1", "ticker": "FIRM", "filing_date": "2021-01-04"},
                  {"accession": "b", "cik": "1", "ticker": "FIRM", "filing_date": "2021-04-05"}]).to_csv(metadata_path, index=False)
    meta, digest = module.read_metadata(metadata_path)
    run = {"metadata_sha256": digest, "requested_tickers": ["FIRM", "SPY"]}
    shares = pd.DataFrame([{"accession": "a", "shares_status": "ok", "shares_reason": "verified", "shares_outstanding": 100},
                          {"accession": "b", "shares_status": "missing", "shares_reason": "no fact", "shares_outstanding": np.nan}])
    shares.iloc[:1].to_csv(tmp_path / "shares.csv", index=False)
    with pytest.raises(ValueError, match="accession coverage"):
        module.finalize_run(run, meta, metadata_path, tmp_path)
    shares.to_csv(tmp_path / "shares.csv", index=False)
    frame = pd.DataFrame({"FIRM": [10.], "SPY": [100.]}, index=pd.to_datetime(["2021-01-04"]))
    for key in ["prices", "raw_prices", "volume", "split_factors"]:
        frame.to_csv(tmp_path / f"{key}.csv")
    pd.DataFrame([{"ticker": t, "status": "ok", "reason": ""} for t in ["FIRM", "SPY"]]).to_csv(tmp_path / "market_download_log.csv", index=False)
    final = module.finalize_run(run, meta, metadata_path, tmp_path)
    assert final["status"] == "complete_with_missing_data"
    assert final["shares_accession_count"] == 2
    assert final["shares_status_counts"] == {"ok": 1, "missing": 1}
    assert final["shares_accession_sha256"] == module.accession_fingerprint(["b", "a"])
    assert "shares.csv" in final["output_sha256"]
    metadata_path.write_text(metadata_path.read_text() + "\n")
    with pytest.raises(ValueError, match="metadata changed"):
        module.finalize_run(run, meta, metadata_path, tmp_path)


def test_inline_xml_encoding_declaration_is_accepted():
    from src.market import select_inline_shares
    raw = '''<?xml version="1.0" encoding="UTF-8"?><html><body>
    <xbrli:context id="c"><xbrli:period><xbrli:instant>2024-01-31</xbrli:instant></xbrli:period></xbrli:context>
    <xbrli:unit id="u"><xbrli:measure>xbrli:shares</xbrli:measure></xbrli:unit>
    <ix:nonFraction name="dei:EntityCommonStockSharesOutstanding" contextRef="c" unitRef="u">123,456</ix:nonFraction>
    </body></html>'''
    result = select_inline_shares(raw, "2024-02-01")
    assert result["shares_outstanding"] == 123456
    assert result["shares_status"] == "ok"
