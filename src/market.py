"""Auditable Yahoo market data and filing-specific outstanding share counts.

Yahoo Close/Volume are split adjusted even with auto_adjust=False. Download
through retrieval day and undo all subsequent splits for nominal historical
prices/volume; Adj Close supplies total returns. Maintainer explanation:
https://github.com/ranaroussi/yfinance/issues/687 . A split on t changes units
on t, so only splits strictly after t enter its undo factor.

Shares are instantaneous outstanding shares, never weighted averages.
Dimensional/multi-class counts without an unambiguous aggregate remain missing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FILING_DIR, PRICE_DIR

SHARE_TAGS = [("dei", "EntityCommonStockSharesOutstanding"),
              ("us-gaap", "CommonStockSharesOutstanding")]
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


def _daily(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    frame.index = idx.normalize()
    if frame.index.has_duplicates:
        raise ValueError("Duplicate market dates")
    return frame.sort_index().rename_axis("Date")


def nominal_history(history: pd.DataFrame) -> pd.DataFrame:
    """Undo splits using full-to-retrieval history, before truncating its dates.

    No filling. Nominal price times nominal volume is invariant to the split
    unit conversion, including reverse splits. Same-day splits are excluded.
    """
    history = _daily(history)
    required = {"Close", "Adj Close", "Volume", "Stock Splits"}
    if not required.issubset(history.columns):
        raise ValueError(f"Yahoo history missing {sorted(required - set(history.columns))}")
    ratios = history["Stock Splits"].fillna(0).replace(0, 1).astype(float)
    if not np.isfinite(ratios).all() or (ratios <= 0).any():
        raise ValueError("Invalid split ratio")
    subsequent = ratios.iloc[::-1].cumprod().iloc[::-1] / ratios
    return pd.DataFrame({"prices": history["Adj Close"],
        "raw_prices": history["Close"] * subsequent,
        "volume": history["Volume"] / subsequent,
        "split_factors": subsequent}, index=history.index)


def download_market_data(tickers: list[str], start: str, end: str,
                         directory: Path | None = None,
                         refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Cache individual Yahoo snapshots/parameters; persist every failure.

    End is exclusive. Cache reuse requires the same analysis request and
    validated schema. Snapshots stay frozen until refresh. Download through
    today to recover splits after the analysis end. Failed tickers remain NaN
    columns, with reasons in market_download_log.csv. No mapping is guessed.
    """
    import yfinance as yf
    directory = Path(directory or PRICE_DIR)
    cache = directory / "yahoo_cache"
    cache.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict] = {k: {} for k in ("prices", "raw_prices", "volume", "split_factors")}
    logs = []
    request_end = (pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)).date().isoformat()
    for ticker in sorted(set(str(t) for t in tickers if pd.notna(t))):
        safe = ticker.encode("utf-8").hex()
        path, manifest_path = cache / f"{safe}.csv", cache / f"{safe}.json"
        entry = {"ticker": ticker, "analysis_start": start, "analysis_end_exclusive": end,
                 "status": "failed", "reason": "", "cache_hit": False}
        try:
            valid = False
            if path.exists() and manifest_path.exists() and not refresh:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                valid = (manifest.get("analysis_start") == start
                    and manifest.get("analysis_end_exclusive") == end and manifest.get("schema_version") == 2)
            if valid:
                history = pd.read_csv(path, index_col=0, parse_dates=True)
                entry["cache_hit"] = True
            else:
                history = yf.Ticker(ticker).history(start=start, end=request_end,
                    auto_adjust=False, back_adjust=False, actions=True, repair=False, raise_errors=True)
                if history.empty:
                    raise ValueError("Yahoo returned no history (possibly unavailable or delisted)")
                history = _daily(history)
                nominal_history(history)
                manifest = {"ticker": ticker, "analysis_start": start, "analysis_end_exclusive": end,
                    "download_end_exclusive": request_end, "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                    "yfinance_version": yf.__version__, "schema_version": 2, "auto_adjust": False, "repair": False}
                history.to_csv(path)
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            converted = nominal_history(history)
            converted = converted.loc[(converted.index >= pd.Timestamp(start)) & (converted.index < pd.Timestamp(end))]
            if converted.empty or not (converted["prices"] > 0).any():
                raise ValueError("No usable adjusted closes in analysis period")
            for key in outputs:
                outputs[key][ticker] = converted[key]
            entry.update(status="ok", observations=int(converted["prices"].notna().sum()),
                retrieved_at_utc=manifest["retrieved_at_utc"], download_end_exclusive=manifest["download_end_exclusive"],
                missing_adjusted_close=int(converted["prices"].isna().sum()))
        except Exception as exc:
            entry["reason"] = f"{type(exc).__name__}: {exc}"
            for key in outputs:
                outputs[key][ticker] = pd.Series(dtype=float)
        logs.append(entry)
        pd.DataFrame(logs).to_csv(directory / "market_download_log.csv", index=False)
    result = {}
    for key, values in outputs.items():
        frame = pd.DataFrame(values).sort_index().rename_axis("Date")
        frame.to_csv(directory / f"{key}.csv")
        result[key] = frame
    result["diagnostics"] = pd.DataFrame(logs)
    return result


def download_prices(tickers: list[str], start: str, end: str, cache_path: Path | None = None) -> pd.DataFrame:
    """Compatibility entry point using the validated cache."""
    path = Path(cache_path or PRICE_DIR / "prices.csv")
    result = download_market_data(tickers, start, end, path.parent)["prices"]
    if path.name != "prices.csv":
        result.to_csv(path)
    return result


def download_volume(tickers: list[str], start: str, end: str, cache_path: Path | None = None) -> pd.DataFrame:
    """Historical nominal share volume, consistent with raw_prices.csv."""
    path = Path(cache_path or PRICE_DIR / "volume.csv")
    result = download_market_data(tickers, start, end, path.parent)["volume"]
    if path.name != "volume.csv":
        result.to_csv(path)
    return result


def _select_candidates(candidates: list[dict], filing_date, source: str) -> dict:
    filing_date = pd.Timestamp(filing_date).normalize()
    dimensional_result = None
    for ns, tag in SHARE_TAGS:
        name = f"{ns}:{tag}"
        eligible = []
        for fact in candidates:
            end = pd.to_datetime(fact.get("end"), errors="coerce")
            filed = pd.to_datetime(fact.get("filed", filing_date), errors="coerce")
            value = pd.to_numeric(fact.get("val"), errors="coerce")
            if (fact.get("tag") == name and fact.get("units") == "shares" and not fact.get("start")
                and pd.notna(end) and end <= filing_date and pd.notna(filed) and filed <= filing_date
                and pd.notna(value) and np.isfinite(value) and value > 0):
                eligible.append({**fact, "end": end, "val": float(value)})
        if not eligible:
            continue
        latest = max(f["end"] for f in eligible)
        aggregate = [f for f in eligible if f["end"] == latest and not f.get("dimensions")]
        if not aggregate:
            dimensional_result = {"shares_status": "missing", "shares_reason": "dimensional_or_multiclass_count",
                                  "shares_source": source, "shares_tag": name}
            continue  # A dimensionless CommonStockSharesOutstanding total may still exist.
        values = {f["val"] for f in aggregate}
        if len(values) != 1:
            return {"shares_status": "missing", "shares_reason": "conflicting_counts_same_as_of",
                    "shares_source": source, "shares_tag": name}
        return {"shares_outstanding": values.pop(), "shares_as_of": latest.date().isoformat(),
            "shares_tag": name, "shares_source": source, "shares_status": "ok",
            "shares_reason": "exact_accession_latest_instant_at_or_before_filing",
            "shares_context": "|".join(sorted({str(f.get("context", "")) for f in aggregate}))}
    return dimensional_result or {"shares_status": "missing",
        "shares_reason": "no_eligible_instantaneous_outstanding_shares", "shares_source": source}


def select_companyfacts_shares(payload: dict, accession: str, filing_date) -> dict:
    """Exact accession, shares units, latest instant <= filing; DEI preferred.

    Companyfacts omits dimensional facts. Duplicate equal counts are harmless;
    conflicts fail. Both the fact's as-of and publication dates must be timely.
    """
    candidates = []
    facts = payload.get("facts", payload)
    for ns, tag in SHARE_TAGS:
        for fact in facts.get(ns, {}).get(tag, {}).get("units", {}).get("shares", []):
            if fact.get("accn") == accession:
                candidates.append({**fact, "tag": f"{ns}:{tag}", "units": "shares"})
    return _select_candidates(candidates, filing_date, "sec_companyfacts")


def select_inline_shares(raw_html: str, filing_date) -> dict:
    """Read exact-filing inline shares with lxml, preserving contexts and units.

    Only supported instantaneous facts and numeric transformations qualify.
    lxml avoids materializing BeautifulSoup wrappers for every unrelated filing
    element; financial tables can contain hundreds of thousands of elements.
    """
    from lxml import html
    document = html.fromstring(raw_html.encode("utf-8"), parser=html.HTMLParser(huge_tree=True, no_network=True))
    names = {f"{ns}:{tag}" for ns, tag in SHARE_TAGS}
    facts = [node for node in document.xpath("//*[@name]") if node.get("name") in names]
    context_ids = {node.get("contextref") for node in facts}
    unit_ids = {node.get("unitref") for node in facts}
    contexts, units = {}, {}

    def local(node):
        return str(node.tag).lower().split("}")[-1].split(":")[-1]

    def text(node):
        return "".join(node.itertext()).strip()

    for node in document.xpath("//*[@id]"):
        node_id = node.get("id")
        if node_id in context_ids and local(node) == "context":
            descendants = list(node.iterdescendants())
            instants = [item for item in descendants if local(item) == "instant"]
            durations = [item for item in descendants if local(item) == "startdate"]
            members = [item for item in descendants if local(item) in ("explicitmember", "typedmember")]
            if len(instants) == 1 and not durations:
                contexts[node_id] = {"end": text(instants[0]), "dimensions": [text(item) for item in members]}
        elif node_id in unit_ids and local(node) == "unit":
            measures = [item for item in node.iterdescendants() if local(item) == "measure"]
            if len(measures) == 1 and text(measures[0]).split(":")[-1] == "shares":
                units[node_id] = "shares"
    candidates = []
    for node in facts:
        context_id = node.get("contextref")
        if context_id not in contexts or node.get("unitref") not in units:
            continue
        fmt = node.get("format", "").lower().split(":")[-1]
        if fmt and fmt not in ("num-dot-decimal", "numdotdecimal", "numcommadot"):
            continue
        try:
            val = float(text(node).replace(",", "").replace(" ", "").replace("\xa0", ""))
            val *= 10 ** int(node.get("scale", "0"))
            if node.get("sign") == "-":
                val *= -1
        except (TypeError, ValueError, OverflowError):
            continue
        candidates.append({**contexts[context_id], "tag": node.get("name"), "units": "shares", "val": val, "context": context_id})
    return _select_candidates(candidates, filing_date, "filing_inline_xbrl")


def get_shares(client, meta: pd.DataFrame, cache_dir: Path | None = None,
               html_dir: Path | None = None) -> pd.DataFrame:
    """One row per accession, including retrieval failures and missing counts.

    Try raw inline cover-page counts before companyfacts. Preserve exact source
    accession, tag, as-of date, URL and failure reasons. No later-filing filling.
    """
    cache_dir = Path(cache_dir or PRICE_DIR / "companyfacts")
    html_dir = Path(html_dir or FILING_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if meta["accession"].duplicated().any():
        raise ValueError("Shares input contains duplicate accessions")
    rows = []
    for cik, group in meta.groupby("cik", sort=True, dropna=False):
        cik = str(cik).zfill(10) if pd.notna(cik) else None
        url, facts, facts_error = FACTS_URL.format(cik=cik) if cik else None, None, ""
        cache_path = cache_dir / f"CIK{cik}.json"
        manifest_path = cache_dir / f"CIK{cik}.metadata.json"
        retrieved_at = None
        try:
            if not cik:
                raise ValueError("missing_cik")
            if cache_path.exists():
                facts = json.loads(cache_path.read_text(encoding="utf-8"))
                if manifest_path.exists():
                    retrieved_at = json.loads(manifest_path.read_text(encoding="utf-8")).get("retrieved_at_utc")
            else:
                facts = client._get(url).json()
                retrieved_at = pd.Timestamp.now(tz="UTC").isoformat()
                cache_path.write_text(json.dumps(facts), encoding="utf-8")
                manifest_path.write_text(json.dumps({"source_url": url, "retrieved_at_utc": retrieved_at}), encoding="utf-8")
        except Exception as exc:
            facts_error = f"{type(exc).__name__}: {exc}"
        for _, filing in group.iterrows():
            accession = filing["accession"]
            result = {"shares_status": "missing", "shares_reason": "raw_html_unavailable"}
            html_path = html_dir / f"{accession.replace('-', '')}.html"
            html_error = ""
            if html_path.exists():
                try:
                    result = select_inline_shares(html_path.read_text(encoding="utf-8", errors="replace"), filing["filing_date"])
                except Exception as exc:
                    html_error = f"{type(exc).__name__}: {exc}"
                    result = {"shares_status": "missing", "shares_reason": "inline_parse_error"}
            inline_reason = result.get("shares_reason", "")
            if facts is not None:
                api_result = select_companyfacts_shares(facts, accession, filing["filing_date"])
                priority = {f"{ns}:{tag}": rank for rank, (ns, tag) in enumerate(SHARE_TAGS)}
                higher_priority = priority.get(api_result.get("shares_tag"), 99) < priority.get(result.get("shares_tag"), 99)
                # A conflicting raw cover-page count cannot be cured by silently
                # selecting one same-tag API fact. An actual aggregate fallback
                # for dimensional classes is allowed and retains inline_result.
                conflict = inline_reason == "conflicting_counts_same_as_of"
                if (api_result["shares_status"] == "ok" and not conflict
                    and (result["shares_status"] != "ok" or higher_priority)) or inline_reason == "raw_html_unavailable":
                    result = api_result
            rows.append({"cik": cik, "accession": accession, "shares_outstanding": np.nan,
                "shares_as_of": None, "shares_tag": None, "shares_context": None, "shares_source": None, **result,
                "shares_source_accession": accession, "companyfacts_url": url,
                "shares_source_url": filing.get("doc_url") if result.get("shares_source") == "filing_inline_xbrl" else url,
                "companyfacts_retrieved_at_utc": retrieved_at,
                "companyfacts_error": facts_error, "inline_error": html_error, "inline_result": inline_reason,
                "shares_selected_at_utc": pd.Timestamp.now(tz="UTC").isoformat()})
    return pd.DataFrame(rows, columns=None if rows else ["cik", "accession", "shares_outstanding",
        "shares_as_of", "shares_tag", "shares_source", "shares_status", "shares_reason"])
