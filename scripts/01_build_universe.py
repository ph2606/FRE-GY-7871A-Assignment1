"""Resolve the frozen six-fund holdings without silently losing positions.

``holdings_resolution.csv`` audits each raw ticker/company/CUSIP identity.
``universe.csv`` contains one row per resolved SEC CIK, including companies with
no 10-K/Q history. Only ``domestic_filer`` rows feed the filing downloader.
An unmatched ticker is an unresolved mapping, not evidence of foreign status.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ARK_FUNDS, SAMPLE_END, SAMPLE_START, SEC_USER_AGENT, UNIVERSE_DIR
from src.edgar import EdgarClient

ARK_CSV_BASE = "https://assets.ark-funds.com/fund-documents/funds-etf-csv/"
ARK_CSV_NAMES = {
    "ARKK": "ARK_INNOVATION_ETF_ARKK_HOLDINGS.csv",
    "ARKQ": "ARK_AUTONOMOUS_TECH._&_ROBOTICS_ETF_ARKQ_HOLDINGS.csv",
    "ARKW": "ARK_NEXT_GENERATION_INTERNET_ETF_ARKW_HOLDINGS.csv",
    "ARKF": "ARK_FINTECH_INNOVATION_ETF_ARKF_HOLDINGS.csv",
    "ARKG": "ARK_GENOMIC_REVOLUTION_ETF_ARKG_HOLDINGS.csv",
    "ARKX": "ARK_SPACE_EXPLORATION_&_INNOVATION_ETF_ARKX_HOLDINGS.csv",
}
RAW_PATH = UNIVERSE_DIR / "ark_holdings_raw.csv"
OUT_PATH = UNIVERSE_DIR / "universe.csv"
FROZEN_HOLDINGS_URL = (
    "https://raw.githubusercontent.com/anmolsingh0219/FRE-GY-7871A-Assignment1/"
    "532c65cf91cdf62a8d37c9bbe6ff0961c152d756/data/universe/ark_holdings_raw.csv"
)
US_EXCHANGE_CODES = {"US", "UN", "UW", "UQ", "UA", "UP", "UR"}
NAME_STOP = set("INC INCORPORATED INC CO CORP CORPORATION LTD LIMITED PLC SA SE AG NV HOLDINGS HOLDING GROUP CLASS CL ADR SPON SP SPONS UNSPONSORED COMMON STOCK COMPANY A B C AND THE OF TECHNOLOGY TECHNOLOGIES THERAPEUTICS".split())
SEC_TICKER_SOURCE = "https://www.sec.gov/files/company_tickers.json"
# Reviewed against issuer/exchange/depositary documents on 2026-09-09. Keys
# include the holding identifier so another issuer reusing a symbol cannot
# inherit the decision. Each decision below records its scope and primary source.
REVIEWED_HOLDINGS = {
    ("ADYEN", "BZ1HM42"): {
        "status": "reviewed_foreign_listing",
        "reason": "Adyen ordinary shares listed on Euronext Amsterdam; no eligible US ticker verified for this holding",
        "source_url": "https://www.adyen.com/press-and-media/adyen-hosts-investor-day-2025-in-amsterdam"},
    ("AIR", "4012250"): {
        "status": "reviewed_foreign_listing",
        "reason": "Airbus European AIR listing; SEC ticker AIR identifies AAR Corp, a different issuer; no eligible US ticker verified",
        "source_url": "https://www.airbus.com/en/investors/share-price-and-information"},
    ("DSY", "6177878"): {
        "status": "reviewed_foreign_listing",
        "reason": "Discovery Limited JSE DSY listing; SEC ticker DSY identifies a different issuer; no eligible US ticker verified",
        "source_url": "https://www.discovery.co.za/assets/discoverycoza/corporate/investor-relations/2026/results-booklet.pdf"},
    ("HO", "4162791"): {
        "status": "reviewed_foreign_listing",
        "reason": "Thales ordinary shares listed on Euronext Paris as HO; no eligible US ticker verified for this holding",
        "source_url": "https://www.thalesgroup.com/sites/default/files/2025-06/Universal%20Registration%20Document%202024%20-%20Thales.pdf"},
    ("ARKY", "00214Q724"): {
        "status": "reviewed_fund_instrument",
        "reason": "ARK Active Autocallable Income ETF is a fund instrument, outside the corporate 10-K/Q universe",
        "source_url": "https://www.ark-funds.com/funds/arky"},
    ("PRNT", "00214Q500"): {
        "status": "reviewed_fund_instrument",
        "reason": "The 3D Printing ETF is a fund instrument, outside the corporate 10-K/Q universe",
        "source_url": "https://www.ark-funds.com/funds/prnt"},
    ("BYDDY", "05606L100"): {
        "status": "mapped", "cik": "0001445162",
        "reason": "Reviewed BYD subject-issuer CIK from SEC F-6 index; 10-K/Q eligibility must be queried",
        "source_url": "https://www.sec.gov/Archives/edgar/data/1201935/0001019155-25-000189-index.htm"},
    ("KMTUY", "500458401"): {
        "status": "mapped", "cik": "0000056594",
        "reason": "Reviewed Komatsu issuer CIK from SEC annual report; 10-K/Q eligibility must be queried",
        "source_url": "https://www.sec.gov/Archives/edgar/data/56594/000095012309018863/c87234e20vf.htm"},
    ("SE", "81141R100"): {
        "status": "mapped", "cik": "0001703399",
        "reason": "Reviewed Sea issuer CIK from SEC annual report; 10-K/Q eligibility must be queried",
        "source_url": "https://www.sec.gov/Archives/edgar/data/1703399/000119312525084311/d940352d20f.htm"},
}


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    EdgarClient._replace_with_retry(temporary, path)


def refresh_holdings() -> pd.DataFrame:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (course assignment)"})
    frames = []
    for fund in ARK_FUNDS:
        response = session.get(ARK_CSV_BASE + ARK_CSV_NAMES[fund], timeout=60)
        response.raise_for_status()
        rows = [r for r in csv.DictReader(io.StringIO(response.text)) if (r.get("ticker") or "").strip()]
        frames.append(pd.DataFrame(rows))
    out = pd.concat(frames, ignore_index=True)
    validate_holdings(out)
    atomic_csv(out, RAW_PATH)
    return out


def get_frozen_holdings() -> pd.DataFrame:
    """Recover exactly the assigned snapshot, never substitute live holdings."""
    if not RAW_PATH.exists():
        response = requests.get(FROZEN_HOLDINGS_URL, timeout=60)
        response.raise_for_status()
        raw = pd.read_csv(io.BytesIO(response.content), dtype=str)
        validate_holdings(raw)
        EdgarClient._atomic_write(RAW_PATH, response.content)
        import hashlib
        provenance = {"url": FROZEN_HOLDINGS_URL,
                      "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                      "sha256": hashlib.sha256(response.content).hexdigest()}
        EdgarClient._atomic_write(RAW_PATH.with_name(RAW_PATH.name + ".provenance.json"),
                                   json.dumps(provenance, indent=2).encode())
    return pd.read_csv(RAW_PATH, dtype=str)


def validate_holdings(raw: pd.DataFrame) -> None:
    required = {"date", "fund", "company", "ticker", "cusip"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Holdings missing columns: {sorted(required - set(raw.columns))}")
    if set(raw["fund"]) != set(ARK_FUNDS):
        raise ValueError(f"Expected all six ARK funds; received {sorted(set(raw['fund']))}")
    if raw[list(required)].isna().any().any():
        raise ValueError("Missing holding identity or snapshot field")
    if pd.to_datetime(raw["date"], format="%m/%d/%Y", errors="coerce").isna().any():
        raise ValueError("Unparseable holdings snapshot date")
    if raw.duplicated(["date", "fund", "ticker", "cusip"]).any():
        raise ValueError("Duplicate position within a fund snapshot")


def classify_ticker(raw: str) -> tuple[str | None, str, str]:
    parts = str(raw or "").strip().upper().split()
    if not parts:
        return None, "missing_ticker", "No ticker in holdings"
    ticker = parts[0]
    if "/" in ticker:
        return None, "excluded_slash_instrument", "Slash-denominated listing is outside starter US common-equity ticker rule"
    if ticker.isdigit():
        return None, "excluded_numeric_listing", "Numeric foreign-exchange listing under starter eligibility rule"
    if len(parts) > 1 and parts[1] not in US_EXCHANGE_CODES:
        return None, "excluded_foreign_exchange", f"Bloomberg exchange code {parts[1]} is not a US exchange code"
    return ticker, "candidate", "Ticker eligible for SEC identity check"


def clean_ticker(raw: str) -> str | None:
    return classify_ticker(raw)[0]


def issuer_names_compatible(ark_name: str, sec_name: str) -> bool:
    """Conservative collision screen; a mismatch stays unresolved, never guessed."""
    def words(value: str) -> set[str]:
        return set(re.findall(r"[A-Z0-9]+", value.upper())) - NAME_STOP
    return any(a == b or (min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a)))
               for a in words(ark_name) for b in words(sec_name))


def resolve_holdings(raw: pd.DataFrame, client: EdgarClient) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_holdings(raw)
    ticker_records = {record["ticker"]: record for record in client.company_ticker_records()}
    outcomes = []
    # Preserve distinct issuers even when a stripped ticker would collide.
    for (raw_ticker, company, cusip), positions in raw.groupby(["ticker", "company", "cusip"], sort=True):
        ticker, status, reason = classify_ticker(raw_ticker)
        record = {"raw_ticker": raw_ticker, "ark_name": company, "cusip": cusip,
                  "ticker": ticker, "cik": None, "sec_name": None,
                  "funds": "|".join(sorted(set(positions["fund"]))),
                  "snapshot_dates": "|".join(sorted(set(positions["date"]))),
                  "n_positions": len(positions), "status": status, "reason": reason,
                  "mapping_source_url": SEC_TICKER_SOURCE if ticker else "starter eligibility rule",
                  "mapping_method": "automatic_ticker_and_issuer_name" if ticker else "starter_listing_filter"}
        review = REVIEWED_HOLDINGS.get((raw_ticker, str(cusip)))
        if review is not None:
            record.update(status=review["status"], reason=review["reason"],
                          mapping_source_url=review["source_url"], mapping_method="documented_manual_review",
                          cik=review.get("cik"))
        elif ticker is not None:
            match = ticker_records.get(ticker)
            if match is None:
                record.update(status="unmatched_sec_ticker", reason="Ticker absent from downloaded SEC ticker snapshot; issuer status unestablished")
            elif not issuer_names_compatible(company, match["sec_name"]):
                record.update(sec_name=match["sec_name"], status="identity_review_required",
                              reason="ARK and SEC issuer names do not pass the collision screen; CIK not assigned")
            else:
                record.update(match)
                record.update(status="mapped", reason="Exact ticker and compatible issuer name in SEC ticker snapshot")
        outcomes.append(record)
    resolution = pd.DataFrame(outcomes)
    records = []
    for cik, holdings in resolution[resolution["status"] == "mapped"].groupby("cik", sort=True):
        first = holdings.sort_values("ticker").iloc[0]
        try:
            filings = client.list_filings(cik, ["10-K", "10-Q"], SAMPLE_START, SAMPLE_END, include_amendments=True)
            if len(filings) and "company" in filings:
                resolution.loc[holdings.index, "sec_name"] = filings["company"].iloc[0]
                first = resolution.loc[first.name]
            n_10k = int((filings["form"] == "10-K").sum()) if len(filings) else 0
            n_10q = int((filings["form"] == "10-Q").sum()) if len(filings) else 0
            n_amendments = int(filings["form"].str.endswith("/A").sum()) if len(filings) else 0
            status = "domestic_filer" if n_10k + n_10q else "no_10x_filings"
            reason = "10-K/Q filings present in the assignment window" if n_10k + n_10q else "No 10-K or 10-Q in cached SEC submissions for assignment filing-date window"
        except Exception as exc:
            n_10k = n_10q = n_amendments = None
            status, reason = "metadata_error", f"{type(exc).__name__}: {exc}"
        resolution.loc[holdings.index, ["status", "reason"]] = [status, reason]
        records.append({"ticker": first["ticker"], "cik": cik, "sec_name": first["sec_name"],
                        "ark_name": first["ark_name"],
                        "ticker_aliases": "|".join(sorted(set(holdings["ticker"]))),
                        "funds": "|".join(sorted(set("|".join(holdings["funds"]).split("|")))),
                        "n_10k": n_10k, "n_10q": n_10q, "n_amendments": n_amendments,
                        "status": status, "reason": reason,
                        "mapping_source_url": first["mapping_source_url"], "mapping_method": first["mapping_method"]})
    columns = ["ticker", "cik", "sec_name", "ark_name", "ticker_aliases", "funds", "n_10k", "n_10q", "n_amendments", "status", "reason", "mapping_source_url", "mapping_method"]
    return resolution, pd.DataFrame(records, columns=columns).sort_values("ticker")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="explicitly replace frozen holdings with live holdings")
    parser.add_argument("--refresh-metadata", action="store_true", help="explicitly refresh cached SEC metadata")
    args = parser.parse_args()
    raw = refresh_holdings() if args.refresh else get_frozen_holdings()
    validate_holdings(raw)
    print(raw.groupby(["fund", "date"]).size().rename("positions").to_string())
    print("Snapshot dates are preserved per fund; a mixed-date snapshot is not described as one date.")
    client = EdgarClient(SEC_USER_AGENT or None, refresh_metadata=args.refresh_metadata)
    resolution, universe = resolve_holdings(raw, client)
    atomic_csv(resolution, UNIVERSE_DIR / "holdings_resolution.csv")
    atomic_csv(universe, OUT_PATH)
    api_errors = int((resolution["status"] == "metadata_error").sum())
    unresolved = int(resolution["status"].isin(["unmatched_sec_ticker", "identity_review_required"]).sum())
    run = {"finished_at_utc": datetime.now(timezone.utc).isoformat(),
           "status": "incomplete" if api_errors else "complete_with_unresolved_mappings" if unresolved else "complete",
           "n_positions": len(raw), "n_holding_identities": len(resolution), "n_resolved_ciks": len(universe),
           "api_error_holdings": api_errors, "unresolved_holdings": unresolved,
           "status_counts": resolution["status"].value_counts().to_dict()}
    EdgarClient._atomic_write(UNIVERSE_DIR / "universe_run.json", json.dumps(run, indent=2).encode())
    print(resolution["status"].value_counts().to_string())
    print(f"Wrote {OUT_PATH}; inspect holdings_resolution.csv for every exclusion and unresolved identity.")
    return 1 if api_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
