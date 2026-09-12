"""Download auditable total-return/nominal prices and filing-specific shares.

Writes prices.csv (Adj Close), raw_prices.csv (historical nominal Close),
volume.csv (nominal shares), split_factors.csv, market_download_log.csv and
shares.csv (one row per accession, with provenance and missing-count reasons).
Raw Yahoo responses and companyfacts are cached below data/prices/.

Run with --refresh to replace frozen Yahoo snapshots. --prices-only performs
no SEC requests and is available before a genuine SEC_USER_AGENT is supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (ALT_BENCHMARK, BENCHMARK, INTERIM_DIR, PRICE_DIR,
                        SEC_USER_AGENT, VIX_TICKER)
from src.edgar import EdgarClient
from src.market import download_market_data, get_shares

# More than 61 pre-event and 63 post-event trading sessions even around holidays.
PRICE_START = "2020-09-01"
PRICE_END = "2026-04-16"  # exclusive


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def accession_fingerprint(values) -> str:
    """SHA256 of sorted unique accession strings joined with one newline."""
    return hashlib.sha256("\n".join(sorted(set(values))).encode("utf-8")).hexdigest()


def write_run(run: dict, directory: Path = PRICE_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    EdgarClient._atomic_write(directory / "market_run.json", json.dumps(run, indent=2).encode("utf-8"))


def read_metadata(path: Path) -> tuple[pd.DataFrame, str]:
    """Hash the exact bytes parsed, rejecting missing keys before acquisition."""
    payload = path.read_bytes()
    meta = pd.read_csv(io.BytesIO(payload), dtype={"cik": str, "accession": str})
    required = {"accession", "cik", "ticker", "filing_date"}
    if not required.issubset(meta.columns) or meta.empty:
        raise ValueError("Full nonempty filing metadata with accession/cik/ticker/filing_date is required")
    if meta[list(required)].isna().any().any() or meta["accession"].duplicated().any():
        raise ValueError("Null metadata keys or duplicate accessions cannot be certified")
    if any(meta[c].astype(str).str.strip().eq("").any() for c in required):
        raise ValueError("Blank metadata keys cannot be certified")
    if pd.to_datetime(meta["filing_date"], errors="coerce").isna().any():
        raise ValueError("Invalid filing dates cannot be certified")
    return meta, hashlib.sha256(payload).hexdigest()


def finalize_run(run: dict, meta: pd.DataFrame, meta_path: Path,
                 directory: Path = PRICE_DIR) -> dict:
    """Certify persisted market artifacts, retaining economic missingness.

    A missing share count is allowed only as an explicit ``missing`` row. A
    partial/trial shares file, missing ticker column, in-progress metadata change
    or absent acquisition diagnostic fails validation instead of shrinking the
    analysis corpus. Output hashes prevent subsequent silent artifact changes.
    """
    if sha256(meta_path) != run["metadata_sha256"]:
        raise ValueError("Filing metadata changed during market acquisition")
    shares = pd.read_csv(directory / "shares.csv", dtype={"accession": str, "cik": str})
    required = {"accession", "shares_status", "shares_reason", "shares_outstanding"}
    if not required.issubset(shares.columns):
        raise ValueError("Shares output lacks per-accession status/provenance fields")
    if shares["accession"].isna().any() or shares["accession"].duplicated().any():
        raise ValueError("Null or duplicate shares accessions")
    if set(shares["accession"]) != set(meta["accession"]):
        raise ValueError("Shares accession coverage differs from full filing metadata")
    if not shares["shares_status"].isin(["ok", "missing"]).all() or shares["shares_reason"].isna().any():
        raise ValueError("Each share row needs a terminal status and a reason")
    usable = shares["shares_status"].eq("ok")
    counts = pd.to_numeric(shares["shares_outstanding"], errors="coerce")
    if not counts[usable].gt(0).all():
        raise ValueError("An ok share row has no positive share count")
    diagnostics = pd.read_csv(directory / "market_download_log.csv")
    if not {"ticker", "status", "reason"}.issubset(diagnostics.columns):
        raise ValueError("Market acquisition diagnostics are missing required columns")
    requested = set(run["requested_tickers"])
    if diagnostics["ticker"].isna().any() or diagnostics["ticker"].duplicated().any() or set(diagnostics["ticker"]) != requested:
        raise ValueError("Market diagnostic ticker coverage differs from request")
    if not diagnostics["status"].isin(["ok", "failed"]).all():
        raise ValueError("Market acquisition has nonterminal ticker statuses")
    for key in ["prices", "raw_prices", "volume", "split_factors"]:
        columns = pd.read_csv(directory / f"{key}.csv", nrows=0).columns[1:]
        if not requested.issubset(columns):
            raise ValueError(f"{key}.csv lacks requested ticker columns")
    failed = diagnostics.loc[diagnostics["status"].eq("failed"), "ticker"].tolist()
    if BENCHMARK in failed:
        raise ValueError("SPY benchmark acquisition failed")
    run.update(status="complete_with_missing_data" if failed or not usable.all() else "complete",
        completed_at_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        shares_accession_count=len(shares), shares_accession_sha256=accession_fingerprint(shares["accession"]),
        shares_status_counts={str(k): int(v) for k, v in shares["shares_status"].value_counts().items()},
        market_summary={"source": "Yahoo Finance via yfinance", "ticker_count": len(diagnostics),
            "usable_histories": int(diagnostics["status"].eq("ok").sum()), "failed_tickers": failed,
            "analysis_start": PRICE_START, "analysis_end_exclusive": PRICE_END,
            "nominal_units": "Undo every later split through each frozen snapshot retrieval date"},
        output_sha256={name: sha256(directory / name) for name in ["prices.csv", "raw_prices.csv", "volume.csv",
            "split_factors.csv", "shares.csv", "market_download_log.csv"]})
    write_run(run, directory)
    return run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="refresh frozen Yahoo snapshots")
    parser.add_argument("--prices-only", action="store_true", help="do not contact the SEC")
    parser.add_argument("--verify-existing", action="store_true",
                        help="validate complete existing artifacts and write provenance, without network requests")
    args = parser.parse_args()
    if args.verify_existing and (args.prices_only or args.refresh):
        parser.error("--verify-existing cannot be combined with --prices-only or --refresh")
    meta_path = INTERIM_DIR / "filings_meta.csv"
    meta, metadata_hash = read_metadata(meta_path)
    tickers = sorted(set(meta["ticker"]) | {BENCHMARK, ALT_BENCHMARK, VIX_TICKER})
    run = {"schema_version": 1, "status": "running", "started_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "metadata_sha256": metadata_hash, "metadata_accession_sha256": accession_fingerprint(meta["accession"]),
        "metadata_accession_count": len(meta), "requested_tickers": tickers,
        "prices_only": args.prices_only, "verification_of_existing_files": args.verify_existing}
    write_run(run)
    if args.verify_existing:
        try:
            final = finalize_run(run, meta, meta_path)
        except Exception as exc:
            run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            write_run(run)
            raise
        print(f"Verified {final['shares_accession_count']} accession rows; {final['status']}")
        return 0
    # Validate identification before a long job that otherwise ends at shares.
    client = None if args.prices_only else EdgarClient(SEC_USER_AGENT or None)
    result = download_market_data(tickers, PRICE_START, PRICE_END, refresh=args.refresh)
    diagnostics = result["diagnostics"]
    failed = diagnostics.loc[diagnostics["status"] != "ok", "ticker"].tolist()
    print(f"Market histories: {len(tickers) - len(failed)}/{len(tickers)} usable; failures: {failed}")
    if client is not None:
        shares = get_shares(client, meta)
        shares.to_csv(PRICE_DIR / "shares.csv", index=False)
        matched = int(shares["shares_status"].eq("ok").sum())
        print(f"Shares: {matched}/{len(shares)} filings; all missing counts retained in shares.csv")
    if BENCHMARK in failed:
        print("SPY download failed; event construction must wait for benchmark data.")
        run.update(status="failed", error="SPY benchmark acquisition failed")
        write_run(run)
        return 1
    if args.prices_only:
        run.update(status="prices_only", completed_at_utc=pd.Timestamp.now(tz="UTC").isoformat())
        write_run(run)
    else:
        finalize_run(run, meta, meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
