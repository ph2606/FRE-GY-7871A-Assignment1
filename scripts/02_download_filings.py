"""Download candidate filings with an auditable manifest and versioned text cache.

The manifest includes amendments, skipped before fetching, and every failed
candidate. ``filings_meta.csv`` contains successfully parsed original filings.
``--limit N`` writes *_trial files and cannot overwrite a full-run manifest.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FILING_DIR, FORMS, INTERIM_DIR, ROOT, SAMPLE_END, SAMPLE_START, SEC_USER_AGENT, UNIVERSE_DIR
from src.edgar import EdgarClient
from src.parse import PARSER_VERSION, html_to_text, tokenize

TEXT_DIR = INTERIM_DIR / "text"
META_PATH = INTERIM_DIR / "filings_meta.csv"
PARSER_SHA256 = hashlib.sha256((ROOT / "src" / "parse.py").read_bytes()).hexdigest()
MANIFEST_COLUMNS = ["ticker", "cik", "company", "sic", "sic_desc", "form", "filing_date",
                    "report_date", "acceptance_datetime", "accession", "primary_document", "doc_url",
                    "parse_status", "error", "n_words", "n_distinct", "text_path", "text_sha256",
                    "parser_version", "parser_sha256"]
FAILURE_COLUMNS = ["ticker", "cik", "accession", "stage", "error"]


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    EdgarClient._replace_with_retry(temporary, path)


def cached_text(client: EdgarClient, filing: dict, text_dir: Path = TEXT_DIR) -> tuple[str, dict]:
    """A text cache is usable only with matching parser and content hashes."""
    accession = filing["accession"].replace("-", "")
    text_path = text_dir / f"{accession}.txt.gz"
    sidecar = text_path.with_name(text_path.name + ".json")
    if text_path.exists() and sidecar.exists():
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            with gzip.open(text_path, "rt", encoding="utf-8") as handle:
                text = handle.read()
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if text.strip() and metadata.get("parser_sha256") == PARSER_SHA256 and metadata.get("text_sha256") == digest:
                return text, metadata
        except (OSError, ValueError, EOFError):
            pass
    raw = client.fetch_document(filing["doc_url"], filing["accession"])
    text = html_to_text(raw)
    if not text.strip() or not tokenize(text):
        raise ValueError("Filing produced no narrative tokens")
    metadata = {"parser_version": PARSER_VERSION, "parser_sha256": PARSER_SHA256,
                "source_url": filing["doc_url"],
                "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "parsed_at_utc": datetime.now(timezone.utc).isoformat()}
    text_dir.mkdir(parents=True, exist_ok=True)
    # gzip.compress(mtime=0) also makes identical text deterministic on disk.
    EdgarClient._atomic_write(text_path, gzip.compress(text.encode("utf-8"), mtime=0))
    EdgarClient._atomic_write(sidecar, json.dumps(metadata, indent=2).encode("utf-8"))
    return text, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="trial with first N distinct CIKs; writes *_trial files")
    parser.add_argument("--drop-html", action="store_true", help="delete raw HTML only after validated text is cached")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    suffix = "_trial" if args.limit is not None else ""
    meta_path = INTERIM_DIR / f"filings_meta{suffix}.csv"
    manifest_path = INTERIM_DIR / f"filings_manifest{suffix}.csv"
    failures_path = INTERIM_DIR / f"acquisition_failures{suffix}.csv"
    run_path = INTERIM_DIR / f"acquisition_run{suffix}.json"
    universe = pd.read_csv(UNIVERSE_DIR / "universe.csv", dtype={"cik": str})
    if (universe["status"] == "metadata_error").any():
        raise RuntimeError("Universe contains SEC metadata errors; resolve them before enumerating the filing baseline")
    universe = universe[universe["status"] == "domestic_filer"].sort_values(["cik", "ticker"])
    universe = universe.drop_duplicates("cik")
    n_universe = len(universe)
    if args.limit is not None:
        universe = universe.head(args.limit)
    client = EdgarClient(SEC_USER_AGENT or None)
    candidates, failures = [], []
    run = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "in_progress",
           "is_trial": args.limit is not None, "limit": args.limit, "n_universe_ciks": n_universe,
           "n_requested_ciks": len(universe), "parser_version": PARSER_VERSION,
           "parser_sha256": PARSER_SHA256, "sample_start": SAMPLE_START, "sample_end": SAMPLE_END}
    EdgarClient._atomic_write(run_path, json.dumps(run, indent=2).encode())

    def checkpoint() -> pd.DataFrame:
        frame = pd.DataFrame(candidates, columns=MANIFEST_COLUMNS)
        for column in ["company", "sic_desc", "error"]:
            frame[column] = frame[column].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        atomic_csv(frame, manifest_path)
        atomic_csv(frame[frame["parse_status"] == "ok"], meta_path)
        atomic_csv(pd.DataFrame(failures, columns=FAILURE_COLUMNS), failures_path)
        return frame

    # Enumerate the baseline first; errors in listing a company make the run
    # incomplete because its absent candidates cannot be counted as zero.
    for _, firm in universe.iterrows():
        try:
            filings = client.list_filings(firm["cik"], FORMS, SAMPLE_START, SAMPLE_END, include_amendments=True)
            for _, filing in filings.iterrows():
                row = filing.to_dict()
                row.update(ticker=firm["ticker"], parse_status="excluded_amendment" if row["form"].endswith("/A") else "pending",
                           error="", n_words=None, n_distinct=None, text_path=None,
                           text_sha256=None, parser_version=PARSER_VERSION, parser_sha256=PARSER_SHA256)
                candidates.append(row)
        except Exception as exc:
            failures.append({"ticker": firm["ticker"], "cik": firm["cik"], "accession": "",
                             "stage": "list_filings", "error": f"{type(exc).__name__}: {exc}"})
        checkpoint()
    accessions = [candidate["accession"] for candidate in candidates]
    if len(accessions) != len(set(accessions)):
        raise ValueError("Duplicate accession in candidate manifest; resolve issuer mapping before downloading")
    print(f"Enumerated {len(candidates)} candidates from {len(universe)} CIKs; amendments remain in manifest.", flush=True)
    for index, candidate in enumerate(candidates, 1):
        if candidate["parse_status"] == "excluded_amendment":
            continue
        try:
            text, provenance = cached_text(client, candidate)
            tokens = tokenize(text)
            candidate.update(parse_status="ok", n_words=len(tokens), n_distinct=len(set(tokens)),
                             text_path=(TEXT_DIR / f"{candidate['accession'].replace('-', '')}.txt.gz").relative_to(ROOT).as_posix(),
                             text_sha256=provenance["text_sha256"])
            if args.drop_html:
                try:
                    (FILING_DIR / f"{candidate['accession'].replace('-', '')}.html").unlink(missing_ok=True)
                except OSError:
                    pass  # A Windows file lock must never drop good text.
        except Exception as exc:
            candidate.update(parse_status="failed", error=f"{type(exc).__name__}: {exc}")
            failures.append({"ticker": candidate["ticker"], "cik": candidate["cik"],
                             "accession": candidate["accession"], "stage": "download_or_parse", "error": candidate["error"]})
        if index % 10 == 0:
            checkpoint()
            print(f"[{index}/{len(candidates)}] {candidate['ticker']} {candidate['accession']} {candidate['parse_status']}", flush=True)
    frame = checkpoint()
    n_listing_failures = sum(failure["stage"] == "list_filings" for failure in failures)
    n_document_failures = len(failures) - n_listing_failures
    # Listing failures leave an unknown baseline. Document failures leave known
    # candidates that can be excluded only after an explicit failure audit.
    if n_listing_failures:
        completion_status = "incomplete"
    elif n_document_failures:
        completion_status = "complete_with_parse_failures"
    else:
        completion_status = "trial_complete" if args.limit is not None else "complete"
    run.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),
               status=completion_status,
               n_candidates=len(frame), n_parsed=int((frame["parse_status"] == "ok").sum()),
               n_amendments=int((frame["parse_status"] == "excluded_amendment").sum()),
               n_failures=len(failures), n_listing_failures=n_listing_failures,
               n_document_failures=n_document_failures,
               parse_failure_audit_required=bool(n_document_failures),
               status_counts=frame["parse_status"].value_counts().to_dict())
    EdgarClient._atomic_write(run_path, json.dumps(run, indent=2).encode())
    assert len(pd.read_csv(manifest_path)) == len(frame)
    print(json.dumps(run, indent=2), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
