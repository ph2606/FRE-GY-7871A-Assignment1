"""A polite, cached SEC EDGAR client.

Documents and JSON metadata are cached with URL, retrieval timestamp and SHA256
provenance. The acceptanceDateTime field is kept in UTC for event alignment.
"""

from __future__ import annotations

import json
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from .config import (
    FILING_DIR,
    SEC_MAX_REQUESTS_PER_SEC,
    SEC_USER_AGENT,
)

_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
_COMPANY_TICKERS = "https://www.sec.gov/files/company_tickers.json"
_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"


class EdgarClient:
    """Rate-limited EDGAR session with an on-disk cache of every document."""

    def __init__(self, user_agent: str | None = None, cache_dir: Path | None = None,
                 refresh_metadata: bool = False):
        ua = user_agent or SEC_USER_AGENT
        if not ua or "@" not in ua:
            raise RuntimeError(
                "Set a real SEC_USER_AGENT, e.g.\n"
                '    $env:SEC_USER_AGENT = "Jane Doe jd123@nyu.edu"   (PowerShell)\n'
                '    export SEC_USER_AGENT="Jane Doe jd123@nyu.edu"   (bash)\n'
                "The SEC blocks unidentified traffic and it is their site, not ours."
            )
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
        self.cache_dir = Path(cache_dir or FILING_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._min_interval = 1.0 / SEC_MAX_REQUESTS_PER_SEC
        self._last_call = 0.0
        self.refresh_metadata = refresh_metadata

    @staticmethod
    def _replace_with_retry(temporary: Path, destination: Path) -> None:
        """Allow brief OneDrive/antivirus locks; preserve temp file on failure."""
        for attempt in range(8):
            try:
                temporary.replace(destination)
                return
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(min(0.1 * 2 ** attempt, 1.0))

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(data)
        EdgarClient._replace_with_retry(temporary, path)

    def _save_response(self, path: Path, data: bytes, url: str) -> None:
        self._atomic_write(path, data)
        provenance = {
            "url": url,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
        self._atomic_write(path.with_name(path.name + ".provenance.json"),
                           json.dumps(provenance, indent=2).encode("utf-8"))

    def get_json(self, url: str) -> dict:
        """Use the first downloaded metadata snapshot unless refresh is explicit."""
        path = self.cache_dir / "metadata" / (hashlib.sha256(url.encode()).hexdigest() + ".json")
        if path.exists() and not self.refresh_metadata:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                pass  # An interrupted/invalid cache is not evidence.
        response = self._get(url)
        payload = response.json()  # Validate before committing a cache entry.
        self._save_response(path, json.dumps(payload).encode("utf-8"), url)
        return payload

    # -- low level -----------------------------------------------------------
    def _get(self, url: str, timeout: int = 60) -> requests.Response:
        for attempt in range(4):
            wait = self._min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                resp = self.session.get(url, timeout=timeout)
            except (requests.ConnectionError, requests.Timeout):
                self._last_call = time.monotonic()
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
                continue
            self._last_call = time.monotonic()
            if resp.status_code == 200:
                return resp
            if resp.status_code in (403, 429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
        raise RuntimeError(f"Unable to retrieve {url}")

    # -- metadata ------------------------------------------------------------
    def ticker_to_cik(self) -> dict[str, str]:
        """{TICKER: 10-digit zero-padded CIK} for every SEC-registered filer."""
        return {r["ticker"]: r["cik"] for r in self.company_ticker_records()}

    def company_ticker_records(self) -> list[dict]:
        """SEC ticker records including issuer names for collision checks."""
        payload = self.get_json(_COMPANY_TICKERS)
        return [{"ticker": v["ticker"].upper(), "cik": str(v["cik_str"]).zfill(10),
                 "sec_name": v["title"]} for v in payload.values()]

    def submissions(self, cik: str) -> dict:
        """Full submission history for one CIK, including the older paged files."""
        cik = str(cik).zfill(10)
        payload = self.get_json(_SUBMISSIONS.format(cik=cik))
        recent = payload["filings"]["recent"]
        frames = [pd.DataFrame(recent)]
        for extra in payload["filings"].get("files", []):
            url = f"https://data.sec.gov/submissions/{extra['name']}"
            frames.append(pd.DataFrame(self.get_json(url)))
        frames = [f.dropna(axis=1, how="all") for f in frames if not f.empty]
        if not frames:
            payload["_filings"] = pd.DataFrame()
        elif len(frames) == 1:
            payload["_filings"] = frames[0]
        else:
            payload["_filings"] = pd.concat(frames, ignore_index=True)
        if not payload["_filings"].empty:
            payload["_filings"] = payload["_filings"].drop_duplicates("accessionNumber")
        return payload

    def list_filings(
        self,
        cik: str,
        forms: list[str],
        start: str,
        end: str,
        include_amendments: bool = False,
    ) -> pd.DataFrame:
        """Filing metadata for one CIK, filtered to `forms` and the date window.

        Columns: cik, company, sic, sic_desc, form, filing_date, report_date,
                 acceptance_datetime, accession, primary_document, doc_url.

        `acceptance_datetime` is UTC and is the timestamp EDGAR actually received
        the document. You need it: a filing accepted at 21:30 UTC on a Friday was
        not tradable on that Friday.
        """
        payload = self.submissions(cik)
        df = payload["_filings"].copy()
        if df.empty:
            return df

        if include_amendments:
            keep = df["form"].isin(set(forms) | {form + "/A" for form in forms})
        else:
            keep = df["form"].isin(forms)
        df = df[keep]
        df = df[(df["filingDate"] >= start) & (df["filingDate"] <= end)]
        if df.empty:
            return pd.DataFrame()

        cik_int = int(cik)
        out = pd.DataFrame({
            "cik": str(cik).zfill(10),
            "company": payload.get("name"),
            "sic": payload.get("sic"),
            "sic_desc": payload.get("sicDescription"),
            "form": df["form"].values,
            "filing_date": pd.to_datetime(df["filingDate"].values),
            "report_date": pd.to_datetime(df["reportDate"].values, errors="coerce"),
            "acceptance_datetime": pd.to_datetime(df["acceptanceDateTime"].values, errors="coerce", utc=True),
            "accession": df["accessionNumber"].values,
            "primary_document": df["primaryDocument"].values,
        })
        out["doc_url"] = [
            _ARCHIVE.format(cik_int=cik_int, acc_nodash=a.replace("-", ""), doc=d)
            for a, d in zip(out["accession"], out["primary_document"])
        ]
        return out.sort_values(["filing_date", "acceptance_datetime", "accession"]).reset_index(drop=True)

    # -- documents -----------------------------------------------------------
    def fetch_document(self, url: str, accession: str) -> str:
        """Download a filing's primary document, caching it under data/filings/."""
        path = self.cache_dir / f"{accession.replace('-', '')}.html"
        if path.exists():
            cached = path.read_text(encoding="utf-8", errors="replace")
            if cached.strip():
                return cached
        text = self._get(url, timeout=180).text
        if not text.strip():
            raise ValueError(f"Empty filing response: {url}")
        self._save_response(path, text.encode("utf-8", errors="replace"), url)
        return text
