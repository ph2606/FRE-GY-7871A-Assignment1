"""Offline regression checks for acquisition bugs; fixtures are synthetic, not results."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import requests

from src.edgar import EdgarClient
from src.parse import html_to_text, tokenize

ROOT = Path(__file__).resolve().parents[1]


def load_script(filename):
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_visible_inline_narrative_survives_hidden_scaffolding():
    raw = """<html><body><ix:header><ix:hidden>NOT NARRATIVE</ix:hidden></ix:header>
    <p>Before</p><ix:nonNumeric name="us-gaap:RiskFactorsTextBlock">LOSS RISK UNCERTAIN</ix:nonNumeric>
    <p style="display: none">HIDDEN MORE</p><p hidden>HIDDEN</p><p>After</p></body></html>"""
    assert html_to_text(raw) == "Before LOSS RISK UNCERTAIN After"


def test_visible_numeric_facts_still_inform_table_filter():
    raw = '<table><tr><td>LOSS</td><td><ix:nonFraction>1234567890</ix:nonFraction></td></tr></table><p>RISK</p>'
    assert html_to_text(raw) == "RISK"
    assert "1234567890" in html_to_text(raw, drop_numeric_tables=False)


def test_tokenization_matches_dictionary_words_at_punctuation():
    assert tokenize("risk-related loss-making company's uncertain 123 A") == [
        "RISK", "RELATED", "LOSS", "MAKING", "COMPANY", "UNCERTAIN"]


def test_metadata_cache_records_provenance_and_avoids_repeat_request(tmp_path, monkeypatch):
    client = EdgarClient("Test test@example.com", cache_dir=tmp_path)
    calls = []
    class Response:
        def json(self):
            return {"test": "metadata"}
    monkeypatch.setattr(client, "_get", lambda url: calls.append(url) or Response())
    url = "https://data.sec.gov/submissions/CIK0000000001.json"
    assert client.get_json(url) == client.get_json(url) == {"test": "metadata"}
    assert calls == [url]
    path = tmp_path / "metadata" / (hashlib.sha256(url.encode()).hexdigest() + ".json")
    provenance = json.loads(path.with_name(path.name + ".provenance.json").read_text())
    assert provenance["url"] == url
    assert provenance["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_network_timeout_is_retried(tmp_path, monkeypatch):
    client = EdgarClient("Test test@example.com", cache_dir=tmp_path)
    responses = [requests.Timeout("temporary"), type("Response", (), {"status_code": 200})()]
    def get(*args, **kwargs):
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(client.session, "get", get)
    monkeypatch.setattr("src.edgar.time.sleep", lambda seconds: None)
    assert client._get("https://data.sec.gov/example").status_code == 200
    assert not responses


def test_atomic_write_recovers_from_brief_windows_lock(tmp_path, monkeypatch):
    original_replace = Path.replace
    attempts = []
    def replace(path, destination):
        attempts.append(path)
        if len(attempts) < 3:
            raise PermissionError("fixture OneDrive lock")
        return original_replace(path, destination)
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr("src.edgar.time.sleep", lambda seconds: None)
    destination = tmp_path / "metadata.json"
    destination.write_bytes(b"old")
    EdgarClient._atomic_write(destination, b"new")
    assert destination.read_bytes() == b"new" and len(attempts) == 3
    assert not (tmp_path / "metadata.json.tmp").exists()


def test_atomic_write_preserves_old_file_and_temp_on_persistent_lock(tmp_path, monkeypatch):
    attempts = []
    def replace(path, destination):
        attempts.append(path)
        raise PermissionError("fixture persistent lock")
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr("src.edgar.time.sleep", lambda seconds: None)
    destination = tmp_path / "metadata.json"
    destination.write_bytes(b"old")
    with pytest.raises(PermissionError):
        EdgarClient._atomic_write(destination, b"new")
    assert destination.read_bytes() == b"old"
    assert (tmp_path / "metadata.json.tmp").read_bytes() == b"new"
    assert len(attempts) == 8


def test_mapping_keeps_every_identity_and_deduplicates_cik():
    module = load_script("01_build_universe.py")
    identities = [("AIR", "AIRBUS SE"), ("DSY FP", "DASSAULT SYSTEMES SE"),
                  ("GOOG", "ALPHABET INC"), ("GOOGL", "ALPHABET INC"),
                  ("1234", "NUMERIC LISTING"), ("UNKNOWN", "UNMAPPED COMPANY")]
    raw = pd.DataFrame([{"date": "09/04/2026", "fund": fund, "ticker": ticker,
                         "company": company, "cusip": f"fixture{index}"}
                        for index, (fund, (ticker, company)) in enumerate(zip(module.ARK_FUNDS, identities))])
    class Client:
        calls = []
        def company_ticker_records(self):
            return [{"ticker": "AIR", "cik": "0000000001", "sec_name": "AAR CORP"},
                    {"ticker": "DSY", "cik": "0000000003", "sec_name": "UNRELATED ISSUER"},
                    {"ticker": "GOOG", "cik": "0000000002", "sec_name": "Alphabet Inc."},
                    {"ticker": "GOOGL", "cik": "0000000002", "sec_name": "Alphabet Inc."}]
        def list_filings(self, cik, *args, **kwargs):
            self.calls.append(cik)
            return pd.DataFrame({"form": ["10-K", "10-Q/A"]})
    client = Client()
    resolution, universe = module.resolve_holdings(raw, client)
    assert len(resolution) == 6 and resolution["n_positions"].sum() == 6
    status = resolution.set_index("raw_ticker")["status"].to_dict()
    assert status["AIR"] == "identity_review_required"
    assert status["DSY FP"] == "excluded_foreign_exchange"
    assert status["UNKNOWN"] == "unmatched_sec_ticker"
    assert status["1234"] == "excluded_numeric_listing"
    assert client.calls == ["0000000002"]
    assert len(universe) == 1 and universe.iloc[0]["ticker_aliases"] == "GOOG|GOOGL"
    assert universe.iloc[0]["n_amendments"] == 1


def test_metadata_error_is_not_reported_as_no_filings():
    module = load_script("01_build_universe.py")
    raw = pd.DataFrame([{"date": "09/04/2026", "fund": fund, "ticker": "EXAMPLE",
                         "company": "EXAMPLE CORP", "cusip": "fixture"} for fund in module.ARK_FUNDS])
    class Client:
        def company_ticker_records(self):
            return [{"ticker": "EXAMPLE", "cik": "0000000001", "sec_name": "EXAMPLE CORP"}]
        def list_filings(self, *args, **kwargs):
            raise requests.Timeout("fixture failure")
    resolution, universe = module.resolve_holdings(raw, Client())
    assert set(resolution["status"]) == {"metadata_error"}
    assert universe.iloc[0]["status"] == "metadata_error"
    assert pd.isna(universe.iloc[0]["n_10k"])


def test_reviewed_cik_is_queried_and_not_assumed_ineligible():
    module = load_script("01_build_universe.py")
    raw = pd.DataFrame([{"date": "09/04/2026", "fund": fund, "ticker": "SE",
                         "company": "SEA LTD-ADR", "cusip": "81141R100"} for fund in module.ARK_FUNDS])
    class Client:
        calls = []
        def company_ticker_records(self):
            return []
        def list_filings(self, cik, *args, **kwargs):
            self.calls.append(cik)
            # A synthetic eligible result proves the override does not assume
            # foreign issuers have no 10-K/Q. This is not an actual Sea filing.
            return pd.DataFrame({"form": ["10-Q"], "company": ["Sea fixture issuer"]})
    client = Client()
    resolution, universe = module.resolve_holdings(raw, client)
    assert client.calls == ["0001703399"]
    assert universe.iloc[0]["status"] == "domestic_filer"
    assert universe.iloc[0]["n_10q"] == 1
    assert resolution.iloc[0]["mapping_method"] == "documented_manual_review"
    assert "1703399" in resolution.iloc[0]["mapping_source_url"]


def test_reviewed_identity_requires_matching_holding_identifier():
    module = load_script("01_build_universe.py")
    raw = pd.DataFrame([{"date": "09/04/2026", "fund": fund, "ticker": "AIR",
                         "company": "UNRELATED NEW ISSUER", "cusip": "different"} for fund in module.ARK_FUNDS])
    class Client:
        def company_ticker_records(self):
            return []
    resolution, universe = module.resolve_holdings(raw, Client())
    assert resolution.iloc[0]["status"] == "unmatched_sec_ticker"
    assert universe.empty


def test_text_cache_invalidates_changed_parser_and_corruption(tmp_path):
    module = load_script("02_download_filings.py")
    class Client:
        calls = 0
        def fetch_document(self, *args):
            self.calls += 1
            return "<p>LOSS RISK</p>"
    client = Client()
    filing = {"accession": "0000000001-21-000001", "doc_url": "https://example.com/fixture"}
    text, provenance = module.cached_text(client, filing, tmp_path)
    assert text == "LOSS RISK"
    module.cached_text(client, filing, tmp_path)
    assert client.calls == 1
    sidecar = next(tmp_path.glob("*.json"))
    provenance["parser_sha256"] = "old-parser"
    sidecar.write_text(json.dumps(provenance))
    module.cached_text(client, filing, tmp_path)
    assert client.calls == 2
    next(tmp_path.glob("*.gz")).write_bytes(b"corrupt-cache")
    module.cached_text(client, filing, tmp_path)
    assert client.calls == 3


@pytest.mark.parametrize("listing_error", [False, True])
def test_trial_manifest_includes_amendments_and_failures_without_overwriting_full(tmp_path, monkeypatch, listing_error):
    module = load_script("02_download_filings.py")
    universe_dir, interim = tmp_path / "universe", tmp_path / "interim"
    universe_dir.mkdir()
    interim.mkdir()
    pd.DataFrame([{"ticker": "EXAMPLE", "cik": "0000000001", "status": "domestic_filer"}]).to_csv(universe_dir / "universe.csv", index=False)
    (interim / "filings_meta.csv").write_text("full-run-is-preserved")
    monkeypatch.setattr(module, "UNIVERSE_DIR", universe_dir)
    monkeypatch.setattr(module, "INTERIM_DIR", interim)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "TEXT_DIR", interim / "text")
    monkeypatch.setattr(module, "SEC_USER_AGENT", "Test test@example.com")
    class Client(EdgarClient):
        def __init__(self, *args, **kwargs):
            pass
        def list_filings(self, *args, **kwargs):
            if listing_error:
                raise requests.Timeout("fixture listing failure; baseline unknown")
            return pd.DataFrame([{"cik": "0000000001", "accession": accession, "form": form,
                                  "doc_url": "https://example.com/fixture"}
                                 for accession, form in [("original", "10-Q"), ("amended", "10-Q/A"), ("failure", "10-K")]])
    monkeypatch.setattr(module, "EdgarClient", Client)
    def get_text(client, filing):
        if filing["accession"] == "failure":
            raise requests.Timeout("fixture download failure")
        return "LOSS RISK", {"text_sha256": "fixture-hash"}
    monkeypatch.setattr(module, "cached_text", get_text)
    monkeypatch.setattr("sys.argv", ["02_download_filings.py", "--limit", "1"])
    assert module.main() == 1
    manifest = pd.read_csv(interim / "filings_manifest_trial.csv")
    assert manifest.set_index("accession")["parse_status"].to_dict() == ({} if listing_error else {
        "original": "ok", "amended": "excluded_amendment", "failure": "failed"})
    assert len(pd.read_csv(interim / "filings_meta_trial.csv")) == (0 if listing_error else 1)
    run = json.loads((interim / "acquisition_run_trial.json").read_text())
    assert run["is_trial"]
    assert run["status"] == ("incomplete" if listing_error else "complete_with_parse_failures")
    assert run["n_candidates"] == (0 if listing_error else 3)
    assert run["n_listing_failures"] == int(listing_error)
    assert run["parse_failure_audit_required"] is (not listing_error)
    assert (interim / "filings_meta.csv").read_text() == "full-run-is-preserved"
