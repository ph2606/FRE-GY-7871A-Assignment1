"""Local analysis orchestration, explicit sample waterfall, and provenance."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import (MEASURES, add_time, quarterly_means, summary_statistics,
                       trend_tests, volatility_tests, return_tests, figure_quarterly)
from .lexicons import load_master_dictionary, lm_word_lists
from .scoring import score_corpus

REQUIRED = ["data/universe/holdings_resolution.csv", "data/universe/universe.csv",
            "data/interim/acquisition_run.json", "data/interim/filings_manifest.csv",
            "data/interim/filings_meta.csv", "data/prices/prices.csv",
            "data/prices/raw_prices.csv", "data/prices/volume.csv", "data/prices/shares.csv",
            "data/prices/market_run.json",
            "data/lexicons/LoughranMcDonald_MasterDictionary.csv"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def readiness(root):
    root = Path(root)
    missing = [name for name in REQUIRED if not (root / name).exists()]
    issues = []
    path = root / "data/interim/acquisition_run.json"
    if path.exists():
        run = json.loads(path.read_text())
        if run.get("limit") is not None or run.get("status") not in ["complete", "complete_with_parse_failures"]:
            issues.append("Full acquisition has not completed: " + str(run))
    market_path = root / "data/prices/market_run.json"
    meta_path = root / "data/interim/filings_meta.csv"
    if market_path.exists() and meta_path.exists():
        market = json.loads(market_path.read_text())
        if market.get("status") not in ["complete", "complete_with_missing_data"] or market.get("prices_only"):
            issues.append("Market acquisition is not terminal with shares")
        if market.get("metadata_sha256") != sha256(meta_path):
            issues.append("Market data refer to different filing metadata; rerun script 03")
        for name, digest in market.get("output_sha256", {}).items():
            artifact = root / "data/prices" / name
            if not artifact.exists() or sha256(artifact) != digest:
                issues.append("Market artifact changed after validation: " + name)
        if market.get("shares_accession_sha256") != market.get("metadata_accession_sha256"):
            issues.append("Shares were not attempted for every parsed accession")
    return {"ready": not missing and not issues, "missing_files": missing, "issues": issues}


def sample_waterfall(manifest, events):
    """Exclusive ordered losses; amendments and parsing are one assignment stage.

    Market completeness and controls follow the five class filters and are reported
    explicitly. A common complete-case corpus keeps paired regressions and IDF
    population identical. Selection on future price availability is a limitation.
    """
    m = manifest.copy()
    if m.accession.duplicated().any():
        raise ValueError("Candidate manifest contains duplicate accessions")
    dates = pd.to_datetime(m.filing_date, errors="raise")
    if not dates.between("2021-01-01", "2025-12-31").all():
        raise ValueError("Out-of-window candidate filing date")
    m["quarter"] = dates.dt.to_period("Q").astype(str)
    m["filing_date"] = dates
    m["cik"] = m.cik.astype(str).str.zfill(10)
    for field in ["n_words", "n_distinct"]:
        m[field] = pd.to_numeric(m[field], errors="coerce")
    m = m.sort_values(["filing_date", "acceptance_datetime", "accession"], na_position="last")
    ledger = m[["accession", "ticker", "cik", "form", "filing_date", "quarter"]].copy()
    ledger["exclusion_stage"] = "retained"
    stages = [{"step": 0, "filter": "All discovered in-window 10-K/Q and amendments", "entering": len(m), "removed": 0, "remaining": len(m)}]

    def apply(label, keep):
        nonlocal m
        keep = pd.Series(keep, index=m.index).fillna(False).astype(bool)
        before = len(m)
        ledger.loc[m.index[~keep], "exclusion_stage"] = label
        m = m.loc[keep].copy()
        stages.append({"step": len(stages), "filter": label, "entering": before, "removed": before-len(m), "remaining": len(m)})

    apply("1 Amendments or failed parsing", m.form.isin(["10-K", "10-Q"]) & m.parse_status.eq("ok"))
    apply("2 Minimum words (10-K 2000; 10-Q 1000)", m.n_words.ge(np.where(m.form.eq("10-K"), 2000, 1000)))
    apply("3 Earliest filing per CIK/calendar quarter", ~m.duplicated(["cik", "quarter"], keep="first"))
    if events.accession.duplicated().any():
        raise ValueError("Duplicate accession in events")
    e = events.set_index("accession")
    for col in ["pass_day0_price", "pass_return_history", "pass_complete_windows", "pass_shares", "pass_controls"]:
        m[col] = m.accession.map(e[col]).fillna(False).astype(bool)
    apply("4 Usable day 0 and nominal day -1 price at least $3", m.pass_day0_price)
    apply("5 At least 60 returns before and after day 0", m.pass_return_history)
    apply("6 Complete prescribed stock/benchmark windows", m.pass_complete_windows)
    apply("7 Filing-specific contemporaneous shares", m.pass_shares)
    apply("8 All regression controls observed", m.pass_controls)
    assert sum(row["removed"] for row in stages) + len(m) == len(ledger)
    # Keep metadata authority for shared columns; append event-specific information.
    new_cols = [c for c in events.columns if c not in m.columns or c == "accession"]
    final = m.merge(events[new_cols], on="accession", how="left", validate="one_to_one")
    return final, pd.DataFrame(stages), ledger


def _read_prices(path):
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if frame.index.duplicated().any() or not frame.index.is_monotonic_increasing:
        raise ValueError(f"Invalid daily index in {path}")
    return frame


def run_analysis(root, *, save=True):
    root = Path(root)
    status = readiness(root)
    if not status["ready"]:
        raise RuntimeError("Analysis input acquisition is incomplete: " + json.dumps(status))
    input_hashes = {name: sha256(root/name) for name in REQUIRED}
    from .events import build_event_panel
    manifest = pd.read_csv(root / "data/interim/filings_manifest.csv", dtype={"cik": str})
    meta = pd.read_csv(root / "data/interim/filings_meta.csv", dtype={"cik": str})
    prices = _read_prices(root / "data/prices/prices.csv")
    raw_prices = _read_prices(root / "data/prices/raw_prices.csv")
    volume = _read_prices(root / "data/prices/volume.csv")
    shares = pd.read_csv(root / "data/prices/shares.csv", dtype={"cik": str})
    events = build_event_panel(meta, prices, raw_prices, volume, shares)
    final, table1, ledger = sample_waterfall(manifest, events)
    if len(final) < 12:
        raise RuntimeError(f"Only {len(final)} eligible filings; inspect waterfall and acquisition before estimating")
    master = load_master_dictionary(root / "data/lexicons/LoughranMcDonald_MasterDictionary.csv")
    lists = {k: v for k, v in lm_word_lists(master).items() if k in ["Negative", "Uncertainty"]}
    scored, terms, scoring_diagnostics = score_corpus(final, lists, root)
    scored = add_time(scored)
    assert set(scored.accession) == set(final.accession)
    table2 = summary_statistics(scored)
    table3 = terms.sort_values(["category", "count", "word"], ascending=[True, False, True]).groupby("category", sort=False).head(30).copy()
    quarters = quarterly_means(scored)
    table4 = trend_tests(scored)
    table5 = volatility_tests(scored)
    table6 = return_tests(scored)
    # Explicit starter-list sensitivity, identical documents and all other choices.
    old_lists = {k: v for k, v in lm_word_lists(master, include_removed=True).items() if k in lists}
    old_scored, _, _ = score_corpus(final, old_lists, root)
    old_scored = add_time(old_scored)
    sensitivity4, sensitivity6 = trend_tests(old_scored), return_tests(old_scored)
    correlations = pd.DataFrame([
        {"weighting": "proportion", "correlation": scored.negative_proportion.corr(scored.uncertainty_proportion)},
        {"weighting": "tfidf", "correlation": scored.negative_tfidf.corr(scored.uncertainty_tfidf)}])
    concentration = terms.sort_values(["category", "count", "word"], ascending=[True, False, True]).groupby("category").head(10).groupby("category").agg(top10_share=("share", "sum"))
    # Select, identify, and excerpt actual discordant filings for human close reading.
    ranks = scored[["negative_proportion", "uncertainty_proportion"]].rank(pct=True)
    delta = ranks.negative_proportion - ranks.uncertainty_proportion
    example_ids = list(dict.fromkeys([delta.idxmax(), delta.idxmin()]))
    examples = scored.loc[example_ids, ["accession", "ticker", "form", "filing_date", "doc_url", "text_path", *MEASURES]].copy()
    excerpts = []
    for _, row in examples.iterrows():
        with gzip.open(root / Path(str(row.text_path).replace("\\", "/")), "rt", encoding="utf-8") as stream:
            text = stream.read()
        # Short real excerpt around first dictionary hit; full text remains local.
        import re
        hits = [hit for hit in re.finditer(r"[A-Za-z]+", text) if hit.group().upper() in lists["Uncertainty"] | lists["Negative"]]
        # Locate an actual substantive passage after the cover/contents area.
        anchors = ["litigation", "adversely", "loss"] if row["negative_proportion"] > row["uncertainty_proportion"] else ["could", "may", "uncertain"]
        substantive = []
        for anchor in anchors:
            substantive = [h for h in re.finditer(r"\b" + anchor + r"\b", text, re.I) if h.start() > 6000]
            if substantive:
                break
        hit = substantive[0].start() if substantive else hits[0].start() if hits else 0
        excerpts.append(text[max(0, hit-160):hit+600])
    examples["excerpt_for_review"] = excerpts
    diagnostics = {"scoring": scoring_diagnostics, "list_sizes": {k: len(v) for k,v in lists.items()},
                   "list_overlap": len(lists["Negative"] & lists["Uncertainty"]),
                   "starter_list_sizes": {k: len(v) for k,v in old_lists.items()},
                   "day0_moved_in_parsed_candidates": int(events.day0_moved.fillna(False).sum()),
                   "day0_moved_in_final_sample": int(scored.day0_moved.fillna(False).sum()),
                   "final_firms": int(scored.cik.nunique()), "final_filings": len(scored),
                   "table2_n_reconciles": bool(scored.groupby("form").size().sum() == len(final)),
                   "full_period_firms": int((scored.groupby("cik")["filing_date"].apply(lambda d: pd.to_datetime(d).dt.year.nunique()) == 5).sum()),
                   "survivorship_unobservable": "The frozen holdings do not identify holdings sold before the snapshot; their number cannot be inferred.",
                   "input_sha256": input_hashes,
                   "implementation_sha256": {str(path.relative_to(root).as_posix()): sha256(path) for path in sorted((root/"src").glob("*.py"))},
                   "failed_regressions": int(sum((t.status != "estimated").sum() for t in [table4,table5,table6]))}
    artifacts = {"table1": table1, "exclusions": ledger, "events": events, "panel": scored, "table2": table2,
                 "table3": table3, "terms": terms, "quarterly": quarters, "table4": table4,
                 "table5": table5, "table6": table6, "starter_sensitivity_table4": sensitivity4,
                 "starter_sensitivity_table6": sensitivity6, "correlations": correlations,
                 "top10_concentration": concentration.reset_index(), "examples": examples}
    if input_hashes != {name: sha256(root/name) for name in REQUIRED}:
        raise RuntimeError("An acquisition input changed during analysis; rerun after downloads finish")
    if save:
        out = root / "outputs"
        out.mkdir(exist_ok=True)
        for name, frame in artifacts.items():
            frame.to_csv(out / f"{name}.csv", index=False)
        (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, default=str), encoding="utf-8")
        if "^VIX" not in prices or prices["^VIX"].dropna().empty:
            raise ValueError("VIX missing: cannot produce Figure 1")
        figure = figure_quarterly(quarters, prices["^VIX"])
        figure.savefig(out / "figure1.png", dpi=180)
        figure.savefig(out / "figure1.pdf")
    return artifacts, diagnostics
