"""Exclusive sample-loss accounting and the final-corpus scoring interface."""

import gzip

import pandas as pd
import pytest

from src.analysis import add_time
from src.pipeline import sample_waterfall
from src.scoring import score_corpus


FLAGS = ["pass_day0_price", "pass_return_history", "pass_complete_windows", "pass_shares", "pass_controls"]


def _manifest_row(accession, cik, *, date="2021-02-01", accepted="2021-02-01T15:00:00Z",
                  form="10-K", words=2000, status="ok"):
    return {"accession": accession, "cik": str(cik), "ticker": f"T{cik}",
            "filing_date": date, "acceptance_datetime": accepted, "form": form,
            "n_words": words, "n_distinct": 2, "parse_status": status,
            "text_path": f"{accession}.txt.gz"}


def _events(manifest):
    rows = []
    for accession in manifest["accession"]:
        rows.append({"accession": accession, **{flag: True for flag in FLAGS},
                     "event_return": 0.01, "pre_volatility": 0.2, "post_volatility": 0.3})
    return pd.DataFrame(rows)


def test_waterfall_losses_are_exclusive_ordered_and_do_not_replace_earliest():
    rows = [
        _manifest_row("amendment", 1, form="10-K/A"),
        _manifest_row("parse_failure", 2, status="failed", words=0),
        _manifest_row("short_q", 3, form="10-Q", words=999),
        _manifest_row("earliest_bad_price", 4),
        _manifest_row("later_good_price", 4, date="2021-02-10", accepted="2021-02-10T15:00:00Z"),
        _manifest_row("short_history", 5),
        _manifest_row("incomplete_window", 6),
        _manifest_row("missing_shares", 7),
        _manifest_row("missing_control", 8),
        _manifest_row("retained_k", 9),
        _manifest_row("retained_q", 9, date="2021-05-01", accepted="2021-05-01T15:00:00Z", form="10-Q", words=1000),
    ]
    manifest = pd.DataFrame(rows).sample(frac=1, random_state=7)
    events = _events(manifest)
    for accession, flag in zip(
        ["earliest_bad_price", "short_history", "incomplete_window", "missing_shares", "missing_control"], FLAGS
    ):
        events.loc[events["accession"] == accession, flag] = False
    final, waterfall, ledger = sample_waterfall(manifest, events)
    assert waterfall["removed"].tolist() == [0, 2, 1, 1, 1, 1, 1, 1, 1]
    assert waterfall["remaining"].tolist() == [11, 9, 8, 7, 6, 5, 4, 3, 2]
    assert waterfall["removed"].sum() + len(final) == len(manifest)
    assert set(final["accession"]) == {"retained_k", "retained_q"}
    assert final["cik"].tolist() == ["0000000009", "0000000009"]
    indexed = ledger.set_index("accession")
    assert indexed.loc["later_good_price", "exclusion_stage"].startswith("3 ")
    assert indexed.loc["earliest_bad_price", "exclusion_stage"].startswith("4 ")
    assert ledger["exclusion_stage"].eq("retained").sum() == len(final)
    assert set(ledger["accession"]) == set(manifest["accession"])


def test_same_day_ties_use_acceptance_then_accession_deterministically():
    manifest = pd.DataFrame([
        _manifest_row("b", 11, accepted="2021-02-01T15:00:00Z"),
        _manifest_row("a", 11, accepted="2021-02-01T15:00:00Z"),
        _manifest_row("earlier_name_later_acceptance", 11, accepted="2021-02-01T16:00:00Z"),
    ])
    final, waterfall, _ = sample_waterfall(manifest, _events(manifest))
    assert final["accession"].tolist() == ["a"]
    assert waterfall.loc[3, "removed"] == 2


def test_metadata_counts_remain_authoritative_and_scoring_preserves_final_corpus(tmp_path):
    manifest = pd.DataFrame([
        _manifest_row("k", 20),
        _manifest_row("q", 20, date="2021-05-01", accepted="2021-05-01T15:00:00Z", form="10-Q", words=1000),
    ])
    events = _events(manifest)
    events["n_words"] = -999  # event join must not replace parser metadata
    for accession, text in [("k", "LOSS GAIN " * 1000), ("q", "RISK GAIN " * 500)]:
        with gzip.open(tmp_path / f"{accession}.txt.gz", "wt", encoding="utf-8") as handle:
            handle.write(text)
    final, _, _ = sample_waterfall(manifest, events)
    scored, _, diagnostics = score_corpus(final, {"Negative": {"LOSS"}, "Uncertainty": {"RISK"}}, tmp_path)
    timed = add_time(scored)
    assert timed["accession"].tolist() == final["accession"].tolist() == ["k", "q"]
    assert timed["n_words"].tolist() == [2000, 1000]
    assert timed["quarter"].tolist() == ["2021Q1", "2021Q2"]
    assert timed["negative_proportion_pp"].tolist() == [50, 0]
    assert diagnostics["n_documents"] == len(final) == len(timed)


@pytest.mark.parametrize("duplicate_in", ["manifest", "events"])
def test_duplicate_accessions_cannot_multiply_the_final_corpus(duplicate_in):
    manifest = pd.DataFrame([_manifest_row("a", 30)])
    events = _events(manifest)
    if duplicate_in == "manifest":
        manifest = pd.concat([manifest, manifest], ignore_index=True)
    else:
        events = pd.concat([events, events], ignore_index=True)
    with pytest.raises(ValueError, match="[Dd]uplicate accession"):
        sample_waterfall(manifest, events)


def test_outside_sample_dates_are_rejected_instead_of_counted_in_the_corpus():
    manifest = pd.DataFrame([_manifest_row("a", 30, date="2020-12-31")])
    with pytest.raises(ValueError, match="Out-of-window"):
        sample_waterfall(manifest, _events(manifest))
