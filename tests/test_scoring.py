"""Numerical source checks and corpus/cache invariants for tone measurement."""

import gzip
import math
from collections import Counter

import pandas as pd
import pytest

from src.scoring import score_corpus, score_counts


def test_readme_equation_one_self_check():
    documents = [Counter(text.split()) for text in (
        "LOSS LOSS RISK GAIN", "LOSS GAIN GAIN", "RISK RISK RISK GAIN"
    )]
    scored, terms, diagnostics = score_counts(documents, {"Negative": {"LOSS", "RISK"}})
    assert scored["a_j"].tolist() == pytest.approx([4 / 3, 1.5, 2.0])
    assert scored["negative_proportion"].tolist() == pytest.approx([0.75, 1 / 3, 0.75])
    assert [f"{value:.4f}" for value in scored["negative_tfidf"]] == ["0.8480", "0.2885", "0.5026"]
    assert terms.set_index("word")["document_frequency"].to_dict() == {"RISK": 2, "LOSS": 2}
    assert diagnostics["n_documents"] == 3


def test_all_words_in_denominators_and_document_presence_in_idf():
    scored, terms, _ = score_counts(
        [{"LOSS": 10, "OTHER": 2}, {"OTHER": 2}], {"Negative": {"LOSS", "OTHER", "ABSENT"}}
    )
    # OTHER is in every document, so its IDF is exactly zero despite its count.
    assert scored.loc[0, "negative_tfidf"] == pytest.approx((1 + math.log(10)) / (1 + math.log(6)) * math.log(2))
    assert scored.loc[1, "negative_tfidf"] == 0
    indexed = terms.set_index("word")
    assert indexed.loc["OTHER", "idf"] == 0
    assert indexed.loc["ABSENT", "document_frequency"] == 0
    assert math.isnan(indexed.loc["ABSENT", "idf"])
    assert indexed["share"].sum() == pytest.approx(1)
    # Outside-list words must still count in the length and distinct-word counts.
    narrow, _, _ = score_counts([{"LOSS": 10, "OTHER": 2}, {"OTHER": 2}], {"Negative": {"LOSS"}})
    assert narrow.loc[0, "negative_proportion"] == pytest.approx(10 / 12)
    assert narrow.loc[0, "a_j"] == 6
    assert narrow.loc[0, "negative_tfidf"] == scored.loc[0, "negative_tfidf"]


def test_overlapping_categories_are_counted_independently():
    scored, terms, _ = score_counts(
        [{"RISK": 2, "GAIN": 1}, {"GAIN": 1}],
        {"Negative": {"RISK"}, "Uncertainty": {"RISK", "MAY"}},
    )
    assert scored.loc[0, "negative_count"] == scored.loc[0, "uncertainty_count"] == 2
    assert scored.loc[0, "negative_tfidf"] == scored.loc[0, "uncertainty_tfidf"]
    assert terms.query("word == 'RISK'")["count"].tolist() == [2, 2]


def _metadata(tmp_path):
    for name, text in [("a", "LOSS-RELATED loss.\nGAIN"), ("b", "GAIN GAIN")]:
        with gzip.open(tmp_path / f"{name}.txt.gz", "wt", encoding="utf-8") as handle:
            handle.write(text)
    return pd.DataFrame({
        "accession": ["0001-a", "0002-b"],
        "text_path": ["a.txt.gz", "b.txt.gz"],
        "n_words": [4, 2], "n_distinct": [3, 1], "form": ["10-K", "10-Q"],
    }, index=[15, 8])


def test_gzip_corpus_preserves_metadata_and_fits_only_passed_rows(tmp_path):
    meta = _metadata(tmp_path)
    scored, terms, diagnostics = score_corpus(meta, {"Negative": {"LOSS"}}, tmp_path)
    assert scored.index.tolist() == [15, 8]
    pd.testing.assert_frame_equal(scored[meta.columns], meta)
    assert scored["negative_proportion"].tolist() == [0.5, 0.0]
    assert terms.loc[0, "document_frequency"] == 1
    assert diagnostics["n_documents"] == 2
    subset, _, subset_diagnostics = score_corpus(meta.iloc[:1], {"Negative": {"LOSS"}}, tmp_path)
    assert subset.loc[15, "negative_tfidf"] == 0
    assert diagnostics["corpus_accession_sha256"] != subset_diagnostics["corpus_accession_sha256"]


def test_stale_count_is_an_error_not_a_silent_sample_change(tmp_path):
    meta = _metadata(tmp_path)
    meta.loc[15, "n_words"] = 3
    with pytest.raises(ValueError, match="Stale cached n_words.*0001-a"):
        score_corpus(meta, {"Negative": {"LOSS"}}, tmp_path)


def test_duplicate_accessions_are_rejected(tmp_path):
    meta = _metadata(tmp_path)
    meta.loc[8, "accession"] = "0001-a"
    with pytest.raises(ValueError, match="unique accession"):
        score_corpus(meta, {"Negative": {"LOSS"}}, tmp_path)


@pytest.mark.parametrize("counts", [[], [{}], [{"LOSS": -1}], [{"LOSS": 1.5}]])
def test_empty_corpus_empty_documents_and_invalid_counts_are_rejected(counts):
    with pytest.raises(ValueError):
        score_counts(counts, {"Negative": {"LOSS"}})
