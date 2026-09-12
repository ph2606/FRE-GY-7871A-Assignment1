"""Proportional tone and the assignment's exact Loughran--McDonald weights.

Fit document frequencies on the metadata rows supplied to ``score_corpus``.
The caller must finish sample selection first; this module neither drops rows
nor fits inverse document frequencies on excluded or external documents.
"""

from __future__ import annotations

import gzip
import hashlib
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from numbers import Integral
from pathlib import Path

import pandas as pd

from .parse import tokenize


def _lists(word_lists: Mapping[str, Iterable[str]]) -> dict[str, set[str]]:
    if not word_lists:
        raise ValueError("At least one word list is required.")
    result = {
        str(category): {str(word).strip().upper() for word in words}
        for category, words in word_lists.items()
    }
    slugs = [_slug(category) for category in result]
    if len(set(slugs)) != len(slugs) or any(not slug for slug in slugs):
        raise ValueError("Category names must yield distinct, nonempty column names.")
    if any("" in words for words in result.values()):
        raise ValueError("A word list contains an empty word.")
    return result


def _slug(category: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")


def _compact_document(
    counts: Mapping[str, int], vocabulary: set[str]
) -> tuple[dict[str, int | float], Counter]:
    """Retain target counts and length statistics, never a corpus of token lists."""
    normalized = Counter()
    for word, count in counts.items():
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValueError(f"Word counts must be nonnegative integers: {word!r}.")
        if count:
            normalized[str(word).upper()] += int(count)
    total = sum(normalized.values())
    distinct = len(normalized)
    if not total:
        raise ValueError("An empty document cannot be scored; apply parse filters first.")
    summary = {"n_words": total, "n_distinct": distinct, "a_j": total / distinct}
    return summary, Counter({word: count for word, count in normalized.items() if word in vocabulary})


def _score_compact(
    documents: list[tuple[dict[str, int | float], Counter]],
    word_lists: dict[str, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not documents:
        raise ValueError("The scoring corpus is empty.")
    n_documents = len(documents)
    document_frequency = Counter()
    corpus_counts = Counter()
    for _, counts in documents:
        document_frequency.update(counts.keys())
        corpus_counts.update(counts)
    # Absent words never enter this map: log(N / 0) is not evaluated.
    idf = {word: math.log(n_documents / df) for word, df in document_frequency.items()}
    rows = []
    for summary, counts in documents:
        row = dict(summary)
        normalizer = 1.0 + math.log(summary["a_j"])
        weighted = {
            word: (1.0 + math.log(count)) * idf[word] / normalizer
            for word, count in counts.items()
        }
        for category, words in word_lists.items():
            slug = _slug(category)
            category_count = sum(counts.get(word, 0) for word in words)
            row[f"{slug}_count"] = category_count
            row[f"{slug}_proportion"] = category_count / summary["n_words"]
            row[f"{slug}_tfidf"] = math.fsum(weighted.get(word, 0.0) for word in sorted(words))
        rows.append(row)

    term_rows = []
    category_totals = {}
    for category, words in word_lists.items():
        total = sum(corpus_counts.get(word, 0) for word in words)
        category_totals[category] = total
        for word in sorted(words):
            count = corpus_counts.get(word, 0)
            term_rows.append({
                "category": category,
                "word": word,
                "count": count,
                "share": count / total if total else float("nan"),
                "document_frequency": document_frequency.get(word, 0),
                "idf": idf.get(word, float("nan")),
            })
    terms = pd.DataFrame(term_rows, columns=[
        "category", "word", "count", "share", "document_frequency", "idf"
    ]).sort_values(["category", "count", "word"], ascending=[True, False, True], ignore_index=True)
    diagnostics = {
        "n_documents": n_documents,
        "total_words": sum(int(summary["n_words"]) for summary, _ in documents),
        "word_list_sizes": {category: len(words) for category, words in word_lists.items()},
        "category_occurrences": category_totals,
        "idf_corpus": "Exactly the documents passed to this scoring call, pooled across form types.",
        "idf_log_base": "natural",
        "a_j": "all tokenizer tokens / distinct tokenizer tokens within each filing",
        "weighted_aggregation": "sum of category word weights, without further normalization",
        "denominator": "all alphabetic tokenizer tokens of at least two characters",
    }
    return pd.DataFrame(rows), terms, diagnostics


def score_counts(
    document_counts: Iterable[Mapping[str, int]],
    word_lists: Mapping[str, Iterable[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Score full-document Counters; convenient for the README's three-document check.

    Counter keys are case-normalized. All positive counts, including words outside
    every sentiment list, enter the total and distinct-word denominators. Returned
    proportions are fractions, not percentages. Term shares also are fractions.
    """
    lists = _lists(word_lists)
    vocabulary = set().union(*lists.values())
    documents = [_compact_document(counts, vocabulary) for counts in document_counts]
    return _score_compact(documents, lists)


def score_corpus(
    meta: pd.DataFrame,
    word_lists: Mapping[str, Iterable[str]],
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Score exactly ``meta`` and retain its rows, index, accessions and metadata.

    Required metadata columns are ``accession`` and ``text_path``. Relative paths
    resolve against ``root``; .gz paths contain UTF-8 gzip text. Cached ``n_words``
    and ``n_distinct``, when present, must agree with the current tokenizer. A stale
    cache or unreadable document raises an error instead of changing the sample.

    Columns added for each category use its lowercase name, e.g.
    ``negative_count``, ``negative_proportion``, ``negative_tfidf``. Also returns
    category/word occurrence shares and document frequencies, plus fit diagnostics.
    """
    required = {"accession", "text_path"}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")
    if meta.empty:
        raise ValueError("The scoring corpus is empty.")
    if meta["accession"].isna().any() or meta["accession"].astype(str).duplicated().any():
        raise ValueError("Scoring requires one nonmissing, unique accession per document.")
    lists = _lists(word_lists)
    vocabulary = set().union(*lists.values())
    documents = []
    root = Path(root)
    for row in meta.to_dict("records"):
        accession = str(row["accession"])
        if pd.isna(row["text_path"]):
            raise ValueError(f"Missing text_path for accession {accession}.")
        path = Path(row["text_path"])
        if not path.is_absolute():
            path = root / path
        opener = gzip.open if path.suffix.lower() == ".gz" else open
        counts = Counter()
        try:
            with opener(path, "rt", encoding="utf-8") as document:
                for line in document:
                    counts.update(tokenize(line))
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"Cannot read text for accession {accession}: {path}") from exc
        summary, target_counts = _compact_document(counts, vocabulary)
        for field in ("n_words", "n_distinct"):
            if field in row:
                try:
                    cached = float(row[field])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid cached {field} for accession {accession}.") from exc
                if not math.isfinite(cached) or cached != summary[field]:
                    raise ValueError(
                        f"Stale cached {field} for accession {accession}: "
                        f"metadata={row[field]}, tokenized={summary[field]}. Reparse before scoring."
                    )
        documents.append((summary, target_counts))
    scores, terms, diagnostics = _score_compact(documents, lists)
    scored = meta.copy()
    for column in scores:
        scored[column] = scores[column].to_numpy()
    fingerprint = "\n".join(sorted(meta["accession"].astype(str)))
    diagnostics["corpus_accession_sha256"] = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
    diagnostics["cached_counts_checked"] = [field for field in ("n_words", "n_distinct") if field in meta]
    return scored, terms, diagnostics
