"""Filing HTML -> a bag of words.

It handles two things that can silently ruin a
dictionary study if you get them wrong:

1. Inline XBRL. Since ~2019 every 10-K and 10-Q is an inline-XBRL document.
   If you strip tags naively you get thousands of tokens like
   "us-gaap MoneyMarketFundsMember" mixed into the text, which inflates the
   denominator of every proportional measure. We drop hidden XBRL scaffolding,
   but unwrap visible inline facts so their narrative remains in the corpus.

2. Tables. Loughran and McDonald exclude tables and exhibits, because tables are
   mostly numbers and boilerplate. But in a modern filing, tables are also used
   for page layout, so dropping every <table> throws away real narrative. We use
   the standard compromise: drop a table only if more than 15% of its non-space
   characters are digits.

Both choices are defensible and both are choices. If you change the thresholds,
say so in your report and show what it does to your results.
"""

from __future__ import annotations

import html
import re
import warnings
from collections import Counter

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

NUMERIC_TABLE_THRESHOLD = 0.15
_XBRL_PREFIXES = ("xbrli:", "xbrldi:", "link:", "xsi:", "xbrl:")
_HIDDEN_IX = {"ix:header", "ix:hidden", "ix:references", "ix:resources"}
_TOKEN_RE = re.compile(r"[A-Za-z]+")
PARSER_VERSION = "visible-inline-facts-alphabetic-v2"


def _numeric_share(text: str) -> float:
    text = re.sub(r"\s+", "", text)
    if not text:
        return 1.0
    return sum(c.isdigit() for c in text) / len(text)


def html_to_text(raw_html: str, drop_numeric_tables: bool = True) -> str:
    """Extract readable narrative text from a filing's primary document."""
    soup = BeautifulSoup(raw_html, "lxml")

    for tag in soup.find_all(["script", "style"]):
        tag.decompose()

    for tag in list(soup.find_all(True)):
        if tag.decomposed:
            continue
        name = (tag.name or "").lower()
        if name.startswith(_XBRL_PREFIXES) or name in _HIDDEN_IX or name == "xbrl":
            tag.decompose()
            continue
        style = (tag.attrs.get("style") or "").replace(" ", "").lower()
        if "display:none" in style or "visibility:hidden" in style or tag.has_attr("hidden"):
            tag.decompose()
            continue
        if name.startswith("ix:"):
            # nonNumeric facts can contain whole narrative sections. Numeric
            # facts also stay visible until numeric tables have been assessed.
            tag.unwrap()

    if drop_numeric_tables:
        for table in list(soup.find_all("table")):
            if table.decomposed:
                continue
            if _numeric_share(table.get_text(" ")) > NUMERIC_TABLE_THRESHOLD:
                table.decompose()

    text = html.unescape(soup.get_text(" ")).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str, min_length: int = 2) -> list[str]:
    """Uppercase alphabetic tokens. Numbers are dropped; the word lists have none.

    Punctuation separates words (RISK-RELATED -> RISK, RELATED); possessive 's
    is dropped by the two-character minimum. No stemming or dictionary-based
    filtering of the denominator is performed.
    """
    return [w.upper() for w in _TOKEN_RE.findall(text) if len(w) >= min_length]


def word_counts(tokens: list[str]) -> Counter:
    return Counter(tokens)


def parse_filing(raw_html: str) -> dict:
    """One filing -> {'n_words', 'counts'} ready for scoring."""
    tokens = tokenize(html_to_text(raw_html))
    return {"n_words": len(tokens), "counts": word_counts(tokens)}
