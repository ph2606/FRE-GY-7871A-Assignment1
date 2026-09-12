"""Loading the word lists.

Everything you need comes from one file, the Loughran-McDonald Master Dictionary:

    Negative      Fin-Neg, size depends on dictionary version and removal flags. How bad the news is.
    Uncertainty   Fin-Unc,   297 words. How sure management is.
    Positive, Litigious, Strong_Modal, Weak_Modal  -- available, not required.

The rest of this module rebuilds the Harvard General Inquirer negative list. That
is OPTIONAL, for the extra-credit comparison only. The Harvard file ships root
words, so LM expanded 2,005 roots into 4,187 inflected forms by hand; the rules
here approximate that and land nearer 5,000, which is a real deviation you would
have to disclose if you use it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import HARVARD_GI_PATH, LM_CATEGORIES, LM_MASTER_DICT_PATH


def load_master_dictionary(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or LM_MASTER_DICT_PATH)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run: python scripts/00_get_lexicons.py")
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, keep_default_na=False)
    required = {"Word", *LM_CATEGORIES}
    if not required.issubset(df.columns):
        raise ValueError(f"Dictionary missing columns: {sorted(required - set(df.columns))}")
    if df["Word"].eq("").any() or df["Word"].isna().any() or df["Word"].str.upper().duplicated().any():
        raise ValueError("Dictionary must have unique, nonmissing words")
    df["Word"] = df["Word"].astype(str).str.upper()
    return df


def lm_word_lists(master: pd.DataFrame | None = None, *, include_removed: bool = False) -> dict[str, set[str]]:
    """{'Negative': {...}, 'Positive': {...}, ...} from the master dictionary.

    Positive years mark membership; negative years mark removal, per the authors'
    current documentation. include_removed=True reproduces the starter's nonzero
    rule for a disclosed sensitivity analysis, not the active dictionary.
    """
    master = master if master is not None else load_master_dictionary()
    out = {}
    for cat in LM_CATEGORIES:
        flags = pd.to_numeric(master[cat], errors="raise").fillna(0)
        selected = flags.ne(0) if include_removed else flags.gt(0)
        out[cat] = set(master.loc[selected, "Word"])
    return out


def lm_vocabulary(master: pd.DataFrame | None = None) -> set[str]:
    """Every word the LM team observed in 10-X filings. Used to filter inflections."""
    master = master if master is not None else load_master_dictionary()
    return set(master["Word"])


def harvard_negative_roots(path: Path | None = None) -> set[str]:
    """The Harvard-IV-4 'Negativ' tag, one entry per root word.

    inqtabs.txt is sense-disambiguated: TAX#1 is neutral, TAX#2 is Negativ. Any
    sense tagged Negativ puts the root on the list, which is exactly the
    behaviour that makes the Harvard list misfire on financial text.
    """
    path = Path(path or HARVARD_GI_PATH)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run: python scripts/00_get_lexicons.py")
    lines = path.read_text(encoding="latin-1").splitlines()
    header = lines[0].split("\t")
    idx = header.index("Negativ")
    roots = set()
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) <= idx or not parts[idx].strip():
            continue
        word = parts[0].split("#")[0].strip().upper()
        if re.fullmatch(r"[A-Z'\-]+", word):
            roots.add(word)
    return roots


def inflections(word: str) -> set[str]:
    """Plural, third person, past and progressive forms of one root word."""
    forms = {word}
    if word.endswith("Y") and len(word) > 2 and word[-2] not in "AEIOU":
        forms |= {word[:-1] + "IES", word[:-1] + "IED"}
    elif word.endswith("E"):
        forms |= {word + "S", word + "D", word[:-1] + "ING"}
    elif word.endswith(("S", "X", "Z", "CH", "SH")):
        forms |= {word + "ES", word + "ED", word + "ING"}
    else:
        forms |= {word + "S", word + "ED", word + "ING"}
        if (len(word) > 3 and word[-1] not in "AEIOUWXY"
                and word[-2] in "AEIOU" and word[-3] not in "AEIOU"):
            forms |= {word + word[-1] + "ED", word + word[-1] + "ING"}
    return forms


def h4n_inflected(master: pd.DataFrame | None = None) -> set[str]:
    """The H4N-Inf list: Harvard Negativ roots plus inflections seen in filings."""
    vocab = lm_vocabulary(master)
    out = set()
    for root in harvard_negative_roots():
        out |= {f for f in inflections(root) if f in vocab}
    return out


def load_all(include_harvard: bool = False) -> dict[str, set[str]]:
    """The word lists the assignment needs, in one dictionary.

    All six Loughran-McDonald categories. The two you are graded on are
    ``Negative`` (Fin-Neg) and ``Uncertainty`` (Fin-Unc).

    ``include_harvard=True`` adds the rebuilt Harvard list under "H4N_Inf".
    That comparison is extra credit only, and it needs inqtabs.txt, which
    ``scripts/00_get_lexicons.py --with-harvard`` downloads.
    """
    master = load_master_dictionary()
    lists = lm_word_lists(master)
    if include_harvard:
        lists["H4N_Inf"] = h4n_inflected(master)
    return lists
