"""Central configuration for the data pipeline.

This file defines the sample: which funds, which window, which forms. The
analysis parameters (event windows, filter thresholds) are specified in the
assignment brief and are yours to set.
"""

import os
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LEXICON_DIR = DATA / "lexicons"
FILING_DIR = DATA / "filings"
PRICE_DIR = DATA / "prices"
UNIVERSE_DIR = DATA / "universe"
INTERIM_DIR = DATA / "interim"
OUTPUT_DIR = ROOT / "outputs"

for _d in (LEXICON_DIR, FILING_DIR, PRICE_DIR, UNIVERSE_DIR, INTERIM_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# SEC EDGAR
# ----------------------------------------------------------------------------
# The SEC requires a declared User-Agent with a real contact address, and rate
# limits to 10 requests/second. Set SEC_USER_AGENT in your environment:
#     Windows PowerShell:  $env:SEC_USER_AGENT = "Your Name your.netid@nyu.edu"
#     macOS / Linux:       export SEC_USER_AGENT="Your Name your.netid@nyu.edu"
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "")
SEC_MAX_REQUESTS_PER_SEC = 6.0  # below the SEC's limit of 10, on purpose

# ----------------------------------------------------------------------------
# Sample definition  (Section 3 of the assignment)
# ----------------------------------------------------------------------------
ARK_FUNDS = ["ARKK", "ARKQ", "ARKW", "ARKF", "ARKG", "ARKX"]

SAMPLE_START = "2021-01-01"          # filing date, inclusive
SAMPLE_END = "2025-12-31"            # filing date, inclusive
FORMS = ["10-K", "10-Q"]             # amendments (10-K/A, 10-Q/A) are excluded

# Series downloaded alongside the filers, for you to use as benchmarks.
BENCHMARK = "SPY"                    # stands in for the CRSP value-weighted index
ALT_BENCHMARK = "ARKK"               # thematic-peer benchmark
VIX_TICKER = "^VIX"                  # market-implied uncertainty

# ----------------------------------------------------------------------------
# Word-list downloads (verified working 2026-09-05; if a link rots, see README)
# ----------------------------------------------------------------------------
LM_MASTER_DICT_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF&export=download&confirm=t"
)
LM_MASTER_DICT_PATH = LEXICON_DIR / "LoughranMcDonald_MasterDictionary.csv"

# Optional. The Harvard General Inquirer negative list is NOT required for this
# assignment; it is here only for the extra-credit comparison in the brief.
HARVARD_GI_URL = "https://inquirer.sites.fas.harvard.edu/inqtabs.txt"
HARVARD_GI_PATH = LEXICON_DIR / "inqtabs.txt"

# Economic Policy Uncertainty index (Baker, Bloom & Davis), free, monthly.
# An optional external series to plot against your own uncertainty measure.
EPU_MONTHLY_URL = "https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.xlsx"

LM_CATEGORIES = [
    "Negative", "Positive", "Uncertainty",
    "Litigious", "Strong_Modal", "Weak_Modal",
]
