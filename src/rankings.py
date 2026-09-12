"""Historical holdings screens from uncertainty outside Item 1A."""
from __future__ import annotations

from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd


REVIEWED_BOUNDARIES = {
    '0001562762-21-000079': '2a34d33a1c28a24418631d100c598ed3b4f34c8213fa23225909d801e395fdc1',
    '0001562762-22-000049': '493d9f9a452d767762f685669898c73cbc96508d8267cf3c94f27ff7560e905b',
}


def reviewed_boundaries(sections, root):
    """Reviewed MELI headings preceded by a direction mark in the original HTML.

    These two annual reports use the unique narrative opening below. Exact
    source hashes prevent applying a reviewed boundary to a changed document.
    Both end at the actual Item 1B body heading, whose text says Not applicable.
    """
    from .parse import html_to_text, tokenize
    from .lexicons import load_master_dictionary, lm_word_lists
    root = Path(root)
    out = sections.copy()
    out['boundary_review'] = ''
    words = lm_word_lists(load_master_dictionary(root/'data/lexicons/LoughranMcDonald_MasterDictionary.csv'))['Uncertainty']
    for accession,digest in REVIEWED_BOUNDARIES.items():
        matches = out.index[out.accession.eq(accession)]
        if len(matches) != 1:
            raise ValueError('Reviewed source accession missing or duplicated: '+accession)
        index = matches[0]
        raw = (root/'data/filings'/(accession.replace('-','')+'.html')).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('Reviewed source changed: '+accession)
        text = html_to_text(raw.decode('utf-8'))
        start_label = 'ITEM 1A. RISK FACTORS Set forth below are the risks'
        end_label = 'ITEM 1B. UNRESOLVED STAFF COMMENTS Not applicable.'
        if text.count(start_label) != 1 or text.count(end_label) != 1:
            raise ValueError('Reviewed narrative headings are not unique: '+accession)
        start,end = text.index(start_label),text.index(end_label)
        if end <= start:
            raise ValueError('Invalid reviewed section order')
        removed = tokenize(text[start:end])
        retained = tokenize(text[:start]+' '+text[end:])
        original = int(out.loc[index,'original_n_words'])
        if len(removed)+len(retained) != original:
            raise ValueError('Reviewed token partition does not match original parsing')
        n_unc = sum(w in words for w in retained)
        removed_unc = sum(w in words for w in removed)
        values = {'status':'found','start':start,'end':end,'start_heading':'ITEM 1A. RISK FACTORS',
            'end_heading':'ITEM 1B. UNRESOLVED STAFF COMMENTS','removed_n_words':len(removed),
            'retained_n_words':len(retained),'uncertainty_count':n_unc,
            'removed_uncertainty_count':removed_unc,'original_uncertainty_count':n_unc+removed_unc,
            'uncertainty_pct':100*n_unc/len(retained),
            'original_uncertainty_pct':100*(n_unc+removed_unc)/original,
            'method':'reviewed-source-verified-item1a-boundaries',
            'reason':'Reviewed body headings; leading direction mark prevented automatic detection',
            'boundary_review':'Source-verified MELI annual-report boundaries',
            'removal_excerpt':text[start:start+500]}
        for name,value in values.items():
            out.loc[index,name] = value
    return out


def text_sample(manifest):
    """Apply the assignment's three text filters, before market-data restrictions."""
    frame = manifest.copy()
    if frame.accession.duplicated().any():
        raise ValueError("Duplicate accession in filing manifest")
    frame["filing_date"] = pd.to_datetime(frame.filing_date, errors="raise")
    frame = frame[frame.filing_date.between("2021-01-01", "2025-12-31")]
    frame = frame[frame.form.isin(["10-K", "10-Q"]) & frame.parse_status.eq("ok")]
    minimum = np.where(frame.form.eq("10-K"), 2000, 1000)
    frame = frame[pd.to_numeric(frame.n_words, errors="coerce").ge(minimum)].copy()
    frame["quarter"] = frame.filing_date.dt.to_period("Q").astype(str)
    frame["cik"] = frame.cik.astype(str).str.zfill(10)
    frame = frame.sort_values(["filing_date", "acceptance_datetime", "accession"])
    return frame.drop_duplicates(["cik", "quarter"], keep="first").reset_index(drop=True)


def holding_rankings(scored, universe, end_year=2025):
    """Return all eligible issuers, with unrankable cases retained and explained.

    Levels are equal-filing means in the final filing year, by form. Annual
    slopes regress equal-year 10-K means on filing year. Quarterly slopes use
    individual filing percentages and quarter-of-year intercepts. Trend screens
    require at least three observed years, including the final year; 10-Q also
    requires eight filings and three residual degrees of freedom. Slopes are
    descriptive, not multiple-tested evidence of a predictable return.
    """
    frame = scored.copy()
    frame["filing_date"] = pd.to_datetime(frame.filing_date)
    frame["year"] = frame.filing_date.dt.year
    frame["season"] = frame.filing_date.dt.quarter
    frame = frame[frame.year.between(2021, end_year)]
    frame["cik"] = frame.cik.astype(str).str.zfill(10)
    if frame.accession.duplicated().any():
        raise ValueError("Duplicate scored accession")
    allowed = frame.status.isin(["found", "absent"])
    valid = frame[allowed & frame.uncertainty_pct.notna() & frame.retained_n_words.gt(0)]
    if not valid.uncertainty_pct.between(0, 100).all():
        raise ValueError("Uncertainty percentages outside [0, 100]")
    issuers = universe.copy()
    issuers["cik"] = issuers.cik.astype(str).str.zfill(10)
    issuers = issuers[issuers.cik.isin(frame.cik)].drop_duplicates("cik")
    rows = []
    for issuer in issuers.itertuples():
        for form in ["10-K", "10-Q"]:
            all_group = frame[frame.cik.eq(issuer.cik) & frame.form.eq(form)]
            group = valid[valid.cik.eq(issuer.cik) & valid.form.eq(form)].copy()
            latest = group[group.year.eq(end_year)]
            level_min = 1 if form == "10-K" else 2
            row = {"cik": issuer.cik, "ticker": issuer.ticker,
                   "company": getattr(issuer, "sec_name", issuer.ticker),
                   "funds": getattr(issuer, "funds", ""), "form": form,
                   "n_filings": len(group), "n_unresolved": len(all_group)-len(group),
                   "n_latest": len(latest), "latest_year": end_year,
                   "n_years": group.year.nunique(),
                   "first_year": int(group.year.min()) if len(group) else np.nan,
                   "last_year": int(group.year.max()) if len(group) else np.nan,
                   "uncertainty_pct": np.nan, "slope_pp_year": np.nan,
                   "level_status": "insufficient final-year filings",
                   "trend_status": "fewer than three years or missing final year"}
            if len(latest) >= level_min:
                row["uncertainty_pct"] = latest.uncertainty_pct.mean()
                row["whole_filing_pct"] = latest.original_uncertainty_pct.mean()
                row["removed_words_pct"] = 100 * latest.removed_n_words.sum() / latest.original_n_words.sum()
                row["level_status"] = "ranked"
            if group.year.nunique() >= 3 and len(latest):
                if form == "10-K":
                    annual = group.groupby("year").uncertainty_pct.mean()
                    x = np.column_stack([np.ones(len(annual)), annual.index.to_numpy()-2021])
                    y = annual.to_numpy()
                else:
                    season = pd.get_dummies(group.season.astype(str), drop_first=True, dtype=float)
                    x = np.column_stack([np.ones(len(group)),
                                         group.year.to_numpy()-2021+(group.season.to_numpy()-1)/4,
                                         season.to_numpy()])
                    y = group.uncertainty_pct.to_numpy()
                adequate = form == "10-K" or (len(group) >= 8 and len(y)-x.shape[1] >= 3)
                if adequate and np.linalg.matrix_rank(x) == x.shape[1]:
                    row["slope_pp_year"] = np.linalg.lstsq(x, y, rcond=None)[0][1]
                    row["trend_status"] = "ranked"
                else:
                    row["trend_status"] = "insufficient quarterly history or unidentified slope"
            rows.append(row)
    result = pd.DataFrame(rows)
    for form in ["10-K", "10-Q"]:
        mask = result.form.eq(form)
        level = result.loc[mask, "uncertainty_pct"]
        slope = result.loc[mask, "slope_pp_year"]
        result.loc[mask, "rank_highest"] = level.rank(ascending=False, method="min")
        result.loc[mask, "rank_lowest"] = level.rank(ascending=True, method="min")
        result.loc[mask, "rank_rise"] = slope.where(slope.gt(1e-12)).rank(ascending=False, method="min")
        result.loc[mask, "rank_fall"] = slope.where(slope.lt(-1e-12)).rank(ascending=True, method="min")
    return result.sort_values(["form", "rank_highest", "ticker"], na_position="last").reset_index(drop=True)


SCREENS = [
    ("Highest uncertainty", "rank_highest", "uncertainty_pct", "%"),
    ("Lowest uncertainty", "rank_lowest", "uncertainty_pct", "%"),
    ("Largest annual rise", "rank_rise", "slope_pp_year", "pp/year"),
    ("Largest annual fall", "rank_fall", "slope_pp_year", "pp/year"),
]


def ranking_leaders(rankings, n=3):
    rows = []
    for form in ["10-K", "10-Q"]:
        group = rankings[rankings.form.eq(form)]
        for label, rank, value, unit in SCREENS:
            selected = group[group[rank].notna()].sort_values([rank, "ticker"]).head(n)
            for r in selected.itertuples():
                rows.append({"form": form, "screen": label, "rank": int(getattr(r, rank)),
                             "ticker": r.ticker, "value": getattr(r, value), "unit": unit,
                             "history": f"{int(r.first_year)}–{int(r.last_year)}",
                             "n_filings": r.n_filings, "n_latest": r.n_latest})
    return pd.DataFrame(rows)


def run_rankings(root, *, save=True):
    from .risk_sections import build_risk_panel
    root = Path(root)
    manifest = pd.read_csv(root / "data/interim/filings_manifest.csv", dtype={"cik": str})
    selected = text_sample(manifest)
    sections = reviewed_boundaries(build_risk_panel(root),root)
    # Metadata authority remains the downloaded manifest.
    fields = [c for c in sections.columns if c not in selected.columns or c == "accession"]
    panel = selected.merge(sections[fields], on="accession", how="left", validate="one_to_one")
    if panel.status.isna().any():
        raise ValueError("Missing Item 1A extraction audit")
    universe = pd.read_csv(root / "data/universe/universe.csv", dtype={"cik": str})
    ranks = holding_rankings(panel, universe)
    leaders = ranking_leaders(ranks)
    summary = {"text_filings": len(panel), "text_firms": panel.cik.nunique(),
               "reviewed_boundaries": int(panel.boundary_review.ne('').sum()),
               "section_status": panel.groupby(["form", "status"]).size().to_dict(),
               "level_firms": ranks[ranks.level_status.eq("ranked")].groupby("form").size().to_dict(),
               "trend_firms": ranks[ranks.trend_status.eq("ranked")].groupby("form").size().to_dict()}
    summary["input_sha256"] = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
        for name in ["data/interim/filings_manifest.csv", "data/universe/universe.csv",
                     "data/lexicons/LoughranMcDonald_MasterDictionary.csv", "src/rankings.py",
                     "src/risk_sections.py", "src/parse.py"]}
    summary["section_status"] = {" / ".join(k): int(v) for k, v in summary["section_status"].items()}
    if save:
        out = root / "outputs"
        out.mkdir(exist_ok=True)
        panel.to_csv(out / "uncertainty_ex_item1a_filings.csv", index=False)
        ranks.to_csv(out / "uncertainty_holdings_rankings.csv", index=False)
        leaders.to_csv(out / "uncertainty_ranking_leaders.csv", index=False)
        (out / "uncertainty_rankings_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return panel, ranks, leaders, summary
