import numpy as np
import pandas as pd
import pytest

from src.rankings import holding_rankings, ranking_leaders, text_sample


def inputs():
    rows = []
    for ticker, cik, base, slope in [("AAA", "0000000001", 1.0, .2), ("BBB", "0000000002", 3.0, -.1)]:
        for year in range(2021, 2026):
            for form, months in [("10-K", [2]), ("10-Q", [5, 8, 11])]:
                for month in months:
                    value = base + slope*(year-2021) + (0 if form == "10-K" else month*.1)
                    rows.append(dict(ticker=ticker, cik=cik, form=form,
                                     accession=f"{ticker}-{year}-{month}", filing_date=f"{year}-{month:02}-15",
                                     uncertainty_pct=value, original_uncertainty_pct=value+.5,
                                     status="found", retained_n_words=9000, original_n_words=10000,
                                     removed_n_words=1000))
    universe = pd.DataFrame([dict(ticker=t, cik=c, sec_name=t, funds="ARKK")
                             for t,c in [("AAA","0000000001"),("BBB","0000000002")]])
    return pd.DataFrame(rows), universe


def test_levels_separate_forms_and_slopes_are_percentage_points():
    frame, universe = inputs()
    ranks = holding_rankings(frame, universe).set_index(["ticker", "form"])
    assert ranks.loc[("AAA","10-K"),"uncertainty_pct"] == pytest.approx(1.8)
    assert ranks.loc[("AAA","10-Q"),"uncertainty_pct"] == pytest.approx(2.6)
    assert ranks.loc[("AAA","10-Q"),"slope_pp_year"] == pytest.approx(.2)
    assert ranks.loc[("BBB","10-K"),"slope_pp_year"] == pytest.approx(-.1)
    assert ranks.loc[("BBB","10-K"),"rank_highest"] == 1
    assert ranks.loc[("AAA","10-K"),"rank_lowest"] == 1
    assert ranks.loc[("AAA","10-K"),"rank_rise"] == 1
    assert ranks.loc[("BBB","10-K"),"rank_fall"] == 1
    assert np.isnan(ranks.loc[("AAA","10-K"),"rank_fall"])


def test_ambiguous_extraction_is_excluded_and_reported():
    frame, universe = inputs()
    mask = frame.ticker.eq("AAA") & frame.filing_date.str.startswith("2025")
    frame.loc[mask, "status"] = "ambiguous"
    ranks = holding_rankings(frame, universe).set_index(["ticker", "form"])
    for form, n in [("10-K",1),("10-Q",3)]:
        row = ranks.loc[("AAA",form)]
        assert row.n_unresolved == n
        assert np.isnan(row.uncertainty_pct)
        assert np.isnan(row.slope_pp_year)
        assert row.level_status != "ranked"


def test_short_history_gets_level_but_no_trend_and_zero_slope_no_direction():
    frame, universe = inputs()
    frame = frame[~frame.ticker.eq("AAA") | frame.filing_date.str[:4].ge("2024")].copy()
    frame.loc[frame.ticker.eq("BBB"), "uncertainty_pct"] = 2
    ranks = holding_rankings(frame, universe).set_index(["ticker","form"])
    assert ranks.loc[("AAA","10-K"), "level_status"] == "ranked"
    assert np.isnan(ranks.loc[("AAA","10-K"), "slope_pp_year"])
    assert np.isnan(ranks.loc[("BBB","10-K"), "rank_rise"])
    assert np.isnan(ranks.loc[("BBB","10-K"), "rank_fall"])


def test_missing_season_composition_does_not_create_quarterly_trend():
    frame, universe = inputs()
    frame = frame[~(frame.form.eq("10-Q") & frame.filing_date.str.startswith("2021-11"))]
    ranks = holding_rankings(frame, universe)
    aaa = ranks[ranks.ticker.eq("AAA") & ranks.form.eq("10-Q")].iloc[0]
    assert aaa.slope_pp_year == pytest.approx(.2)


def test_text_filters_do_not_require_market_data():
    frame, _ = inputs()
    frame["parse_status"] = "ok"
    frame["n_words"] = 5000
    frame["acceptance_datetime"] = frame.filing_date
    duplicate = frame.iloc[[0]].copy()
    duplicate["accession"] = "duplicate-quarter"
    duplicate["filing_date"] = "2021-03-01"
    amendment = frame.iloc[[1]].copy()
    amendment["accession"] = "amended"
    amendment["form"] = "10-Q/A"
    result = text_sample(pd.concat([frame,duplicate,amendment], ignore_index=True))
    assert len(result) == len(frame)
    assert "duplicate-quarter" not in set(result.accession)
    assert "amended" not in set(result.accession)


def test_leaders_never_label_a_decline_as_a_rise():
    frame, universe = inputs()
    leaders = ranking_leaders(holding_rankings(frame, universe))
    assert (leaders.loc[leaders.screen.eq("Largest annual rise"), "value"] > 0).all()
    assert (leaders.loc[leaders.screen.eq("Largest annual fall"), "value"] < 0).all()


def test_reviewed_boundaries_use_body_and_reject_changed_source(tmp_path,monkeypatch):
    import hashlib
    import src.rankings as module
    from src.parse import html_to_text,tokenize
    from src.config import LM_CATEGORIES
    accession='0000000001-21-000001'
    raw='''<p>ITEM 1A. RISK FACTORS 3</p><p>ITEM 1B. UNRESOLVED STAFF COMMENTS 4</p>
    <p>Business may grow.</p><p>\u200e ITEM 1A. RISK FACTORS Set forth below are the risks</p>
    <p>Demand could fall.</p><p>ITEM 1B. UNRESOLVED STAFF COMMENTS Not applicable.</p>'''
    path=tmp_path/'data/filings'/(accession.replace('-','')+'.html')
    path.parent.mkdir(parents=True);path.write_text(raw,encoding='utf-8')
    lex=tmp_path/'data/lexicons/LoughranMcDonald_MasterDictionary.csv'
    lex.parent.mkdir(parents=True)
    pd.DataFrame({'Word':['MAY','COULD','RISK'],**{c:([2011]*3 if c=='Uncertainty' else [0]*3)
                  for c in LM_CATEGORIES}}).to_csv(lex,index=False)
    monkeypatch.setattr(module,'REVIEWED_BOUNDARIES',{accession:hashlib.sha256(path.read_bytes()).hexdigest()})
    frame=pd.DataFrame([{'accession':accession,'original_n_words':len(tokenize(html_to_text(raw)))}])
    result=module.reviewed_boundaries(frame,tmp_path).iloc[0]
    assert result.status=='found'
    assert result.uncertainty_count==2  # Contents RISK and business MAY remain.
    assert result.removed_uncertainty_count==2  # Body title RISK and narrative COULD.
    assert result.original_n_words==result.removed_n_words+result.retained_n_words
    path.write_text(raw+' ',encoding='utf-8')
    with pytest.raises(ValueError,match='Reviewed source changed'):
        module.reviewed_boundaries(frame,tmp_path)
