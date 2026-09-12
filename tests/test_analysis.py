"""Independent regression and inference checks on a small unbalanced panel."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from scipy.stats import norm

from src.analysis import add_time, fit_model, return_tests, trend_tests, volatility_tests


@pytest.fixture
def panel():
    """Missing firm-quarters and changing annual-filing seasons prevent balance shortcuts."""
    rng = np.random.default_rng(2291)
    periods = pd.period_range("2021Q1", "2025Q4", freq="Q")
    quarterly_shock = rng.normal(0, 0.008, len(periods))
    rows = []
    for firm in range(12):
        for q, period in enumerate(periods):
            if (firm * 3 + q) % 11 == 0:
                continue
            is_10k = q % 4 == (firm + q // 4) % 4
            negative = (0.015 + 0.0008 * q / 4 + firm * 0.0002
                        + 0.0011 * is_10k + (q % 4) * 0.0003 + rng.normal(0, 0.001))
            uncertainty = 0.009 + 0.0002 * q / 4 + firm * 0.0001 + rng.normal(0, 0.001)
            pre_volatility = rng.uniform(0.2, 0.8)
            log_size = 20 + 0.1 * firm + rng.normal(0, 0.5)
            prior_return = rng.normal(0, 0.08)
            rows.append({
                "cik": f"cik-{firm}", "accession": f"{firm}-{q}",
                "filing_date": period.start_time + pd.Timedelta(days=35),
                "form": "10-K" if is_10k else "10-Q",
                "negative_proportion": negative,
                # Scaling identity gives an independent check of MDE units.
                "negative_tfidf": 1000 * negative,
                "uncertainty_proportion": uncertainty,
                "uncertainty_tfidf": 1000 * uncertainty,
                "log_size": log_size,
                "log_dollar_volume": 16 + 0.12 * firm + rng.normal(0, 0.6),
                "prior_excess_return": prior_return,
                "pre_volatility": pre_volatility,
                "post_volatility": (0.12 + 0.35 * pre_volatility + 4 * uncertainty
                                    + 0.006 * log_size + firm * 0.003
                                    + quarterly_shock[q] + rng.normal(0, 0.025)),
                "event_return": (-0.4 * negative + 0.02 * prior_return
                                 + quarterly_shock[q] + rng.normal(0, 0.025)),
                "n_words": 5000,
            })
    return add_time(pd.DataFrame(rows))


def test_common_within_firm_trend_matches_formula_lsdv_on_unbalanced_panel(panel):
    row = fit_model(panel, "negative_proportion_pp", "time_years",
                    ["is_10k"], ["cik", "season"], "firm")
    reference = smf.ols(
        "negative_proportion_pp ~ time_years + is_10k + C(cik) + C(season)", data=panel
    ).fit(cov_type="cluster", cov_kwds={"groups": panel["cik"], "use_correction": True,
                                      "df_correction": True}, use_t=True)
    assert row["status"] == "estimated", row["reason"]
    assert row["coefficient"] == pytest.approx(reference.params["time_years"], rel=1e-10)
    assert row["standard_error"] == pytest.approx(reference.bse["time_years"], rel=1e-9)
    assert row["n"] == reference.nobs == len(panel)
    assert panel.groupby("cik").size().nunique() > 1


def test_trend_table_retains_the_common_lsdv_coefficient(panel):
    table = trend_tests(panel, lags=(4,))
    selected = table.loc[(table["form"] == "pooled")
                         & (table["measure"] == "negative_proportion")
                         & (table["specification"] == "within_firm_season_adjusted")].iloc[0]
    reference = smf.ols(
        "negative_proportion_pp ~ time_years + is_10k + C(cik) + C(season)", data=panel
    ).fit()
    assert selected["status"] == "estimated", selected["reason"]
    assert selected["coefficient"] == pytest.approx(reference.params["time_years"], rel=1e-10)


def test_trend_aggregation_rejects_missing_scores_before_group_means(panel):
    broken = panel.copy()
    broken.loc[0, "negative_proportion"] = np.nan
    with pytest.raises(ValueError, match="Missing trend inputs"):
        trend_tests(broken, lags=(4,))


def test_saturated_quarter_effects_absorb_the_common_trend(panel):
    row = fit_model(panel, "negative_proportion_pp", "time_years",
                    ["is_10k"], ["cik", "quarter"], "firm")
    assert row["status"] == "not_estimated"
    assert "Rank-deficient" in row["reason"]
    assert np.isnan(row["coefficient"])


def test_season_effects_nested_in_firms_do_not_destroy_an_identified_trend():
    rng = np.random.default_rng(35)
    rows = []
    # A realistic annual-filing pattern: each firm's 10-K always arrives in
    # the same season, but different firms use different seasons. Season FE
    # are redundant with firm FE; the within-firm time variation is not.
    for firm in range(12):
        for year in range(5):
            season = firm % 4 + 1
            time = year + (season - 1) / 4
            rows.append({"cik": str(firm), "quarter": f"{2021 + year}Q{season}",
                         "season": str(season), "time_years": time,
                         "tone": 0.2 * time + 0.1 * firm + rng.normal(0, 0.02)})
    sample = pd.DataFrame(rows)
    nested = fit_model(sample, "tone", "time_years", fixed_effects=["cik", "season"], covariance="firm")
    reference = smf.ols("tone ~ time_years + C(cik)", data=sample).fit(
        cov_type="cluster", cov_kwds={"groups": sample["cik"], "use_correction": True,
                                      "df_correction": True}, use_t=True)
    assert nested["status"] == "estimated", nested["reason"]
    assert nested["coefficient"] == pytest.approx(reference.params["time_years"], rel=1e-10)
    assert nested["standard_error"] == pytest.approx(reference.bse["time_years"], rel=1e-9)
    assert nested["df_resid"] == reference.df_resid


@pytest.mark.parametrize("column", ["negative_proportion_pp", "time_years", "is_10k", "cik", "season"])
def test_missing_inputs_never_silently_shrink_the_regression(panel, column):
    broken = panel.copy()
    broken.loc[0, column] = np.nan
    row = fit_model(broken, "negative_proportion_pp", "time_years",
                    ["is_10k"], ["cik", "season"], "firm")
    assert row["status"] == "not_estimated"
    assert "Missing model inputs" in row["reason"]
    assert row["n"] == len(panel)
    assert np.isnan(row["coefficient"])


def test_missing_cluster_label_is_rejected_without_a_firm_fixed_effect(panel):
    broken = panel.copy()
    broken.loc[0, "cik"] = np.nan
    row = fit_model(broken, "negative_proportion_pp", "time_years", covariance="firm")
    assert row["status"] == "not_estimated"
    assert "Missing" in row["reason"]
    assert row["n"] == len(panel)


def test_hac_rejects_unsorted_quarters_instead_of_using_arbitrary_row_order(panel):
    aggregate = panel.groupby("quarter", as_index=False).agg(
        negative_proportion_pp=("negative_proportion_pp", "mean"),
        time_years=("time_years", "first"),
    )
    ordered = fit_model(aggregate, "negative_proportion_pp", "time_years", covariance="HAC")
    assert ordered["status"] == "estimated", ordered["reason"]
    shuffled = aggregate.sample(frac=1, random_state=24)
    unordered = fit_model(shuffled, "negative_proportion_pp", "time_years", covariance="HAC")
    assert unordered["status"] == "not_estimated"
    assert any(word in unordered["reason"].lower() for word in ["chronolog", "order"])


def test_paired_volatility_models_use_identical_complete_samples(panel):
    table = volatility_tests(panel)
    for (form, regressor), pair in table.groupby(["form", "regressor"]):
        assert len(pair) == 2
        assert set(pair["pre_volatility_control"]) == {False, True}
        expected = len(panel) if form == "pooled" else panel["form"].eq(form).sum()
        assert pair["n"].tolist() == [expected, expected]
    pooled = table.loc[table["form"] == "pooled"]
    assert pooled["status"].eq("estimated").all(), pooled[["status", "reason"]].to_dict("records")
    # Prefiling risk carries independent information in this constructed sample.
    assert pooled.loc[pooled["pre_volatility_control"], "coefficient"].iloc[0] != pytest.approx(
        pooled.loc[~pooled["pre_volatility_control"], "coefficient"].iloc[0])


def test_missing_prefiling_control_does_not_create_a_smaller_paired_sample(panel):
    broken = panel.copy()
    broken.loc[0, "pre_volatility"] = np.nan
    table = volatility_tests(broken)
    controlled = table.loc[(table["form"] == "pooled") & table["pre_volatility_control"]]
    assert controlled["n"].eq(len(panel)).all()
    assert controlled["status"].eq("not_estimated").all()
    assert controlled["reason"].str.contains("Missing model inputs").all()


def test_return_mde_is_in_percentage_points_and_invariant_to_signal_units(panel):
    table = return_tests(panel).set_index("regressor")
    assert table["status"].eq("estimated").all(), table[["status", "reason"]].to_dict("index")
    proportional = table.loc["negative_proportion"]
    weighted = table.loc["negative_tfidf"]
    normal_critical_sum = norm.ppf(0.975) + norm.ppf(0.8)
    expected_return_fraction = normal_critical_sum * proportional["standard_error"] * panel["negative_proportion"].std(ddof=1)
    assert proportional["mde_80pct_one_sd_return_pp"] == pytest.approx(100 * expected_return_fraction)
    assert weighted["standard_error"] * 1000 == pytest.approx(proportional["standard_error"], rel=1e-8)
    assert weighted["mde_80pct_one_sd_return_pp"] == pytest.approx(proportional["mde_80pct_one_sd_return_pp"], rel=1e-8)
