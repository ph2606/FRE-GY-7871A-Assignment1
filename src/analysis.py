"""Exhibits and transparent OLS specifications; no automatic observation deletion."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

MEASURES = ["negative_proportion", "negative_tfidf", "uncertainty_proportion", "uncertainty_tfidf"]
CONTROLS = ["log_size", "log_dollar_volume", "prior_excess_return"]


def add_time(frame):
    out = frame.copy()
    dates = pd.to_datetime(out["filing_date"])
    out["quarter"] = dates.dt.to_period("Q").astype(str)
    out["season"] = dates.dt.quarter.astype(str)
    out["time_years"] = ((dates.dt.year - 2021) * 4 + dates.dt.quarter - 1) / 4
    out["is_10k"] = out["form"].eq("10-K").astype(float)
    # Proportional trend slopes are then percentage points per year.
    for name in ("negative_proportion", "uncertainty_proportion"):
        if name in out:
            out[name + "_pp"] = out[name] * 100
    return out


def _design(frame, numeric, fixed_effects):
    # Build a basis for nuisance fixed effects before introducing regressors.
    # Redundant firm/season dummies need not make a valid trend inestimable.
    # Crucially, saturated quarter effects retain their full span, so they
    # still absorb and reject a common time trend.
    from scipy.linalg import qr
    nuisance = pd.DataFrame({"const": np.ones(len(frame))}, index=frame.index)
    for name in fixed_effects:
        dummies = pd.get_dummies(frame[name].astype(str), prefix=name, drop_first=True, dtype=float)
        nuisance = pd.concat([nuisance, dummies], axis=1)
    _, r, pivot = qr(nuisance.to_numpy(), mode="economic", pivoting=True)
    tolerance = np.finfo(float).eps * max(nuisance.shape) * (abs(r[0, 0]) if r.size else 1)
    rank = int((np.abs(np.diag(r)) > tolerance).sum())
    x = nuisance.iloc[:, sorted(pivot[:rank])].copy()
    for name in numeric:
        if frame[name].nunique() < 2:
            if name == "is_10k":
                continue
            raise ValueError(f"No variation in regressor {name}")
        x[name] = frame[name].astype(float)
    return x.astype(float)


def fit_model(frame, outcome, focal, controls=(), fixed_effects=(), covariance="firm", maxlags=4):
    """Return a report row; fail explicitly on missingness/collinearity or tiny samples.

    Clustered models use small-sample corrections and t reference distributions.
    Two-way clustering allows firm dependence and common quarterly shocks; there
    are only twenty quarter clusters, which must temper inference.
    """
    result = {"outcome": outcome, "regressor": focal, "n": len(frame),
              "n_firms": frame["cik"].nunique() if "cik" in frame else np.nan,
              "n_quarters": frame["quarter"].nunique(), "covariance": covariance,
              "status": "not_estimated", "reason": "", "coefficient": np.nan,
              "standard_error": np.nan, "t_stat": np.nan, "p_value": np.nan,
              "ci_low": np.nan, "ci_high": np.nan, "r_squared": np.nan}
    try:
        cols = list(dict.fromkeys([outcome, focal, *controls, *fixed_effects, "quarter"] + (["cik"] if covariance in ("firm", "firm_quarter") else [])))
        if frame[cols].isna().any().any():
            raise ValueError("Missing model inputs: filtering must occur explicitly upstream")
        if len(frame) < 12:
            raise ValueError("Fewer than 12 observations; analysis is not estimable reliably")
        numeric = list(dict.fromkeys([focal, *controls]))
        x = _design(frame, numeric, fixed_effects)
        y = frame[outcome].astype(float)
        if not np.isfinite(x.to_numpy()).all() or not np.isfinite(y).all():
            raise ValueError("Nonfinite model inputs")
        if np.linalg.matrix_rank(x.to_numpy()) != x.shape[1]:
            raise ValueError("Rank-deficient design; no pseudoinverse coefficient reported")
        if len(frame) - x.shape[1] < 5:
            raise ValueError("Fewer than five residual degrees of freedom")
        model = sm.OLS(y, x, missing="raise")
        ordinary = model.fit()
        if covariance == "HAC":
            if frame["quarter"].nunique() != len(frame) or not pd.PeriodIndex(frame["quarter"], freq="Q").is_monotonic_increasing:
                raise ValueError("HAC expects one chronologically ordered row per quarter")
            fitted = model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags, "use_correction": True}, use_t=True)
        elif covariance in ("firm", "firm_quarter"):
            if frame["cik"].nunique() < 5 or frame["quarter"].nunique() < 5:
                raise ValueError("Fewer than five firm or quarter clusters")
            firm = pd.factorize(frame["cik"])[0]
            groups = firm if covariance == "firm" else np.column_stack([firm, pd.factorize(frame["quarter"])[0]])
            fitted = model.fit(cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True, "df_correction": True}, use_t=True)
        else:
            raise ValueError(f"Unknown covariance {covariance}")
        covariance_matrix = fitted.cov_params()
        focal_variance = float(covariance_matrix.loc[focal, focal])
        se = np.sqrt(focal_variance) if focal_variance > 0 else np.nan
        if not np.isfinite(se) or se <= 0:
            raise ValueError("Nonpositive/nonfinite robust variance; inference unavailable")
        from scipy.stats import t as student_t
        inference_df = getattr(fitted, "df_resid_inference", fitted.df_resid)
        critical = student_t.ppf(.975, inference_df)
        coefficient = float(fitted.params[focal])
        statistic = coefficient / se
        low, high = coefficient - critical * se, coefficient + critical * se
        result.update(status="estimated", coefficient=float(fitted.params[focal]),
                      standard_error=se, t_stat=float(statistic),
                      p_value=float(2 * student_t.sf(abs(statistic), inference_df)), ci_low=float(low), ci_high=float(high),
                      r_squared=float(fitted.rsquared), ols_t_stat=float(ordinary.tvalues[focal]),
                      df_resid=float(fitted.df_resid), inference_df=float(inference_df),
                      negative_covariance_diagonal=int((np.diag(covariance_matrix) < 0).sum()), maxlags=maxlags if covariance == "HAC" else np.nan)
    except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as exc:
        result["reason"] = str(exc)
    return result


def summary_statistics(panel):
    return (panel.groupby("form")[MEASURES + ["n_words"]]
            .describe(percentiles=[.25, .5, .75]).stack(level=0, future_stack=True)
            .rename_axis(index=["form", "measure"]).reset_index())


def quarterly_means(panel):
    out = panel.groupby(["quarter", "form"])[MEASURES].mean().reset_index()
    sizes = panel.groupby(["quarter", "form"]).agg(n_filings=("accession", "size"), n_firms=("cik", "nunique")).reset_index()
    return out.merge(sizes, on=["quarter", "form"], validate="one_to_one")


def trend_tests(panel, lags=(4, 1, 2)):
    if panel[MEASURES + ["cik", "quarter", "form"]].isna().any().any():
        raise ValueError("Missing trend inputs would change the aggregation sample")
    rows = []
    for form in ("pooled", "10-K", "10-Q"):
        sub = panel if form == "pooled" else panel.loc[panel.form.eq(form)]
        numeric = ["is_10k"] if form == "pooled" else []
        for measure in MEASURES:
            outcome = measure + "_pp" if measure.endswith("proportion") else measure
            # Equal filing/firm weights: upstream permits one filing per firm-quarter.
            agg = sub.groupby("quarter")[outcome].mean().reset_index().sort_values("quarter")
            periods = pd.PeriodIndex(agg["quarter"], freq="Q")
            agg["time_years"] = [(p.year - 2021) + (p.quarter - 1) / 4 for p in periods]
            agg["season"] = [str(p.quarter) for p in periods]
            for lag in lags:
                row = fit_model(agg, outcome, "time_years", fixed_effects=["season"], covariance="HAC", maxlags=lag)
                if len(agg) != 20:
                    row.update(status="not_estimated", reason="Not all 20 quarters observed; no compressed-time HAC inference", coefficient=np.nan, standard_error=np.nan, t_stat=np.nan, p_value=np.nan, ci_low=np.nan, ci_high=np.nan)
                row.update(form=form, measure=measure, specification="aggregate_season_adjusted", robustness=lag != 4)
                rows.append(row)
            row = fit_model(sub, outcome, "time_years", numeric, ["cik", "season"], "firm")
            row.update(form=form, measure=measure, specification="within_firm_season_adjusted", robustness=False)
            rows.append(row)
    return pd.DataFrame(rows)


def volatility_tests(panel):
    rows = []
    for form in ("pooled", "10-K", "10-Q"):
        sub = panel if form == "pooled" else panel.loc[panel.form.eq(form)]
        for measure in ["uncertainty_proportion", "uncertainty_tfidf"]:
            for controlled in [False, True]:
                controls = CONTROLS + (["is_10k"] if form == "pooled" else []) + (["pre_volatility"] if controlled else [])
                row = fit_model(sub, "post_volatility", measure, controls, ["cik", "quarter"], "firm_quarter")
                row.update(form=form, pre_volatility_control=controlled, specification="firm_and_quarter_FE")
                rows.append(row)
    return pd.DataFrame(rows)


def return_tests(panel):
    rows = []
    for measure in ["negative_proportion", "negative_tfidf"]:
        row = fit_model(panel, "event_return", measure,
                        CONTROLS + ["pre_volatility", "is_10k"], ["cik", "quarter"], "firm_quarter")
        # Approximate ex ante detectable effect under normal 5% two-sided/80% power.
        # This is a design diagnostic conditional on the estimated SE, not observed power.
        row["mde_80pct_per_unit"] = (1.959963984540054 + .8416212335729143) * row["standard_error"]
        row["regressor_sd"] = panel[measure].std(ddof=1)
        row["mde_80pct_one_sd_return_pp"] = 100 * row["mde_80pct_per_unit"] * row["regressor_sd"]
        row["specification"] = "firm_and_quarter_FE"
        rows.append(row)
    return pd.DataFrame(rows)


def figure_quarterly(quarterly, vix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    quarters = pd.period_range("2021Q1", "2025Q4", freq="Q").astype(str)
    v = vix.copy()
    v.index = pd.to_datetime(v.index)
    v = v.loc["2021-01-01":"2025-12-31"].groupby(v.loc["2021-01-01":"2025-12-31"].index.to_period("Q")).mean()
    v.index = v.index.astype(str)
    style = {"font.family": "DejaVu Sans", "font.size": 11,
             "axes.titlesize": 12, "axes.labelsize": 10,
             "xtick.labelsize": 10, "ytick.labelsize": 10,
             "text.color": "#202636", "axes.labelcolor": "#465064",
             "xtick.color": "#465064", "ytick.color": "#465064",
             "pdf.fonttype": 42, "ps.fonttype": 42}
    titles = ["A  Negative sentiment · proportion", "B  Negative sentiment · tf.idf",
              "C  Uncertainty · proportion", "D  Uncertainty · tf.idf"]
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.2), sharex=True)
        for ax, measure, title in zip(axes.flat, MEASURES, titles):
            for form, color, marker in [("10-K", "#57068C", "o"),
                                         ("10-Q", "#087E8B", "s")]:
                values = quarterly.loc[quarterly.form.eq(form)].set_index("quarter")[measure].reindex(quarters)
                factor = 100 if measure.endswith("proportion") else 1
                ax.plot(range(20), values * factor, marker=marker, markersize=3.1,
                        linewidth=1.7, label=form, color=color, zorder=3)
            ax.set_title(title, loc="left", fontweight="normal", pad=11)
            ax.set_ylabel("Words (%)" if measure.endswith("proportion") else "Weighted score")
            ax.set_xlim(-.4, 19.4)
            ax.set_xticks([0, 4, 8, 12, 16], ["2021", "2022", "2023", "2024", "2025"])
            ax.tick_params(length=0, pad=5)
            ax.grid(axis="y", color="#E7E9EE", linewidth=.7, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            for edge in ["left", "bottom"]:
                ax.spines[edge].set_color("#D4D8E1")
                ax.spines[edge].set_linewidth(.7)
            other = ax.twinx()
            other.plot(range(20), v.reindex(quarters), color="#7C8290",
                       linewidth=1.2, linestyle=(0, (4, 3)), label="VIX (right axis)", zorder=1)
            other.set_ylabel("VIX", color="#737B88")
            other.tick_params(length=0, pad=5, colors="#737B88")
            other.spines[["top", "left", "bottom"]].set_visible(False)
            other.spines["right"].set_color("#D4D8E1")
            other.spines["right"].set_linewidth(.7)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        vix_handle = other.get_legend_handles_labels()[0][0]
        fig.legend(handles + [vix_handle], labels + ["VIX (right axis)"],
                   loc="upper center", bbox_to_anchor=(.5, 1.005), ncol=3,
                   frameon=False, handlelength=2.7, columnspacing=2.5)
        fig.supxlabel("Calendar year · quarterly observations", fontsize=10, y=.01, color="#465064")
        fig.tight_layout(rect=(0, .035, 1, .925), h_pad=2.1, w_pad=2.2)
    return fig
