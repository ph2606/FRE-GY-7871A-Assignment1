"""A disclosed sensitivity prompted by observed market-window extremes."""
import pandas as pd
from .analysis import volatility_tests, return_tests


def market_winsor_sensitivity(panel):
    """Cap only four market variables at common-sample 1%/99% quantiles.

    Preserve every accession, text score, size/liquidity control and fixed effect.
    The primary results stay uncapped. This is an influence diagnostic, not a
    claim that a large valid return is an error.
    """
    modified = panel.copy()
    bounds = []
    for name in ['pre_volatility', 'post_volatility', 'prior_excess_return', 'event_return']:
        low, high = panel[name].quantile([.01, .99])
        modified[name] = panel[name].clip(low, high)
        bounds.append({'variable': name, 'lower': low, 'upper': high,
                       'changed': int(modified[name].ne(panel[name]).sum())})
    assert modified.accession.equals(panel.accession)
    return {'winsorized_table5': volatility_tests(modified),
            'winsorized_table6': return_tests(modified),
            'winsorization_bounds': pd.DataFrame(bounds)}