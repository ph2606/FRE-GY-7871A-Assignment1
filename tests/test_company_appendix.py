import pandas as pd
import pytest
from src.company_appendix import filter_counts

def test_company_counts_follow_first_failed_filter():
    ledger=pd.DataFrame({'cik':['1']*5+['2'],
        'exclusion_stage':['1 Amendments or failed parsing','3 Earliest filing per CIK/calendar quarter',
                           '4 Usable day 0 and price','7 Filing-specific contemporaneous shares','retained','retained']})
    result=filter_counts(ledger).set_index('cik')
    assert result.loc['0000000001',['Candidates','Original','Word minimum','Text','Price','History','Windows','Shares','Final']].tolist()==[5,4,4,3,2,2,2,1,1]
    assert result.Final.sum()==2

def test_unknown_filter_label_is_not_silently_retained():
    with pytest.raises(ValueError):
        filter_counts(pd.DataFrame({'cik':['1'],'exclusion_stage':['unknown']}))
