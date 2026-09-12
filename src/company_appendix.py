"""Company identities, public-listing evidence, and sequential sample eligibility."""
from pathlib import Path
import hashlib
import json
import re

import pandas as pd


# Sources supplement the reviewed issuer mapping for non-US listings and funds.
LISTING_SOURCES = {
    '2618': ('Public company', 'Hong Kong', 'https://www1.hkexnews.hk/listedco/listconews/sehk/2025/1009/2025100900073.pdf'),
    '4689': ('Public company', 'Tokyo', 'https://www.lycorp.co.jp/en/ir/faq.html'),
    '6301': ('Public company', 'Tokyo', 'https://www.komatsu.jp/en/ir/faq'),
    'KMTUY': ('Public company; ADR holding', 'Tokyo issuer / OTC ADR', 'https://www.komatsu.jp/en/ir/shares'),
    'SE': ('Public company', 'NYSE', 'https://www.sec.gov/Archives/edgar/data/1703399/000119312526088151/d106475dex991.htm'),
    'DSY FP': ('Public company', 'Euronext Paris', 'https://investor.3ds.com/'),
    'ETHQ/U': ('Listed fund', 'Toronto', 'https://www.3iq.io/press-media/3iq-seeks-to-deliver-north-americas-first-solana-etp'),
    'SOLQ/U': ('Listed fund', 'Toronto', 'https://www.3iq.io/press-media/3iq-launches-solana-staking-etf-solq-backed-by-lead-investors-including-skybridge-capital'),
}


def filter_counts(ledger):
    """Counts after each filter; a filing is removed at its first failed stage."""
    frame = ledger.copy()
    frame['cik'] = frame.cik.astype(str).str.zfill(10)
    def stage(value):
        if value == 'retained':
            return 9
        match = re.match(r'([1-8]) ', value)
        if not match:
            raise ValueError('Unknown exclusion stage: '+str(value))
        return int(match[1])
    frame['failed_at'] = frame.exclusion_stage.map(stage)
    labels = [(0,'Candidates'),(1,'Original'),(2,'Word minimum'),(3,'Text'),
              (4,'Price'),(5,'History'),(6,'Windows'),(7,'Shares'),(8,'Final')]
    counts = pd.DataFrame(index=sorted(frame.cik.unique()))
    counts.index.name = 'cik'
    for n,label in labels:
        counts[label] = frame[frame.failed_at.gt(n)].groupby('cik').size().reindex(counts.index,fill_value=0)
    if not (counts.diff(axis=1).iloc[:,1:] <= 0).all().all():
        raise ValueError('Non-monotone company filter counts')
    return counts.reset_index()


def build_company_appendix(root, risk_panel, rankings, *, save=True):
    root = Path(root)
    holdings = pd.read_csv(root/'data/universe/holdings_resolution.csv', dtype=str, keep_default_na=False)
    universe = pd.read_csv(root/'data/universe/universe.csv', dtype={'cik':str}, keep_default_na=False)
    ledger = pd.read_csv(root/'outputs/exclusions.csv', dtype={'cik':str})
    counts = filter_counts(ledger)
    companies = universe[universe.status.eq('domestic_filer')].merge(counts,on='cik',validate='one_to_one')
    retained = ledger[ledger.exclusion_stage.eq('retained')]
    for form,col in [('10-K','Final K'),('10-Q','Final Q')]:
        companies[col] = companies.cik.map(retained[retained.form.eq(form)].groupby('cik').size()).fillna(0).astype(int)
    periods = risk_panel.groupby('cik').filing_date.agg(['min','max'])
    companies['First filing'] = companies.cik.map(periods['min']).astype(str).str[:10]
    companies['Last filing'] = companies.cik.map(periods['max']).astype(str).str[:10]
    if 'sic_desc' in risk_panel:
        companies['Industry'] = companies.cik.map(risk_panel.groupby('cik').sic_desc.first())
    else:
        companies['Industry'] = ''
    for form,label in [('10-K','K'),('10-Q','Q')]:
        subset = rankings[rankings.form.eq(form)].set_index('cik')
        companies[label+' level'] = companies.cik.map(subset.level_status).eq('ranked').map({True:'Yes',False:'No'})
        companies[label+' trend'] = companies.cik.map(subset.trend_status).eq('ranked').map({True:'Yes',False:'No'})
    valid = risk_panel[risk_panel.status.isin(['found','absent'])]
    companies['Ex Item1A'] = companies.cik.map(valid.groupby('cik').size()).fillna(0).astype(int)
    companies['Market eligible'] = companies.Final.gt(0).map({True:'Yes',False:'No'})
    identity_rows = []
    metadata_hashes = {}
    for row in holdings.to_dict('records'):
        raw, cik = row['raw_ticker'], row['cik']
        public, exchange, proof, asof = 'Not established', '', row['mapping_source_url'], '2026 snapshot / reviewed source'
        if cik:
            url = 'https://data.sec.gov/submissions/CIK'+cik+'.json'
            path = root/'data/filings/metadata'/(hashlib.sha256(url.encode()).hexdigest()+'.json')
            if path.exists():
                data = json.loads(path.read_text(encoding='utf-8'))
                metadata_hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
                if data.get('tickers') and data.get('exchanges'):
                    public, exchange, proof = 'Public company', '/'.join(dict.fromkeys(data['exchanges'])), url
                sidecar = path.with_name(path.name+'.provenance.json')
                if sidecar.exists():
                    asof = json.loads(sidecar.read_text()).get('retrieved_at_utc',asof)
        if raw in LISTING_SOURCES:
            public, exchange, proof = LISTING_SOURCES[raw]
            asof = '2026-09-12 (source review)'
        elif row['status'] == 'reviewed_foreign_listing':
            public, exchange = 'Public company', 'Foreign listing'
        elif row['status'] == 'reviewed_fund_instrument':
            public, exchange = 'Listed fund', 'US ETF'
        passed = companies[companies.cik.eq(cik)]
        row.update(public_status=public, listing=exchange, listing_source=proof,
                   listing_evidence_date=str(asof),
                   text_eligible='Yes' if len(passed) and passed.iloc[0].Text>0 else 'No',
                   market_eligible='Yes' if len(passed) and passed.iloc[0].Final>0 else 'No')
        identity_rows.append(row)
    identities = pd.DataFrame(identity_rows).sort_values(['raw_ticker','ark_name']).reset_index(drop=True)
    companies = companies.sort_values('ticker').reset_index(drop=True)
    summary = {'holding_identities':len(identities),'eligible_issuers':len(companies),
               'market_eligible_issuers':int(companies.Final.gt(0).sum()),
               'public_status_counts':identities.public_status.value_counts().to_dict(),
               'filter_totals':{c:int(companies[c].sum()) for c in counts.columns if c!='cik'},
               'metadata_sha256':metadata_hashes}
    assert companies['Final K'].sum()+companies['Final Q'].sum() == companies.Final.sum()
    if save:
        out=root/'outputs';out.mkdir(exist_ok=True)
        identities.to_csv(out/'company_identities.csv',index=False)
        companies.to_csv(out/'company_filter_appendix.csv',index=False)
        (out/'company_appendix_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return identities, companies, summary
