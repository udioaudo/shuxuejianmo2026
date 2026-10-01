"""Required C3 time-series descriptive analysis with source and count flags."""

from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
PATH=ROOT/'02_data/raw/real_attachments/C_efficiency_evolution/leaderboard_extended_timeseries.csv'

def main():
    data=pd.read_csv(PATH)
    required={'Model','Year','Average','Source'}
    if not required.issubset(data):raise ValueError('C3 columns changed')
    data['Year']=pd.to_numeric(data.Year,errors='coerce')
    data['Average']=pd.to_numeric(data.Average,errors='coerce')
    table=[]
    for (year,source),g in data.dropna(subset=['Year','Average']).groupby(['Year','Source']):
        table.append({'year':int(year),'source':str(source),'n':len(g),
                      'median':float(g.Average.median()),
                      'q90':float(g.Average.quantile(.9)),
                      'max':float(g.Average.max())})
    report={'source':'C3 leaderboard_extended_timeseries.csv',
            'rows':len(data),'by_source_and_year':table,
            'usage':'historical ability time-series context only; matched C1/C4 model drives association fit',
            'warning':'Early historical years have tiny samples; year-specific maximum is not a stable ability frontier.'}
    (ROOT/'05_results/tables/q4_C3_timeseries_profile.json').write_text(
        json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'rows':len(data),'year_source_cells':len(table),
                      'recent': [r for r in table if r['year'] in (2024,2025)]},ensure_ascii=False))

if __name__=='__main__':main()
