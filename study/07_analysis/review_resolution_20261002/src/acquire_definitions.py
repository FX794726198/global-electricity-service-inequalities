"""Archive published definition rows without selecting on reliability outcomes."""
from pathlib import Path
import concurrent.futures, hashlib, json, re, time
import requests
import pandas as pd

P=Path(__file__).resolve().parents[1]
raw=P/'raw/worldbank_profiles';raw.mkdir(exist_ok=True)
docs=json.loads((P/'raw/worldbank_profile_search.json').read_text())['documents']
docs=[v for v in docs.values() if isinstance(v,dict) and v.get('lang')=='English'
      and 'Economy Profile of' in str(v.get('docna')) and v.get('txturl')]

def get(v):
    title=' '.join(v['docna']['0']['docna'].split())
    country=title.split('Economy Profile of ',1)[1]
    u=v['txturl'].replace('http://documents.worldbank.org/','https://documents1.worldbank.org/')
    dest=raw/(v['id']+'.txt')
    receipt=dict(document_id=v['id'],country=country,catalogue_country=v.get('count'),
                 title=title,url=u,pdf_url=v['pdfurl'].replace('http:','https:'),
                 downloaded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    try:
        if dest.exists():s=dest.read_text()
        else:
            r=requests.get(u,timeout=35);r.raise_for_status();s=r.text
            assert 'Doing Business 2020' in s and len(s)>10000
            dest.write_text(s)
        receipt.update(status='archived',bytes=dest.stat().st_size,
                       sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
        rows=[]
        for j,m in enumerate(re.finditer(r'What is the minimum outage time[^\r\n]*',s,re.I)):
            line=m.group().strip();tail=re.split(r'SAIDI\s*/\s*SAIFI',line,flags=re.I)[-1].strip()
            value=float(tail) if re.fullmatch(r'\d+(?:\.\d+)?',tail) else None
            before=s[max(0,m.start()-1000):m.start()]
            def metric(code):
                mm=re.findall(r'\('+code+r'\)\s*([^\r\n]+)',before)
                val=mm[-1].strip().replace(',','') if mm else ''
                return float(val) if re.fullmatch(r'\d+(?:\.\d+)?',val) else None
            rows.append(dict(document_id=v['id'],country=country,table_order=j+1,
                             minimum_minutes=value,minimum_native=tail,
                             saidi_published=metric('SAIDI'),saifi_published=metric('SAIFI'),
                             source_line=line,source_context=before[-650:],source_url=u))
        receipt['definition_tables']=len(rows)
        return receipt,rows
    except Exception as e:
        receipt.update(status='failed',error=str(e));return receipt,[]

receipts=[];rows=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    for i,(rec,rr) in enumerate(ex.map(get,docs),1):
        receipts.append(rec);rows.extend(rr)
        if i%30==0:print(f'Archived {i}/{len(docs)} profile responses',flush=True)
pd.DataFrame(rows).to_csv(P/'data/profile_definition_rows.csv',index=False)
pd.DataFrame(receipts).to_csv(P/'metadata/profile_acquisition.csv',index=False)
print(pd.DataFrame(receipts).status.value_counts().to_dict(), 'tables',len(rows),flush=True)
