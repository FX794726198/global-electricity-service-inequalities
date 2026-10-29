"""Download live-linked monthly public Table 5 workbooks, with receipts."""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import json, hashlib, unicodedata
import requests

ROOT = Path(__file__).resolve().parents[1]
links = json.loads((ROOT/'metadata/sedas_table5_links.json').read_text())
months = ['OCAK','SUBAT','MART','NISAN','MAYIS','HAZIRAN','TEMMUZ','AGUSTOS','EYLUL','EKIM','KASIM','ARALIK']
def ascii_upper(s):
    return ''.join(c for c in unicodedata.normalize('NFKD', s.upper().replace('İ','I')) if not unicodedata.combining(c))
jobs = {}
for row in links:
    label = ascii_upper(row['label'])
    for year in [2024,2025]:
        if str(year) in label and label.endswith('.XLSX'):
            matches = [i+1 for i,m in enumerate(months) if m in label]
            if len(matches) == 1:
                key = f'sedas_table5_{year}_{matches[0]:02d}'
                if key in jobs and jobs[key]['url'] != row['url']:
                    raise ValueError(f'Ambiguous monthly workbook: {key}')
                jobs[key] = row | dict(year=year,month=matches[0])
assert len(jobs) == 24, len(jobs)
def get(pair):
    key,job=pair
    p=ROOT/'raw'/(key+'.xlsx')
    if p.exists(): return dict(file=p.name,status='already_archived')
    meta=job | dict(retrieved_utc=datetime.now(timezone.utc).isoformat())
    try:
        r=requests.get(job['url'],timeout=(15,40))
        meta.update(status=r.status_code,bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest())
        p=ROOT/'raw'/(key+('.xlsx' if r.content[:2]==b'PK' else '.response'))
        p.write_bytes(r.content); meta['file']=str(p.relative_to(ROOT))
    except requests.RequestException as e: meta['error']=str(e)
    (ROOT/'metadata'/(key+'_receipt.json')).write_text(json.dumps(meta,indent=2,ensure_ascii=False))
    return meta
with ThreadPoolExecutor(max_workers=2) as pool:
    for m in pool.map(get,jobs.items()): print(json.dumps(m,ensure_ascii=False),flush=True)
