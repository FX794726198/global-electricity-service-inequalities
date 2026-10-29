"""Acquire public household sources and retain source-specific provenance."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from urllib.parse import urlparse, unquote
import hashlib
import json
import zipfile
import requests

P=Path(__file__).resolve().parents[1]
catalog=json.loads((P/'raw/energydata_mtf_catalog.json').read_text())['result']['results']
jobs=[]
for d in catalog:
    if 'multi-tier' not in d['name']:
        continue
    country=d['title'].split(' - ')[0]
    for r in d['resources']:
        url=r.get('url',''); name=r.get('name',''); low=(url+' '+name).lower()
        if not url or any(x in low for x in ['questionnaire_community','questionnaire_facility','executive_summary']):
            continue
        dest=P/'raw/mtf'/d['name']/unquote(Path(urlparse(url).path).name)
        jobs.append(dict(dataset=d['name'],country=country,resource_name=name,url=url,path=str(dest.relative_to(P)),
                         dataset_url='https://energydata.info/dataset/'+d['name'],license=d.get('license_title'),
                         metadata_modified=d.get('metadata_modified'),role='Public household data or documentation'))
for name,url,role in [
    ('afrobarometer_r9.sav','https://www.afrobarometer.org/wp-content/uploads/2025/06/R9.Merge_39ctry.20Nov23.final_.release_Updated.4Jun25-3.sav','Adult survey microdata; household electricity experience'),
    ('afrobarometer_r9_codebook.pdf','https://www.afrobarometer.org/wp-content/uploads/2025/07/AB_R9.MergeCodebook_25Jun24.final_.pdf','Questionnaire labels and codes')]:
    jobs.append(dict(dataset='Afrobarometer Round 9',country='39 African countries',resource_name=name,url=url,
                     path='raw/afrobarometer/'+name,dataset_url='https://www.afrobarometer.org/data/merged-data/',
                     license='Afrobarometer attribution and source terms; preserved codebook',role=role))
(P/'metadata/download_jobs.json').write_text(json.dumps(jobs,ensure_ascii=False,indent=2))

def get(j):
    f=P/j['path']; f.parent.mkdir(parents=True,exist_ok=True)
    row=dict(j)
    try:
        if not f.exists():
            r=requests.get(j['url'],timeout=(20,45),stream=True); r.raise_for_status()
            tmp=f.with_suffix(f.suffix+'.part')
            with tmp.open('wb') as out:
                for block in r.iter_content(1024*1024):out.write(block)
            tmp.replace(f)
            row['content_type']=r.headers.get('Content-Type');row['final_url']=r.url
        row.update(status='downloaded',bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                   retrieved_utc=datetime.now(timezone.utc).isoformat())
        if f.suffix.lower()=='.zip':
            with zipfile.ZipFile(f) as z:
                row['archive_members']=z.namelist()
                target=f.parent/'extracted'
                for info in z.infolist():
                    q=(target/info.filename).resolve()
                    if target.resolve() not in q.parents or '__MACOSX' in info.filename or Path(info.filename).name.startswith('._'):
                        continue
                    if not info.is_dir():
                        q.parent.mkdir(parents=True,exist_ok=True)
                        q.write_bytes(z.read(info))
    except Exception as e:
        row.update(status='failed',error=str(e))
    return row

rows=[]
with ThreadPoolExecutor(max_workers=5) as pool:
    for future in as_completed([pool.submit(get,j) for j in jobs]):
        row=future.result();rows.append(row)
        print(row['status'],row['country'],row['resource_name'],row.get('bytes',0),flush=True)
        (P/'metadata/source_receipts.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
print('Completed',len(rows),'resources; successful',sum(r['status']=='downloaded' for r in rows),flush=True)
