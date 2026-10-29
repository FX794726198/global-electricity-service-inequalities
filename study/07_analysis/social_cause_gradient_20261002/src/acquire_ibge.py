"""Acquire documented official Census tables, keeping exact requests and hashes."""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib, json, requests

P=Path(__file__).resolve().parents[1]
for name in ['raw','metadata','data']:(P/name).mkdir(exist_ok=True)
base='https://servicodados.ibge.gov.br/api/v3/agregados'
jobs=[]
# Split all twelve published categories into small, disjoint requests.
meta=json.loads((P/'raw/ibge_10296_metadata.json').read_text())
cats=[str(c['id']) for d in meta['classificacoes'] if d['id']==386 for c in d['categorias']]
for start in range(0,len(cats),3):
    selected=','.join(cats[start:start+3])
    jobs.append((f'ibge_income_bands_{start//3+1}.json',f'{base}/10296/periodos/2022/variaveis/13604',
                 {'localidades':'N6[all]','classificacao':f'2[6794]|86[95251]|386[{selected}]'}))
jobs.append(('ibge_income_mean_median.json',f'{base}/10295/periodos/2022/variaveis/13604|13431|13534',
             {'localidades':'N6[all]','classificacao':'2[6794]|86[95251]|58[95253]'}))

def acquire(job):
    name,url,params=job
    requested=requests.Request('GET',url,params=params).prepare().url
    out=P/'raw'/name;receipt=P/'metadata'/(name+'.receipt.json')
    if out.exists() and receipt.exists():
        prior=json.loads(receipt.read_text())
        if prior['url']==requested and prior['sha256']==hashlib.sha256(out.read_bytes()).hexdigest():
            json.loads(out.read_text());return prior
    r=requests.get(url,params=params,timeout=(15,90));r.raise_for_status();data=r.json()
    assert isinstance(data,list) and data and 'resultados' in data[0],str(data)[:300]
    out.write_bytes(r.content)
    info=dict(url=r.url,requested_url=requested,table=10296 if 'bands' in name else 10295,
              path=str(out.relative_to(P)),bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest(),
              status=r.status_code,retrieved_utc=datetime.now(timezone.utc).isoformat(),provider='IBGE',period=2022)
    receipt.write_text(json.dumps(info,indent=2,ensure_ascii=False));print(name,len(r.content),flush=True)
    return info

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:receipts=list(pool.map(acquire,jobs))
    (P/'metadata/ibge_acquisition.json').write_text(json.dumps(receipts,indent=2,ensure_ascii=False))
