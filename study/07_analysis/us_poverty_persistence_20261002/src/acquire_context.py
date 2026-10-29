"""Acquire pre-period official Census context and record failed alternative access."""
from pathlib import Path
import requests,hashlib,json
from datetime import datetime,timezone
P=Path(__file__).resolve().parents[1]
def get(name,url):
    target=P/'raw'/name;receipt=P/'metadata'/(name+'.receipt.json')
    if target.exists() and receipt.exists():return json.loads(receipt.read_text())
    r=requests.get(url,timeout=(15,45));r.raise_for_status()
    target.write_bytes(r.content)
    item=dict(url=url,final_url=r.url,status=r.status_code,bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest(),retrieved_utc=datetime.now(timezone.utc).isoformat(),path=str(target.relative_to(P)))
    receipt.write_text(json.dumps(item,indent=2));print(name,len(r.content),flush=True);return item
if __name__=='__main__':
    rows=[]
    for name,url in [
        ('saipe_est18all.txt','https://www2.census.gov/programs-surveys/saipe/datasets/2018/2018-state-and-county/est18all.txt'),
        ('saipe_2018_layout.txt','https://www2.census.gov/programs-surveys/saipe/technical-documentation/file-layouts/state-county/2018-estimate-layout.txt')]:
        rows.append(get(name,url))
    (P/'metadata/acquisition_manifest.json').write_text(json.dumps(rows,indent=2))
