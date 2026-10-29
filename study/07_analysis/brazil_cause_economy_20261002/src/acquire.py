"""Download a public, commit-pinned secondary archive; never execute its code."""
from pathlib import Path
import requests, json, hashlib, datetime
from concurrent.futures import ThreadPoolExecutor

P=Path(__file__).resolve().parents[1]
REV='e02d65674c5098067d63668fd5932f533fe494df'
BASE=f'https://raw.githubusercontent.com/TomasAugusto/InterrupcoesDeEnergia/{REV}/'
FILES=['README.md','Dados/manifesto.json','zeki/dados.py','zeki/curadoria/marts.py','zeki/curadoria/esquema.py',
       'zeki/curadoria/dimensoes.py','zeki/curadoria/fato.py','zeki/qualidade.py',
       'zeki/fontes/aneel.py','Dados/ouro/mart_mensal.parquet','Dados/ouro/mart_duracao.parquet',
       'Dados/ouro/dim_conjunto.parquet','Dados/ouro/ponte_conjunto_municipio.parquet',
       'Dados/dm-interrupcoes-de-energia-eletrica-nas-redes-de-distribuicao.pdf']

def fetch(name):
    url=BASE+name; dest=P/'raw'/name;dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        r=requests.get(url,timeout=(15,90));r.raise_for_status();dest.write_bytes(r.content)
    b=dest.read_bytes()
    return dict(path=str(dest.relative_to(P)),url=url,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),
                source_role='secondary publisher aggregate or code; ANEEL-origin claim',revision=REV)

if __name__=='__main__':
    rows=list(ThreadPoolExecutor(4).map(fetch,FILES))
    receipt=dict(acquired_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 original_catalog='https://dadosabertos.aneel.gov.br/dataset/interrupcoes-de-energia-eletrica-nas-redes-de-distribuicao',
                 official_numeric_acquisition=False,individual_events_acquired=False,files=rows)
    (P/'metadata/acquisition.json').write_text(json.dumps(receipt,indent=2))
    for x in rows:print(x['path'],x['bytes'],flush=True)
