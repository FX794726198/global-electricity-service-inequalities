"""Retrieve public source notes and the US Census poverty table, with receipts."""
from pathlib import Path
import hashlib,json,requests
from datetime import datetime,timezone
P=Path(__file__).resolve().parents[1]
JOBS={
'ibge_census2022_income_presentation.pdf':'https://agenciadenoticias.ibge.gov.br/media/com_mediaibge/arquivos/53fde9c24f6d8df3108c83f6378e018f.pdf',
'ibge_table_symbols.pdf':'https://ftp.ibge.gov.br/Indicadores_Sociais/Sintese_de_Indicadores_Sociais/Indicadores_Sociais_2018_tabulacoes_especiais/documentacao/Convencoes_Utilizadas_nas_Tabelas.pdf',
'acsdt5y2022-b17001.dat':'https://www2.census.gov/programs-surveys/acs/summary_file/2022/table-based-SF/data/5YRData/acsdt5y2022-b17001.dat',
'acs_b17001_metadata.json':'https://api.census.gov/data/2022/acs/acs5/groups/B17001.json'}
for name,url in JOBS.items():
    f=P/'raw'/name
    receipt=P/'metadata'/(name+'.receipt.json')
    if name=='acsdt5y2022-b17001.dat':receipt=P/'metadata/acs_poverty_receipt.json'
    if f.exists() and receipt.exists():
        meta=json.loads(receipt.read_text())
        assert hashlib.sha256(f.read_bytes()).hexdigest()==meta['sha256']
        print('Verified cached',name);continue
    r=requests.get(url,stream=True,timeout=(15,60));r.raise_for_status()
    tmp=f.with_suffix(f.suffix+'.partial');h=hashlib.sha256();size=0
    with tmp.open('wb') as out:
        for chunk in r.iter_content(2**20):out.write(chunk);h.update(chunk);size+=len(chunk)
    if name.endswith('.pdf'):assert tmp.read_bytes().startswith(b'%PDF')
    if name.endswith('.json'):json.loads(tmp.read_text())
    tmp.rename(f)
    receipt.write_text(json.dumps(dict(url=r.url,status=r.status_code,bytes=size,sha256=h.hexdigest(),
        retrieved_utc=datetime.now(timezone.utc).isoformat()),indent=2))
    print('Acquired',name,size)

