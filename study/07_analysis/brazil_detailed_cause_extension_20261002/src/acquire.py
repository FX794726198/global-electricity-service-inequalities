"""Acquire public release assets and inspectable source code; never execute publisher code."""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import requests

P = Path(__file__).resolve().parents[1]
ROOT = P.parents[1]
REPO = 'vsvilela39-oss/projeto-aneel'
SHA = '06c3f7bc82d3583332e9377769d581e3150b7890'
for name in ['raw', 'metadata', 'data', 'figures']:
    (P/name).mkdir(parents=True, exist_ok=True)
release_path = ROOT/'07_analysis/brazil_cause_economy_20261002/metadata/raw_source_search_20261002/vsvilela39-oss_projeto-aneel_releases.json'
releases = json.loads(release_path.read_text())
release = next(x for x in releases if x['tag_name'] == 'v1.0-dados')
(P/'metadata/public_release.json').write_text(json.dumps(release, indent=2, ensure_ascii=False))
targets = [(a['browser_download_url'], 'raw/'+a['name'], a['size'], a.get('digest')) for a in release['assets']]
targets += [(f'https://raw.githubusercontent.com/{REPO}/{SHA}/{f}', 'raw/publisher/'+f, None, None)
            for f in ['README.md','src/processar_aneel_colab_github.py','config/fontes.json',
                      'docs/01_Arquitetura.md','docs/02_Modelo_Estrela.md','docs/03_Medidas_DAX.md']]

def fetch(item):
    url, rel, size, digest = item
    path = P/rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, timeout=(15, 60), stream=True) as r:
        r.raise_for_status()
        partial = path.with_suffix(path.suffix+'.partial')
        with partial.open('wb') as h:
            for chunk in r.iter_content(1024*1024):
                h.write(chunk)
        if size is not None:
            assert partial.stat().st_size == size, (rel, partial.stat().st_size, size)
        with partial.open('rb') as h:
            sha = hashlib.file_digest(h, 'sha256').hexdigest()
        if digest and digest.startswith('sha256:'):
            assert sha == digest.split(':',1)[1], rel
        partial.replace(path)
        out = dict(url=url, path=rel, bytes=path.stat().st_size, sha256=sha,
                   source_digest=digest, final_url=r.url, content_type=r.headers.get('Content-Type'),
                   accessed_utc=datetime.now(timezone.utc).isoformat())
        print(rel, path.stat().st_size, flush=True)
        return out

with ThreadPoolExecutor(3) as pool:
    files = list(pool.map(fetch, targets))
(P/'metadata/acquisition.json').write_text(json.dumps(dict(repository=REPO, commit=SHA,
    release='v1.0-dados', publisher_code_executed=False, acquired=files,
    source_role='Public secondary aggregate release from ANEEL; semantics pending source audit'),
    indent=2, ensure_ascii=False))
