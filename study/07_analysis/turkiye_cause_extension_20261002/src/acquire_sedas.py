"""Archive public SEDAS Table 1 Excel exports exposed by its public web page."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, required=True)
parser.add_argument('--month', type=int, required=True)
parser.add_argument('--city', default='-2')
args = parser.parse_args()
params = dict(city=args.city, district='-2', year=args.year, month=args.month, type=1)
url = 'https://sedas.com/Home/DownloadExcelTablo'
stamp = f'sedas_tablo1_{args.city}_{args.year}_{args.month:02d}'
receipt = dict(request_url=url, params=params, retrieved_utc=datetime.now(timezone.utc).isoformat(),
               landing_page='https://sedas.com/Tr/SupplyContinuityUrl/Tablo1',
               endpoint_provenance='Public Excel download JavaScript in archived sedas_table1.html')
try:
    response = requests.get(url, params=params, timeout=(20, 55))
    ext = '.xlsx' if response.content[:2] == b'PK' else '.response'
    output = ROOT / 'raw' / (stamp + ext)
    output.write_bytes(response.content)
    receipt.update(status=response.status_code, final_url=response.url,
                   content_type=response.headers.get('Content-Type'), bytes=len(response.content),
                   sha256=hashlib.sha256(response.content).hexdigest(), file=str(output.relative_to(ROOT)))
except requests.RequestException as exc:
    receipt.update(error=str(exc))
(ROOT / 'metadata' / (stamp + '.json')).write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt, indent=2))
