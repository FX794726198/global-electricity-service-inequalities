#!/usr/bin/env python3
"""Verify every file listed in the distributed SHA-256 manifest."""
from pathlib import Path
import hashlib

root=Path(__file__).resolve().parent
count=0
for line in (root/'metadata/SHA256SUMS').read_text().splitlines():
    digest,relative=line.split('  ',1)
    path=root/relative
    if not path.is_file():raise SystemExit(f'Missing file: {relative}')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise SystemExit(f'Checksum mismatch: {relative}')
    count+=1
print(f'PASS: {count} file checksums verified.')
