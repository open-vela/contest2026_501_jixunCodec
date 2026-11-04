#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fetch four checksum-pinned source archives into a local cache."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--cache',required=True,type=Path);a=p.parse_args()
repo=Path(__file__).resolve().parents[1]
deps=json.loads((repo/'board/kickpi-k7/sources.lock.json').read_text())['dependencies']
a.cache.mkdir(parents=True,exist_ok=True)
for name,d in deps.items():
    target=a.cache/d['archive']
    if not target.exists():
        tmp=target.with_name(target.name+'.download')
        with urllib.request.urlopen(d['url'],timeout=120) as source,tmp.open('wb') as out:
            while True:
                data=source.read(1048576)
                if not data:break
                out.write(data)
        assert hashlib.sha256(tmp.read_bytes()).hexdigest()==d['sha256'],name
        tmp.replace(target)
    assert hashlib.sha256(target.read_bytes()).hexdigest()==d['sha256'],name
    print('SOURCE_ARCHIVE_VERIFIED',name)
