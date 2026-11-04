#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check the K7 source update locally. No hardware or remote writes."""
import hashlib,json,re,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
repo=Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
app=repo/'app/jixun-k7'
provenance=json.loads((app/'SOURCE_PROVENANCE.json').read_text())
for name,sha in provenance['application_files'].items():assert digest(app/name)==sha,name
for row in provenance['source_normalizations']:
    assert row['change'] in ['generator comment only; remove workstation paths','CRLF to LF only','generator comment and CRLF to LF only']
    assert digest(app/row['path'])==row['prepared_sha256']
lock=json.loads((repo/'board/kickpi-k7/platform.lock.json').read_text())
for component in lock['components'].values():
    assert digest(repo/'board/kickpi-k7/patches'/component['patch'])==component['patch_sha256']
source=json.loads((repo/'board/kickpi-k7/sources.lock.json').read_text())
assert digest(repo/'board/kickpi-k7/configs/openvela/defconfig')==source['configuration_input']['prepared_sha256']
manifest=ET.parse(repo/'contest2026_501_jixunCodec_k7.xml').getroot()
projects={p.attrib['path']:p.attrib for p in manifest.findall('project')}
for directory,name in [('nuttx','nuttx'),('apps','apps'),('contest2026_062_PharosTech','bsp'),('vendor','vendor'),('external','external'),('build','build')]:
    assert projects[directory]['revision']==source['source_components'][name]['resolved_commit'],directory
build=json.loads((repo/'docs/kickpi-k7/build-validation.json').read_text())
assert build['passed'] and all(build['roles'][role]['passed'] for role in ['master','slave'])
paths=[]
for name in ['app/jixun-k7','board/kickpi-k7','docs/kickpi-k7','host/k7']:
    paths.extend(p for p in (repo/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
paths.extend(repo/'scripts'/name for name in ['prepare_k7_workspace.py','build_k7.sh','fetch_k7_dependencies.py','check_k7_submission.py'])
bad=re.compile(rb'(?:[A-Za-z]:\\|/(?:home|Users)/|192[.]168[.][0-9]|-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----|\b(?:sk|tp)-[A-Za-z0-9]{20,})')
for path in paths:
    assert path.stat().st_size<20*1024*1024,'Oversized source asset: '+str(path)
    if path.suffix not in {'.png'}:
        assert not bad.search(path.read_bytes()),'Local path/credential material in '+str(path.relative_to(repo))
        assert path.suffix not in {'.bin','.elf','.zip','.gz','.o','.a','.wav'},str(path)
legacy=['app/jixun-codec/','board/esp32s3-box/','logs/']
changed=subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=repo,text=True).splitlines()
assert not any(name.startswith(tuple(legacy)) for name in changed),'Legacy ESP32/log files changed'
assert subprocess.run(['git','check-ignore','--quiet','deliverables/kickpi-k7/current/probe.bin'],cwd=repo).returncode==0
print('K7_SOURCE_PATCH_MANIFEST_BUILD_AND_SCOPE_CHECKS_PASS',len(paths))
