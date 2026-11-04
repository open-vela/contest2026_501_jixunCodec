#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare a separate RK3576 SDK from pinned sources; never flash hardware."""
import argparse,hashlib,json,os,shutil,subprocess,tarfile,tempfile,zipfile
from pathlib import Path

def sha(path):
    if not path.is_file():return None
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def git_apply(root,patch,check=False,reverse=False):
    cmd=['git','apply','--whitespace=nowarn']
    if check:cmd.append('--check')
    if reverse:cmd.append('--reverse')
    return subprocess.run(cmd+[str(patch)],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.PIPE)

def extract(archive,target):
    target.mkdir(parents=True,exist_ok=False)
    if archive.suffix=='.zip':
        with zipfile.ZipFile(archive) as z:
            for m in z.infolist():
                bits=Path(m.filename).parts[1:]
                if not bits or m.is_dir():continue
                assert '..' not in bits and not Path(*bits).is_absolute()
                f=target.joinpath(*bits);f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(z.read(m))
    else:
        with tarfile.open(archive) as t:
            for m in t:
                bits=Path(m.name).parts[1:]
                if not bits or not m.isfile():continue
                assert '..' not in bits and not Path(*bits).is_absolute()
                f=target.joinpath(*bits);f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(t.extractfile(m).read())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('workspace',type=Path);p.add_argument('--role',choices=['master','slave'],required=True)
    p.add_argument('--cache',required=True,type=Path);p.add_argument('--apply',action='store_true')
    a=p.parse_args();work=a.workspace.resolve();cache=a.cache.resolve()
    repo=Path(__file__).resolve().parents[1];board=repo/'board/kickpi-k7'
    lock=json.loads((board/'platform.lock.json').read_text())
    deps=json.loads((board/'sources.lock.json').read_text())['dependencies']
    assert (work/'nuttx/tools/build.sh').is_file(),'RK3576 NuttX source missing'
    assert (work/'apps/Application.mk').is_file(),'Pinned apps source missing'
    assert (work/'contest2026_062_PharosTech').is_dir(),'Pinned RK3576 BSP missing'
    assert repo!=work and repo not in work.parents,'Use an SDK workspace outside the source repository'
    # Complete preflight first. Unknown source content fails before any SDK write.
    plans=[]
    for name,c in lock['components'].items():
        if name=='microadb':continue
        root=work/c['directory'];patch=board/'patches'/c['patch']
        assert sha(patch)==c['patch_sha256'],str(patch)
        states=[]
        for f in c['files']:
            current=sha(root/f['path'])
            if current==f['after_sha256']:states.append('after')
            elif current==f['before_sha256']:states.append('before')
            else:raise RuntimeError('Unknown source content: '+str(root/f['path']))
        assert len(set(states))==1,'Partially applied component: '+name
        if states[0]=='before':
            result=git_apply(root,patch,check=True)
            assert not result.returncode,result.stderr.decode(errors='replace')
            plans.append((root,patch,c))
    for dep in deps.values():
        assert sha(cache/dep['archive'])==dep['sha256'],'Missing/wrong dependency archive: '+dep['archive']
    link_sources={
        'vendor/rockchip/boards':work/'contest2026_062_PharosTech/boards',
        'vendor/rockchip/chips':work/'contest2026_062_PharosTech/chips',
        'apps/system/nbootctl':work/'contest2026_062_PharosTech/app/nbootctl',
        'apps/system/audioctl':work/'contest2026_062_PharosTech/app/audioctl',
        'apps/examples/jixun_k7':repo/'app/jixun-k7',
        'apps/vendor':work/'vendor','apps/external':work/'external',
        'build.sh':work/'nuttx/tools/build.sh',
        'configs/jixun_k7_ui/Make.defs':work/'contest2026_062_PharosTech/boards/rk3576/kickpi-k7/scripts/Make.defs',
    }
    # Other BSP apps are needed by Kconfig even when the target disables them.
    import xml.etree.ElementTree as ET
    manifest=work/'contest2026_062_PharosTech/contest2026_062_PharosTech.xml'
    for link in ET.parse(manifest).getroot().findall('./project/linkfile'):
        link_sources.setdefault(link.attrib['dest'],work/'contest2026_062_PharosTech'/link.attrib['src'])
    for dest,source in link_sources.items():
        path=work/dest
        assert not path.exists() and not path.is_symlink() or path.is_symlink() and path.resolve()==source.resolve(),'Occupied SDK path: '+str(path)
    profile=work/'configs/jixun_k7_ui/defconfig'
    profile_input_sha=sha(board/'configs/openvela/defconfig')
    input_lf=(board/'configs/openvela/defconfig').read_bytes().replace(b'\r\n',b'\n')
    input_variants={profile_input_sha,hashlib.sha256(input_lf).hexdigest(),hashlib.sha256(input_lf.replace(b'\n',b'\r\n')).hexdigest()}
    profile_owned=profile.parent/'.jixun-profile-owned.json'
    owned=json.loads(profile_owned.read_text()) if profile_owned.is_file() else {}
    assert not profile.exists() or sha(profile) in input_variants or (owned.get('source_defconfig_sha256') in input_variants and owned.get('profile_sha256')==sha(profile)),'Unrecognized target profile'
    rcs=work/'contest2026_062_PharosTech/boards/rk3576/kickpi-k7/src/etc/init.d/rcS'
    assert sha(rcs) in lock['recognized_startup_sha256'],'Unknown startup script'
    # Validate dependency patches on temporary copies as part of the dry run.
    dep_roots={}
    temp=tempfile.TemporaryDirectory(prefix='jixun-k7-check-')
    try:
        for name,dep in deps.items():
            root=work/dep['directory'];dep_roots[name]=root
            probe=Path(temp.name)/name
            if root.exists():
                shutil.copytree(root,probe,symlinks=True,ignore=shutil.ignore_patterns('.git','*.o','*.a','*.pyc','__pycache__'))
            else:extract(cache/dep['archive'],probe)
            if name=='libuv' and sha(probe/'src/unix/nuttx.c')!=dep['patched_probe_sha256']:
                for patch in [work/'apps/system/libuv/0001-libuv-port-for-nuttx.patch',board/'patches/libuv-tls.patch']:
                    if git_apply(probe,patch,check=True,reverse=True).returncode:
                        result=git_apply(probe,patch);assert not result.returncode,result.stderr.decode(errors='replace')
                assert sha(probe/'src/unix/nuttx.c')==dep['patched_probe_sha256']
            if name=='microadb':
                c=lock['components']['microadb'];f=c['files'][0];patch=board/'patches'/c['patch']
                assert sha(patch)==c['patch_sha256']
                assert sha(probe/f['path']) in [f['before_sha256'],f['after_sha256']]
                if sha(probe/f['path'])==f['before_sha256']:
                    result=git_apply(probe,patch);assert not result.returncode,result.stderr.decode(errors='replace')
                assert sha(probe/f['path'])==f['after_sha256']
        print('PREFLIGHT_PASS source patches=%d dependency archives=%d'%(len(plans),len(deps)))
        if not a.apply:
            print('No SDK changes. Add --apply to prepare the independent workspace.');return
        for root,patch,c in plans:
            result=git_apply(root,patch);assert not result.returncode,result.stderr.decode(errors='replace')
            for f in c['files']:assert sha(root/f['path'])==f['after_sha256']
        for name,dep in deps.items():
            root=dep_roots[name]
            if not root.exists():extract(cache/dep['archive'],root)
            if name=='libuv' and sha(root/'src/unix/nuttx.c')!=dep['patched_probe_sha256']:
                for patch in [work/'apps/system/libuv/0001-libuv-port-for-nuttx.patch',board/'patches/libuv-tls.patch']:
                    if git_apply(root,patch,check=True,reverse=True).returncode:
                        result=git_apply(root,patch);assert not result.returncode,result.stderr.decode(errors='replace')
            if name=='microadb':
                c=lock['components']['microadb'];f=c['files'][0]
                if sha(root/f['path'])==f['before_sha256']:
                    result=git_apply(root,board/'patches'/c['patch']);assert not result.returncode,result.stderr.decode(errors='replace')
            if dep.get('sdk_archive'):
                dest=work/dep['sdk_archive']
                shutil.copy2(cache/dep['archive'],dest)
                # Make must not delete an already checked source tree to unpack it.
                os.utime(root,None)
                (root/dep.get('stamp','.patch')).touch()
        for dest,source in link_sources.items():
            target=work/dest;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.is_symlink():target.symlink_to(os.path.relpath(source,target.parent))
        # Preserve the owned minimal defconfig emitted by savedefconfig. Copying
        # the input back on every role switch forces an unnecessary full rebuild.
        if not profile.exists():shutil.copy2(board/'configs/openvela/defconfig',profile)
        role='a' if a.role=='master' else 'b'
        data=(board/'rcS.base').read_bytes()+('\n#ifdef CONFIG_EXAMPLES_JIXUN_K7\nk7_pair node '+role+'\n#endif\n').encode()+b'\n#ifdef CONFIG_EXAMPLES_JIXUN_K7_UI\nk7_ui &\n#endif\n'
        assert hashlib.sha256(data).hexdigest()==lock['role_startup_sha256'][a.role]
        rcs.write_bytes(data)
        print('K7_WORKSPACE_READY role='+a.role)
    finally:temp.cleanup()

if __name__=='__main__':main()
