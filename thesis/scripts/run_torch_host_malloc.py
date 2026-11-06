#!/usr/bin/env python3
"""Run isolated stock/OFF/ON GH200 checks with process-local opt-in only."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess as sp
import sys
import tempfile
import time
from torch_host_malloc_common import (PACKAGE, PATCH, REPO, TORCH_COMMIT, capture,
    default_root, digest, progress, runtime_environment, write_json)

CASES = [('stock_standard','stock','standard',False),('stock_streams','stock','streams',False),
         ('off_standard','host','standard',False),('off_streams','host','streams',False),
         ('off_frozen','host','frozen_off',False),('host_standard','host','standard',True),
         ('host_streams','host','streams',True),('host_uncached','host','uncached',True),
         ('host_oom','host','oom',True),('host_graph_reject','host','graph_reject',True),
         ('host_fraction_reject','host','fraction_reject',True),
         ('host_frozen','host','frozen_on',True),('host_bad_flag','host','bad_flag',True),
         ('host_async_reject','host','backend_async',True),('host_expandable_reject','host','expandable',True),
         ('host_reconfigure_reject','host','reconfigure_reject',True)]


def validate_log(path, maximum):
    live={}
    allocated=freed=0
    config=False
    for line in path.read_text().splitlines():
        row=json.loads(line)
        if row['event']=='config':
            assert not config and row['limit']==maximum and row['node']==0
            config=True
        elif row['event']=='alloc':
            assert row['ptr'] not in live and row['node']==0
            live[row['ptr']]=row['charge'];allocated+=1
            assert sum(live.values())==row['live']<=maximum
        elif row['event']=='free':
            assert row['ptr'] in live and row['node']==0
            live.pop(row['ptr']);freed+=1
            assert sum(live.values())==row['live']
    assert config
    return dict(allocations=allocated,frees=freed,retained_segments=len(live),retained_bytes=sum(live.values()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=default_root())
    p.add_argument('--plan',action='store_true')
    p.add_argument('--case',choices=[x[0] for x in CASES],action='append')
    args=p.parse_args()
    cases=[x for x in CASES if not args.case or x[0] in args.case]
    root=args.root.expanduser().resolve()
    print('Build root:',root)
    for name,mode,case,host in cases:
        print(f'  {name}: {mode} build, feature={"ON" if host else "OFF"}, check={case}')
    if args.plan:
        return 0
    manifest=json.loads((root/'manifest.json').read_text())
    if json.loads((root/'build_status.json').read_text())['state']!='complete':
        raise RuntimeError('Build did not complete')
    if manifest['upstream']!=TORCH_COMMIT or manifest['patch_sha256']!=digest(PATCH) or manifest['helper_sha256']!=digest(PACKAGE/'ThesisHostMalloc.h'):
        raise RuntimeError('Build/patch provenance mismatch. Rebuild in a fresh --root')
    if os.environ.get('LD_PRELOAD'):
        raise RuntimeError('Unset LD_PRELOAD for this isolated test')
    for mode,wheel in manifest['wheels'].items():
        py=Path(wheel['python']).resolve()
        expected=(root/('stock_env' if mode=='stock' else 'fa3_host_malloc')).resolve()
        if py.parent.parent!=expected or expected==Path(manifest['original_prefix']).resolve():
            raise RuntimeError('Refusing to execute outside the isolated test prefixes')
        if digest(wheel['path'])!=wheel['sha256']:
            raise RuntimeError('Built wheel content mismatch')
    for cmd in ['numactl','nvidia-smi']:
        if not shutil.which(cmd):raise RuntimeError(cmd+' not found')
    outputs=REPO/'thesis/outputs'
    outputs.mkdir(parents=True,exist_ok=True)
    lock=(outputs/'.tma_sweep.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    apps=capture(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'])
    if any(s.strip().isdigit() for s in apps.splitlines()):
        raise RuntimeError('GPU compute process present; run on an idle GPU')
    env=runtime_environment(Path(manifest['cuda_home']))
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=Path(tempfile.mkdtemp(prefix='torch_host_malloc_'+stamp+'_',dir=outputs))
    worker=REPO/'thesis/tests/test_torch_host_malloc.py'
    write_json(out/'manifest.json',dict(build=manifest,cases=cases,files={str(x.relative_to(REPO)):digest(x)
        for x in [Path(__file__),worker,PATCH,PACKAGE/'ThesisHostMalloc.h']}))
    shutil.copyfile(PATCH,out/PATCH.name)
    shutil.copyfile(PACKAGE/'ThesisHostMalloc.h',out/'ThesisHostMalloc.h')
    print('Outputs:',out,flush=True)
    completed=[];active=None;telemetry=None;telemetry_file=None;exit_code=1;error=None
    maximum=256*(1<<20)
    def interrupt(signum,frame):raise KeyboardInterrupt
    previous={s:signal.signal(s,interrupt) for s in [signal.SIGTERM,signal.SIGHUP]}
    try:
        with (out/'environment.txt').open('w') as f:
            for command in [['nvidia-smi','-q'],['numactl','--hardware'],['uname','-a']]:
                f.write('\nCOMMAND '+repr(command)+'\n');f.flush()
                sp.run(command,stdout=f,stderr=sp.STDOUT,check=True)
        telemetry_file=(out/'telemetry.csv').open('w')
        telemetry=sp.Popen(['nvidia-smi','--query-gpu=timestamp,uuid,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used',
                           '--format=csv','--loop=1'],stdout=telemetry_file,stderr=sp.STDOUT)
        progress(0,len(cases),'start isolated checks')
        for index,(name,mode,case,host) in enumerate(cases,1):
            folder=out/name;folder.mkdir()
            case_env=dict(env,THESIS_HOST_MALLOC='1' if host else '0',
                THESIS_HOST_MALLOC_MAX_BYTES=str(maximum),THESIS_HOST_MALLOC_LOG=str(folder/'allocations.jsonl'),
                PYTORCH_CUDA_ALLOC_CONF='backend:native,expandable_segments:False')
            prefix=Path(manifest['wheels'][mode]['python']).parent.parent
            case_env.update(CONDA_PREFIX=str(prefix),CONDA_DEFAULT_ENV=str(prefix),
                PATH=str(prefix/'bin')+os.pathsep+env['PATH'],
                LD_LIBRARY_PATH=str(prefix/'lib')+os.pathsep+env['LD_LIBRARY_PATH'])
            if case=='uncached':case_env['PYTORCH_NO_CUDA_MEMORY_CACHING']='1'
            if case=='bad_flag':case_env['THESIS_HOST_MALLOC']='invalid'
            if case=='backend_async':case_env['PYTORCH_CUDA_ALLOC_CONF']='backend:cudaMallocAsync'
            if case=='expandable':case_env['PYTORCH_CUDA_ALLOC_CONF']='backend:native,expandable_segments:True'
            cmd=['numactl','--cpunodebind=0','--membind=0',manifest['wheels'][mode]['python'],str(worker),
                 '--case',case,'--output',str(folder/'result.json')]+(['--host'] if host else [])
            write_json(folder/'command.json',dict(argv=cmd,env={k:v for k,v in case_env.items()
                if k.startswith(('THESIS_HOST_MALLOC','PYTORCH_'))}))
            with (folder/'worker.log').open('w') as log:
                active=sp.Popen(cmd,env=case_env,cwd=root,stdout=log,stderr=sp.STDOUT,start_new_session=True)
                rc=active.wait(timeout=300);active=None
            if rc or not (folder/'result.json').is_file():
                raise RuntimeError(f'{name} failed (exit {rc}); inspect {folder}/worker.log')
            result=json.loads((folder/'result.json').read_text())
            if result['status']!='PASS' or result['torch_commit']!=TORCH_COMMIT:
                raise RuntimeError('Invalid worker result: '+name)
            if (folder/'allocations.jsonl').exists():
                if not host:raise RuntimeError('Feature-OFF unexpectedly used host backend')
                result['allocation_audit']=validate_log(folder/'allocations.jsonl',maximum)
            elif host and case not in ['bad_flag','backend_async','expandable']:
                raise RuntimeError('Missing host provenance log: '+name)
            completed.append(dict(name=name,**result))
            write_json(out/'status.json',dict(exit_code=None,completed=completed,expected=len(cases)))
            progress(index,len(cases),name+' PASS')
        exit_code=0
    except BaseException as e:
        error=repr(e);exit_code=130 if isinstance(e,KeyboardInterrupt) else 1
        print('Run stopped:',error,flush=True)
    finally:
        if active and active.poll() is None:
            os.killpg(active.pid,signal.SIGTERM)
            try:active.wait(timeout=10)
            except sp.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
        if telemetry:
            telemetry.terminate();telemetry.wait(timeout=10)
        if telemetry_file:telemetry_file.close()
        for sig,handler in previous.items():signal.signal(sig,handler)
        write_json(out/'status.json',dict(exit_code=exit_code,completed=completed,expected=len(cases),error=error))
        lines=['# PyTorch host malloc checks','',f'- Result: {"PASS" if exit_code==0 else "INCOMPLETE_OR_FAILED"}',
               f'- Completed checks: {len(completed)} / {len(cases)}',
               '- Scope: single-GPU eager inference, native allocator, bounded host allocations.',
               '- CUDA tensor storage and logical allocator counters do not imply physical HBM placement.',
               '- Host-to-host CUDA tensor copies use a synchronized CPU copy; GPU arithmetic uses CUDA kernels.',
               '- Timings are diagnostic test durations, not performance benchmark results.',
               '- No claim is made about full models, graphs, IPC, distributed execution or third-party extensions.','']
        lines += [f'- {r["name"]}: {r["status"]} ({len(r["checks"])} assertions groups)' for r in completed]
        if error:lines+=['','Failure: '+error]
        (out/'summary.md').write_text('\n'.join(lines)+'\n')
        print('Outputs:',out,flush=True)
    return exit_code


if __name__=='__main__':
    sys.exit(main())
