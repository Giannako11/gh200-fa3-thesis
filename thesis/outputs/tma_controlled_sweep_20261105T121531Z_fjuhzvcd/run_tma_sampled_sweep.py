#!/usr/bin/env python3
"""Universal warmed GH200 sweep: HBM, pinned, registered, and malloc_bound.

Defaults: 10 warmup launches/kernel, 30 samples/kernel, 580 groups, 1740 medians.
Run in fa3. Automatic PNG plots, checkpoints, resume, and diagnostic archive.
"""
import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess as sp
import sys
import tarfile
import tempfile
import time
from zoneinfo import ZoneInfo

from run_malloc_bound_sweep import bound_source, configurations

SOURCES = ['hbm', 'pinned', 'registered', 'malloc_bound']
KERNELS = ['loads_direct', 'loads_shared', 'tma_shared']
PHASES = ['before', 'before_samples', 'after', 'after_verification']
FIELDS = ['input_bytes','tile_rows','tile_cols','tile_bytes','source','kernel','gpu_threads',
          'first_sample_position','kernel_ms','effective_input_gb_s','status','cpu_node',
          'mbind_success','cpu_residency_verified']
for _phase in PHASES:
    FIELDS += [key+'_'+_phase for key in ['pages','on_cpu','unknown','query_errno']]
FIELDS += ['warmup_launches','sample_count','timing_statistic','min_ms','p25_ms','p75_ms',
           'p95_ms','max_ms','mean_ms','stddev_ms','cv','first_sample_ms','message']
RAW_FIELDS = ['input_bytes','tile_rows','tile_cols','source','kernel','sample_index',
              'position','kernel_ms','effective_input_gb_s']


def quantile(values, q):
    values = sorted(values)
    position = q*(len(values)-1)
    low = int(position)
    return values[low]+(values[min(low+1,len(values)-1)]-values[low])*(position-low)


def sampled_source(original, measurement):
    source = bound_source(original)
    old = '            if (m.host) { diag_policy(m.host); require_cpu_policy(m.host, node); }'
    if source.count(old) != 1: raise ValueError('Unexpected allocation policy hook')
    source = source.replace(old, '            if (m.host) diag_policy(m.host);\n'
                            '            if (m.source == "malloc_bound") require_cpu_policy(m.host, node);')
    old = '    Location before, after, after_verification;'
    if source.count(old) != 1: raise ValueError('Unexpected Row layout')
    source = source.replace(old, '    Location before, before_samples, after, after_verification;\n'
                            '    int warmup = 0;\n'
                            '    std::vector<double> samples;\n'
                            '    std::vector<int> positions;')
    start = source.index('constexpr const char* header =')
    end = source.index(';',start)+1
    source = source[:start]+'constexpr const char* header = '+json.dumps(','.join(FIELDS))+';'+source[end:]
    start = source.index('template<int H, int W>\nvoid measure(')
    end = source.index('// Το message',start)
    source = source[:start]+measurement+'\n\n'+source[end:]
    start = source.index('void save(',source.index('// Το message'))
    end = source.index('\nint main(',start)
    return source[:start]+source[end:]


def load_csv(path, fields):
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != fields: raise ValueError('CSV schema mismatch: '+str(path))
        rows = list(reader)
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError('Malformed CSV: '+str(path))
    return rows


def read_case(folder, config, warmup, sample_count, isolated=False):
    n,h,w,source = config
    rows = load_csv(folder/'rows.csv',FIELDS)
    raw = load_csv(folder/'samples.csv',RAW_FIELDS)
    if len(rows)!=3 or {r['kernel'] for r in rows} != set(KERNELS):
        raise ValueError('Missing/duplicate kernels: '+str(folder))
    by_kernel = {k:[] for k in KERNELS}
    seen = set()
    for r in raw:
        if (int(r['input_bytes']),int(r['tile_rows']),int(r['tile_cols']),r['source']) != config:
            raise ValueError('Raw sample configuration mismatch')
        key = r['kernel'],int(r['sample_index'])
        if key in seen or key[0] not in by_kernel or not 1<=key[1]<=sample_count or int(r['position']) not in (0,1,2):
            raise ValueError('Invalid/duplicate raw sample')
        seen.add(key)
        ms=float(r['kernel_ms'])
        if not math.isfinite(ms) or ms<=0 or not math.isclose(float(r['effective_input_gb_s']),n/(ms*1e6),rel_tol=1e-7):
            raise ValueError('Invalid raw time/rate')
        by_kernel[key[0]].append(r)
    for r in rows:
        if (int(r['input_bytes']),int(r['tile_rows']),int(r['tile_cols']),r['source']) != config:
            raise ValueError('Aggregate configuration mismatch')
        if int(r['tile_bytes'])!=h*w*4 or int(r['gpu_threads'])!=128:
            raise ValueError('Tile/thread mismatch')
        if r['status'] not in {'ok','error','not_run','device_fault','worker_error'}:
            raise ValueError('Unknown worker status')
        samples=sorted(by_kernel[r['kernel']],key=lambda x:int(x['sample_index']))
        if int(r['sample_count'])!=len(samples):raise ValueError('Raw/aggregate sample count mismatch')
        if r['status']!='ok':continue
        if len(samples)!=sample_count or int(r['warmup_launches'])!=warmup or r['timing_statistic']!='median':
            raise ValueError('Successful row has incomplete sampling/warmup')
        times=[float(x['kernel_ms']) for x in samples]
        expected={'kernel_ms':statistics.median(times),'min_ms':min(times),'max_ms':max(times),
                  'p25_ms':quantile(times,.25),'p75_ms':quantile(times,.75),'p95_ms':quantile(times,.95),
                  'mean_ms':statistics.mean(times),'stddev_ms':statistics.stdev(times),
                  'cv':statistics.stdev(times)/statistics.mean(times),'first_sample_ms':times[0]}
        expected['effective_input_gb_s']=n/(expected['kernel_ms']*1e6)
        for field,value in expected.items():
            if not math.isclose(float(r[field]),value,rel_tol=1e-7,abs_tol=1e-12):
                raise ValueError('Sample statistic mismatch: '+field)
        if int(r['first_sample_position']) != int(samples[0]['position']):
            raise ValueError('First sample position mismatch')
        positions=[int(x['position']) for x in samples]
        if isolated and len(set(positions))!=1:
            raise ValueError('Isolated kernel samples have mixed block positions')
        if not isolated and any(positions.count(p)!=sample_count//3 for p in range(3)):
            raise ValueError('Unbalanced kernel positions')
        if source=='hbm':
            if r['cpu_residency_verified']!='-1':raise ValueError('HBM residency marker mismatch')
        else:
            resident=True
            for phase in PHASES:
                pages,cpu,unknown,error=[int(r[k+'_'+phase]) for k in ['pages','on_cpu','unknown','query_errno']]
                if pages<=0 or not 0<=cpu<=pages or not 0<=unknown<=pages-cpu:
                    raise ValueError('Invalid page counts')
                resident &= pages==cpu and unknown==0 and error==0
            if int(r['cpu_residency_verified'])!=int(resident):raise ValueError('Placement flag mismatch')
        if source in ('registered','malloc_bound') and r['mbind_success']!='1':
            raise ValueError('Explicit range binding did not succeed')
    if all(r['status']=='ok' for r in rows):
        if isolated:
            if {int(r['first_sample_position']) for r in rows}!={0,1,2}:
                raise ValueError('Invalid isolated block ordering')
            return rows,raw
        for sample in range(1,sample_count+1):
            current=[r for r in raw if int(r['sample_index'])==sample]
            if len(current)!=3 or {int(r['position']) for r in current}!={0,1,2}:
                raise ValueError('Invalid per-round kernel ordering')
        # All six permutations must occur equally, not just balanced positions.
        orders=[]
        for sample in range(1,sample_count+1):
            current=sorted((r for r in raw if int(r['sample_index'])==sample),key=lambda r:int(r['position']))
            orders.append(tuple(r['kernel'] for r in current))
        if len(set(orders))!=6 or any(orders.count(p)!=sample_count//6 for p in set(orders)):
            raise ValueError('Six-permutation order protocol is not balanced')
    return rows,raw


def write_csv(path, fields, rows):
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    temp.replace(path)


def checkpoint(out, records, expected_groups, samples):
    rows=[r for group in records.values() for r in group]
    write_csv(out/'results.csv',FIELDS,rows)
    good=[r for r in rows if r['status']=='ok']
    hosts=[r for r in good if r['source']!='hbm']
    placed=sum(r['cpu_residency_verified']=='1' for r in hosts)
    passed=len(rows)==3*expected_groups and len(good)==len(rows) and placed==len(hosts)
    noisy=sum(float(r['cv'])>.10 for r in good)
    first_slow=sum(float(r['first_sample_ms'])>1.5*float(r['kernel_ms']) for r in good)
    lines=['# Warmed and sampled TMA sweep','',f'- Completed allocation groups: {len(records)} / {expected_groups}',
           f'- Expected aggregate measurements: {3*expected_groups}',f'- Recorded measurements: {len(rows)}',
           f'- Correct outputs (pre/post sample-loop validation): {len(good)}',
           f'- Failed or not run: {len(rows)-len(good)}',f'- Missing measurements: {3*expected_groups-len(rows)}',
           f'- Expected raw timing samples: {3*expected_groups*samples}',
           f'- Recorded raw timing samples: {sum(int(r["sample_count"]) for r in rows)}',
           f'- CPU placement verified at all four snapshots: {placed} / {len(hosts)}',
           f'- Rows with within-run CV > 10%: {noisy}',f'- Rows with first sample > 1.5 x median: {first_slow}',
           f'- Overall correctness/placement: {"PASS" if passed else "INCOMPLETE_OR_FAILED"}','',
           '| Source | Correct / recorded | CPU placement verified / correct host rows |','|---|---:|---:|']
    for source in SOURCES:
        rr=[r for r in rows if r['source']==source];gg=[r for r in rr if r['status']=='ok']
        host=[r for r in gg if source!='hbm']
        lines.append(f'| {source} | {len(gg)} / {len(rr)} | {sum(r["cpu_residency_verified"]=="1" for r in host)} / {len(host)} |')
    lines += ['', 'kernel_ms is the median; raw samples and p25/p75/p95, min/max, mean, sample SD and CV are retained.',
              'Each sample round runs all three kernels once. All six order permutations occur equally.',
              'Two untimed correctness launches per kernel bracket warmup/sampling; warmup launches are additional.',
              'Placement snapshots: initial, after warmup, after samples before verification, after final verification.',
              'Samples reuse the same input/output buffers after warmup; caches are NOT flushed. This is a warmed-buffer protocol.',
              'Small inputs can be cache-resident. Effective input GB/s is not physical DRAM/NVLink bandwidth.',
              'CUDA events include the complete kernel, including output stores to HBM.',
              'IQR is within-run sample dispersion, not a confidence interval or independent-run replication.',
              'CV and first-sample thresholds are descriptive diagnostics; correctness PASS does not assert statistical timing quality.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    return passed,rows


def rebuild_samples(out, records):
    """Repair interrupted appends from the authoritative, validated case files."""
    temp=out/'samples.csv.tmp'
    with temp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=RAW_FIELDS);writer.writeheader()
        for n,h,w,source in records:
            folder=out/'cases'/f'{n}_{h}x{w}_{source}'
            writer.writerows(load_csv(folder/'samples.csv',RAW_FIELDS))
    temp.replace(out/'samples.csv')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',action='store_true')
    parser.add_argument('--warmup',type=int,default=10)
    parser.add_argument('--samples',type=int,default=30,help='At least 6, divisible by 6 for balanced ordering')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--skip-plots',action='store_true')
    parser.add_argument('--smoke',action='store_true',help='Only 1 MiB / 32x32, all four allocations')
    args=parser.parse_args()
    if args.warmup<1 or args.samples<6 or args.samples%6:parser.error('Require --warmup >=1 and --samples >=6 divisible by 6')
    pairs=[(1<<20,32,32)] if args.smoke else configurations()
    configs=[(*pair,source) for pair in pairs for source in SOURCES]
    print(f'Plan: {len(configs)} groups, {3*len(configs)} aggregate measurements; '
          f'{args.warmup} warmups + {args.samples} samples/kernel; {3*len(configs)*args.samples} raw timing samples',flush=True)
    if args.plan:return 0
    repo=Path(__file__).resolve().parents[2];outputs=repo/'thesis/outputs'
    original=repo/'thesis/microbenchmarks/tma_sweep/tma_sweep.cu'
    template=repo/'thesis/microbenchmarks/tma_sweep/tma_sampled_measure.inc'
    plot=repo/'thesis/scripts/plots/tma_plots_first.py'
    generated=sampled_source(original.read_text(),template.read_text())
    if not args.skip_plots:sp.run([sys.executable,str(plot),'--help'],stdout=sp.DEVNULL,check=True)
    outputs.mkdir(exist_ok=True)
    lock=(outputs/'.tma_sweep.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    apps=sp.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    if any(x.strip().isdigit() for x in apps.splitlines()):raise RuntimeError('GPU compute process detected')
    candidates=[Path(os.environ.get('CONDA_PREFIX','/nonexistent'))/'cuda-12.8.1/bin/nvcc',Path('/usr/local/cuda-12.8/bin/nvcc')]
    nvcc=next((str(p) for p in candidates if p.is_file()),shutil.which('nvcc'))
    if not nvcc:raise RuntimeError('Activate fa3: nvcc not found')
    manifest=dict(protocol='warmed_balanced_samples_v1',configs=configs,warmup=args.warmup,samples=args.samples,
                  cuda_sha256=hashlib.sha256(generated.encode()).hexdigest(),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  compiler=sp.check_output([nvcc,'--version'],text=True),
                  gpu=sp.check_output(['nvidia-smi','--query-gpu=uuid,driver_version','--format=csv,noheader'],text=True),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  cuda_module_loading=os.environ.get('CUDA_MODULE_LOADING','driver default'))
    manifest=json.loads(json.dumps(manifest))
    if args.resume:
        out=args.resume.resolve()
        if out.parent!=outputs.resolve() or not out.name.startswith('tma_sampled_sweep_'):raise ValueError('Invalid sampled resume folder')
        if json.loads((out/'manifest.json').read_text())!=manifest:raise ValueError('Resume code, sampling protocol, compiler or GPU mismatch')
    else:
        stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        out=Path(tempfile.mkdtemp(prefix=f'tma_sampled_sweep_{stamp}_',dir=outputs))
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        (out/'sampled.cu').write_text(generated)
        for path in [original,template,Path(__file__),Path(__file__).with_name('run_malloc_bound_sweep.py'),Path(__file__).with_name('run_malloc_diagnostics.py')]:
            shutil.copyfile(path,out/path.name)
        (out/'base_commit.txt').write_text(sp.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True))
    if hashlib.sha256((out/'sampled.cu').read_bytes()).hexdigest()!=manifest['cuda_sha256']:raise ValueError('Archived source mismatch')
    print(f'Outputs: {out}',flush=True)
    records={};active=None;exit_code=1;started=time.monotonic()
    def interrupted(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGHUP,interrupted)
    try:
        # Rebuild the raw aggregate from authoritative completed cases on resume,
        # so a crash between append/checkpoint operations cannot create duplicates.
        raw_temp=out/'samples.csv.tmp'
        with raw_temp.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=RAW_FIELDS);writer.writeheader()
            for config in configs:
                n,h,w,source=config;folder=out/'cases'/f'{n}_{h}x{w}_{source}'
                if not (folder/'done.json').exists():continue
                info=json.loads((folder/'done.json').read_text())
                for name in ['rows.csv','samples.csv']:
                    if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=info[name]:raise ValueError('Checkpoint content mismatch')
                rows,raw=read_case(folder,config,args.warmup,args.samples)
                records[tuple(config)]=rows;writer.writerows(raw)
        raw_temp.replace(out/'samples.csv')
        checkpoint(out,records,len(configs),args.samples)
        with (out/'environment.txt').open('a') as log:
            for cmd in [['uname','-a'],['nvidia-smi','-q'],['numactl','--hardware'],['numactl','--show']]:
                log.write('\nCOMMAND '+repr(cmd)+'\n');log.flush();sp.run(cmd,stdout=log,stderr=sp.STDOUT,check=False)
            for name in ['/proc/sys/kernel/numa_balancing','/sys/module/nvidia_uvm/parameters/uvm_perf_access_counter_migration_enable']:
                log.write('\n'+name+'\n')
                try:log.write(Path(name).read_text())
                except OSError as e:log.write(str(e)+'\n')
        binary=out/'sampled';toolkit=Path(nvcc).resolve().parents[1]
        print('Compiling sampled CUDA benchmark...',flush=True)
        with (out/'build.log').open('w') as log:
            sp.run([nvcc,'-std=c++17','-O3','-lineinfo','-arch=sm_90a',str(out/'sampled.cu'),
                    '-L'+str(toolkit/'lib64/stubs'),'-lcuda','-o',str(binary)],stdout=log,stderr=sp.STDOUT,check=True)
        env=dict(os.environ,TMA_SAMPLE_WARMUP=str(args.warmup),TMA_SAMPLE_COUNT=str(args.samples))
        for index,config in enumerate(configs):
            n,h,w,source=config;folder=out/'cases'/f'{n}_{h}x{w}_{source}'
            if tuple(config) not in records:
                folder.mkdir(parents=True,exist_ok=True)
                cmd=[str(binary),str(n),str(h),str(w),source,'128',str(index%3),str(folder/'rows.csv')]
                (folder/'command.json').write_text(json.dumps(dict(argv=cmd,warmup=args.warmup,samples=args.samples)))
                with (folder/'worker.log').open('w') as log:
                    active=sp.Popen(cmd,env=env,stdout=log,stderr=sp.STDOUT,start_new_session=True)
                    try:rc=active.wait(timeout=900)
                    except sp.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait();rc=124
                    active=None
                if rc in (3,124) or rc<0:raise RuntimeError(f'GPU fault/timeout/signal in {folder.name}; stopping')
                rows,raw=read_case(folder,config,args.warmup,args.samples)
                done=dict(returncode=rc,**{name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ['rows.csv','samples.csv']})
                (folder/'done.json').write_text(json.dumps(done))
                records[tuple(config)]=rows
                with (out/'samples.csv').open('a',newline='') as f:csv.DictWriter(f,fieldnames=RAW_FIELDS).writerows(raw)
                checkpoint(out,records,len(configs),args.samples)
            count=len(records);filled=count*30//len(configs);elapsed=int(time.monotonic()-started)
            print(f'Progress: [{"#"*filled}{"-"*(30-filled)}] {100*count//len(configs):3d}% '
                  f'({count}/{len(configs)} groups) {n} bytes {h}x{w} {source} elapsed={elapsed}s',flush=True)
        passed,rows=checkpoint(out,records,len(configs),args.samples);exit_code=0 if passed else 1
        if not args.skip_plots and any(r['status']=='ok' for r in rows):
            date=dt.datetime.now(ZoneInfo('Europe/Athens')).strftime('%d_%m_%y')
            dest=outputs/'plots'/f'tma_sampled_sweep_plots_{date}'
            cmd=[sys.executable,str(plot),str(out/'results.csv'),'--sampled','--output-dir',str(dest)]
            with (out/'plots.log').open('w') as log:
                proc=sp.run(cmd,stdout=sp.PIPE,stderr=sp.STDOUT,text=True)
                log.write(proc.stdout);print(proc.stdout,flush=True)
            (out/'plots-status.json').write_text(json.dumps(dict(returncode=proc.returncode)))
            if proc.returncode:exit_code=2
    finally:
        if active is not None and active.poll() is None:os.killpg(active.pid,signal.SIGKILL);active.wait()
        rebuild_samples(out,records)
        checkpoint(out,records,len(configs),args.samples)
        (out/'status.json').write_text(json.dumps(dict(exit_code=exit_code)))
        archive=out.with_suffix('.tar.gz')
        with tarfile.open(archive,'w:gz') as tar:
            for path in out.rglob('*'):
                if path.is_file() and path!=out/'sampled':tar.add(path,arcname=str(Path(out.name)/path.relative_to(out)))
        print((out/'summary.md').read_text(),flush=True)
        print(f'Outputs: {out}\nSEND_THIS_FILE: {archive}',flush=True)
    return exit_code


if __name__=='__main__':
    try:sys.exit(main())
    except KeyboardInterrupt:sys.exit(130)
