#!/usr/bin/env python3
"""Full GH200 sweep: page-aligned system allocation + per-range MPOL_BIND.

Run in fa3. Default: 145 configurations, 435 kernels, followed by seven PNGs.
Existing benchmark sources and results are never changed. See --help.
"""
import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess as sp
import sys
import tarfile
import tempfile
from zoneinfo import ZoneInfo

from run_malloc_diagnostics import instrument

SIZES = [2**p for p in range(12, 35)]
TILES = [(32,32), (32,128), (64,64), (128,32), (64,128), (128,64), (128,128)]
KERNELS = {'loads_direct', 'loads_shared', 'tma_shared'}


def configurations():
    return [(n,h,w) for n in SIZES for h,w in TILES if n >= h*w*4 and n % (h*w*4) == 0]


def bound_source(original):
    source = instrument(original).replace('"malloc"', '"malloc_bound"')
    def change(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError('Unsupported source layout: ' + repr(old))
        source = source.replace(old, new)
    change('std::string mode = std::getenv("MALLOC_DIAG_MODE") ? std::getenv("MALLOC_DIAG_MODE") : "plain";',
           'std::string mode = "rangebind";')
    change('    Location before, after;', '    Location before, after, after_verification;')
    change('stage("after_verification"); (void)locate(m.host, m.bytes, node);',
           'stage("after_verification"); row.after_verification = locate(m.host, m.bytes, node);')
    change('r.before.verified() && r.after.verified()',
           'r.before.verified() && r.after.verified() && r.after_verification.verified()')
    change('"pages_after,on_cpu_after,unknown_after,query_errno_after,message";',
           '"pages_after,on_cpu_after,unknown_after,query_errno_after,"\n'
           '    "pages_after_verification,on_cpu_after_verification,unknown_after_verification,query_errno_after_verification,message";')
    change('            << quoted(r.message) << \'\\n\';',
           '            << r.after_verification.pages << \',\' << r.after_verification.on_cpu << \',\'\n'
           '            << r.after_verification.unknown << \',\' << r.after_verification.error << \',\'\n'
           '            << quoted(r.message) << \'\\n\';')
    policy = r'''
void require_cpu_policy(void* address, unsigned node) {
    int mode = -1;
    unsigned long mask[16] = {};
    if (syscall(SYS_get_mempolicy, &mode, mask, 1024UL, address, MPOL_F_ADDR) != 0)
        throw std::runtime_error(std::string("get_mempolicy: ")+std::strerror(errno));
    if (mode != MPOL_BIND) throw std::runtime_error("Allocation is not MPOL_BIND");
    for (unsigned i=0; i<1024; ++i) {
        bool set = mask[i/(8*sizeof(unsigned long))] & (1UL << (i%(8*sizeof(unsigned long))));
        if (set != (i == node)) throw std::runtime_error("Allocation policy is not bound exclusively to CPU node");
    }
}
'''
    change('struct Location {', policy + '\nstruct Location {')
    change('            if (m.host) diag_policy(m.host);',
           '            if (m.host) { diag_policy(m.host); require_cpu_policy(m.host, node); }')
    return source


def read_case(path, config):
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    n,h,w = config
    if len(rows) != 3 or {r['kernel'] for r in rows} != KERNELS:
        raise ValueError('Incomplete/duplicate kernels in ' + str(path))
    successful_orders = [int(r['launch_order']) for r in rows if r['status']=='ok']
    if len(set(successful_orders)) != len(successful_orders) or any(o not in (0,1,2) for o in successful_orders):
        raise ValueError('Invalid launch order in ' + str(path))
    for r in rows:
        if r['status'] not in {'ok','error','not_run','device_fault','worker_error'}:
            raise ValueError('Unknown status in ' + str(path))
        if (int(r['input_bytes']),int(r['tile_rows']),int(r['tile_cols']),r['source']) != (n,h,w,'malloc_bound'):
            raise ValueError('Configuration mismatch in ' + str(path))
        if int(r['tile_bytes']) != h*w*4 or int(r['gpu_threads']) != 128:
            raise ValueError('Tile/thread mismatch in ' + str(path))
        if r['status']=='ok':
            ms = float(r['kernel_ms'])
            if not math.isfinite(ms) or ms <= 0 or not math.isclose(float(r['effective_input_gb_s']), n/(ms*1e6), rel_tol=1e-6):
                raise ValueError('Invalid timing/rate in ' + str(path))
            resident = True
            for phase in ['before','after','after_verification']:
                pages,cpu,unknown,error = [int(r[k+'_'+phase]) for k in ['pages','on_cpu','unknown','query_errno']]
                if pages <= 0 or not 0 <= cpu <= pages or not 0 <= unknown <= pages-cpu:
                    raise ValueError('Invalid page counts in ' + str(path))
                resident &= pages == cpu and unknown == 0 and error == 0
            if int(r['cpu_residency_verified']) != int(resident) or int(r['mbind_success']) != 1:
                raise ValueError('Invalid placement/binding flag in ' + str(path))
    return rows


def collect(out, configs):
    rows = []
    for config in configs:
        n,h,w = config
        folder = out/'cases'/f'{n}_{h}x{w}'
        if (folder/'done.json').exists():
            rows.extend(read_case(folder/'rows.csv', config))
    if rows:
        with (out/'results.csv.tmp').open('w',newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        (out/'results.csv.tmp').replace(out/'results.csv')
    expected = 3*len(configs)
    good = [r for r in rows if r['status']=='ok']
    verified = sum(r['cpu_residency_verified']=='1' for r in good)
    passed = len(rows)==expected and len(good)==expected and verified==expected
    lines = ['# Full malloc_bound sweep', '', f'- Expected measurements: {expected}',
             f'- Recorded measurements: {len(rows)}', f'- Correct outputs: {len(good)}',
             f'- Failed or not run: {len(rows)-len(good)}', f'- Missing measurements: {expected-len(rows)}',
             f'- CPU residency verified at all three checks: {verified} / {len(good)}',
             f'- Overall: {"PASS" if passed else "INCOMPLETE_OR_FAILED"}', '',
             'Mechanism: page-aligned posix_memalign + explicit per-buffer mbind(MPOL_BIND) to the CPU node.',
             'Effective allocation policy is queried and required to bind exclusively to that node.',
             'Source label: malloc_bound. No cudaHostRegister; the GPU uses the system pointer.',
             'Placement checks: before GPU, after GPU before CPU verification, and after verification.',
             'Checks are snapshots, not continuous residency tracing.',
             'Three GPU kernels per configuration; 128 threads; one launch each; no warmup or cache control.',
             'Time includes HBM output stores. Use repeated measurements for final performance claims.',
             'Do not replace plain malloc rows in the original sweep with these measurements.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    return passed, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',action='store_true')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--skip-plots',action='store_true',help='Collect data without requiring Matplotlib/NumPy')
    args = parser.parse_args()
    configs = configurations()
    print(f'Plan: {len(configs)} allocation groups, {3*len(configs)} kernel measurements; source=malloc_bound',flush=True)
    if args.plan: return 0
    repo = Path(__file__).resolve().parents[2]
    outputs = repo/'thesis/outputs'
    plot = repo/'thesis/scripts/plots/tma_plots_first.py'
    original = repo/'thesis/microbenchmarks/tma_sweep/tma_sweep.cu'
    generated = bound_source(original.read_text())
    if not args.skip_plots:
        sp.run([sys.executable,str(plot),'--help'],stdout=sp.DEVNULL,check=True)
    outputs.mkdir(exist_ok=True)
    lock = (outputs/'.tma_sweep.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    apps = sp.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    if any(line.strip().isdigit() for line in apps.splitlines()):
        raise RuntimeError('GPU compute process detected')
    candidates = [Path(os.environ.get('CONDA_PREFIX','/nonexistent'))/'cuda-12.8.1/bin/nvcc',Path('/usr/local/cuda-12.8/bin/nvcc')]
    nvcc = next((str(p) for p in candidates if p.is_file()),shutil.which('nvcc'))
    if not nvcc: raise RuntimeError('Activate fa3: CUDA nvcc not found')
    manifest = dict(configurations=configs,source='malloc_bound',threads=128,
                    generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    compiler=sp.check_output([nvcc,'--version'],text=True),
                    gpu=sp.check_output(['nvidia-smi','--query-gpu=uuid,driver_version','--format=csv,noheader'],text=True),
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    manifest = json.loads(json.dumps(manifest))
    if args.resume:
        out = args.resume.resolve()
        if out.parent != outputs.resolve() or not out.name.startswith('malloc_bound_sweep_'):
            raise ValueError('Resume must name a malloc_bound_sweep folder under thesis/outputs')
        if json.loads((out/'manifest.json').read_text()) != manifest:
            raise ValueError('Resume source, configuration, compiler or GPU mismatch')
    else:
        stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        out = Path(tempfile.mkdtemp(prefix=f'malloc_bound_sweep_{stamp}_',dir=outputs))
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        (out/'malloc_bound.cu').write_text(generated)
        shutil.copyfile(original,out/'original.cu')
        shutil.copyfile(__file__,out/'runner-used.py')
        shutil.copyfile(Path(__file__).with_name('run_malloc_diagnostics.py'),out/'diagnostic-helper-used.py')
        (out/'base_commit.txt').write_text(sp.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True))
    print(f'Outputs: {out}',flush=True)
    if hashlib.sha256((out/'malloc_bound.cu').read_bytes()).hexdigest() != manifest['generated_sha256']:
        raise ValueError('Archived CUDA source mismatch')
    active = None
    exit_code = 1
    def interrupted(signum,frame): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGHUP,interrupted)
    try:
        collect(out,configs)
        with (out/'environment.txt').open('a') as log:
            for cmd in [['uname','-a'],['nvidia-smi','-q'],['numactl','--hardware'],['numactl','--show']]:
                log.write('\nCOMMAND '+repr(cmd)+'\n');log.flush()
                sp.run(cmd,stdout=log,stderr=sp.STDOUT,check=False)
            for name in ['/proc/sys/kernel/numa_balancing','/sys/module/nvidia_uvm/parameters/uvm_perf_access_counter_migration_enable']:
                log.write('\n'+name+'\n')
                try:log.write(Path(name).read_text())
                except OSError as e:log.write(str(e)+'\n')
        binary = out/'malloc_bound'
        toolkit = Path(nvcc).resolve().parents[1]
        print('Compiling malloc_bound CUDA source...',flush=True)
        with (out/'build.log').open('w') as log:
            sp.run([nvcc,'-std=c++17','-O3','-lineinfo','-arch=sm_90a',str(out/'malloc_bound.cu'),
                    '-L'+str(toolkit/'lib64/stubs'),'-lcuda','-o',str(binary)],stdout=log,stderr=sp.STDOUT,check=True)
        for index,config in enumerate(configs):
            n,h,w = config
            folder = out/'cases'/f'{n}_{h}x{w}'
            folder.mkdir(parents=True,exist_ok=True)
            if not (folder/'done.json').exists():
                cmd = [str(binary),str(n),str(h),str(w),'malloc_bound','128',str(index%3),str(folder/'rows.csv')]
                (folder/'command.json').write_text(json.dumps(cmd))
                with (folder/'worker.log').open('w') as log:
                    active = sp.Popen(cmd,stdout=log,stderr=sp.STDOUT,start_new_session=True)
                    try:rc = active.wait(timeout=900)
                    except sp.TimeoutExpired:
                        os.killpg(active.pid,signal.SIGKILL);active.wait();rc=124
                    active = None
                if rc in (3,124) or rc < 0:
                    raise RuntimeError(f'GPU fault, timeout or signal in {folder.name}; stopping')
                read_case(folder/'rows.csv',config)
                (folder/'done.json').write_text(json.dumps(dict(returncode=rc)))
                collect(out,configs)
            done=index+1;filled=done*30//len(configs)
            print(f'Progress: [{"#"*filled}{"-"*(30-filled)}] {done*100//len(configs):3d}% ({done}/{len(configs)} groups) {n} bytes tile={h}x{w}',flush=True)
        passed,rows = collect(out,configs)
        exit_code = 0 if passed else 1
        if not args.skip_plots and any(r['status']=='ok' for r in rows):
            date = dt.datetime.now(ZoneInfo('Europe/Athens')).strftime('%d_%m_%y')
            dest = outputs/'plots'/f'malloc_bound_sweep_plots_{date}'
            cmd = [sys.executable,str(plot),str(out/'results.csv'),'--malloc-bound-only','--output-dir',str(dest)]
            with (out/'plots.log').open('w') as log:
                proc=sp.run(cmd,stdout=sp.PIPE,stderr=sp.STDOUT,text=True)
                log.write(proc.stdout);print(proc.stdout,flush=True)
            (out/'plots-status.json').write_text(json.dumps(dict(returncode=proc.returncode)))
            if proc.returncode:exit_code=2
    finally:
        if active is not None and active.poll() is None:
            os.killpg(active.pid,signal.SIGKILL);active.wait()
        collect(out,configs)
        (out/'status.json').write_text(json.dumps(dict(exit_code=exit_code)))
        archive = out.with_suffix('.tar.gz')
        with tarfile.open(archive,'w:gz') as tar:
            for p in out.rglob('*'):
                if p.is_file() and p != out/'malloc_bound':tar.add(p,arcname=str(Path(out.name)/p.relative_to(out)))
        print((out/'summary.md').read_text(),flush=True)
        print(f'Outputs: {out}\nSEND_THIS_FILE: {archive}',flush=True)
    return exit_code


if __name__=='__main__':
    try:sys.exit(main())
    except KeyboardInterrupt:sys.exit(130)
