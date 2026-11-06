#!/usr/bin/env python3
"""Profile selected timed TMA launches around the 32 MiB host bimodality.

Uses the exact compiled executable from a completed controlled sweep. Nsight
Compute application replay preserves the program's eviction/primer sequence.
Profiler timings are diagnostic, not replacements for unprofiled CUDA events.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess as sp
import sys
import tarfile
import tempfile
import time

MODES = ('gpu_evicted', 'self_primed')
SOURCES = ('registered', 'pinned', 'malloc_bound')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', action='store_true', help='Show the exact cases and launch indices without profiling')
    p.add_argument('--sweep', type=Path, help='Completed tma_controlled_sweep_* directory (defaults to latest)')
    p.add_argument('--sources', nargs='+', choices=SOURCES, default=['registered'])
    p.add_argument('--replicates', type=int, default=3)
    p.add_argument('--sample-indices', nargs='+', type=int, default=[3, 5],
                   help='1-based timed sample numbers to inspect; default 3 and 5')
    args = p.parse_args()
    if args.replicates < 1 or len(set(args.sources)) != len(args.sources) or not args.sample_indices or \
            len(set(args.sample_indices)) != len(args.sample_indices) or min(args.sample_indices) < 1 or \
            max(args.sample_indices) > 6:
        p.error('Unique sources and 1..6 unique sample indices; replicates>=1')
    repo = Path(__file__).resolve().parents[2]
    outputs = repo/'thesis/outputs'
    if args.sweep:
        sweep = args.sweep.resolve()
    else:
        candidates = [x for x in outputs.glob('tma_controlled_sweep_*') if x.is_dir() and (x/'status.json').exists()]
        candidates = [x for x in candidates if json.loads((x/'status.json').read_text()).get('exit_code') == 0]
        if not candidates:
            raise RuntimeError('No completed controlled sweep found; use --sweep DIR')
        sweep = max(candidates, key=lambda x: x.stat().st_mtime_ns)
    if sweep.parent != outputs.resolve() or not sweep.name.startswith('tma_controlled_sweep_'):
        raise ValueError('Supply a controlled sweep directory inside this repo outputs')
    manifest = json.loads((sweep/'manifest.json').read_text())
    if manifest['protocol'] != 'isolated_gpu_conditioned_v1' or manifest['warmup'] != 10:
        raise ValueError('Expected the original controlled sweep and its 10 warmups')
    if json.loads((sweep/'status.json').read_text()).get('exit_code') != 0:
        raise ValueError('Sweep is incomplete')
    binary = sweep/'controlled'
    if not binary.is_file() or hashlib.sha256((sweep/'controlled.cu').read_bytes()).hexdigest() != manifest['cuda_sha256']:
        raise ValueError('Missing compiled binary or archived source mismatch')
    # Only the third benchmark kernel matches the ncu name filter. Its first
    # invocation is correctness, followed by 10 warmups. In self_primed each
    # sample contributes two invocations: primer and timed target. ncu's
    # --launch-skip counts only matching kernel names (NVIDIA CLI definition).
    cases = []
    for source in args.sources:
        for rep in range(args.replicates):
            for mode in (MODES if rep % 2 == 0 else MODES[::-1]):
                for sample in args.sample_indices:
                    skip = 1 + manifest['warmup'] + (sample-1)*(2 if mode == 'self_primed' else 1)
                    if mode == 'self_primed':
                        skip += 1  # skip this sample's primer; select its timed launch
                    cases.append((source, rep, mode, sample, skip))
    print('Source sweep:', sweep, flush=True)
    print(f'Plan: {len(cases)} separate profiler reports; 32 MiB, 128x128, tma_shared; '
          'ncu application replay, cache-control none, clock-control none.', flush=True)
    for source, rep, mode, sample, skip in cases:
        print(f'  {source} allocation {rep+1}, {mode}, timed sample {sample}: '
              f'skip {skip} matching TMA launches, profile 1', flush=True)
    if args.plan:
        return 0
    ncu = shutil.which('ncu') or next((str(x) for x in
        [Path(os.environ.get('CONDA_PREFIX', '/nonexistent'))/'cuda-12.8.1/bin/ncu',
         Path('/usr/local/cuda-12.8/bin/ncu')] if x.is_file()), None)
    if not ncu:
        raise RuntimeError('Nsight Compute (ncu) not found; activate fa3 / check toolkit')
    if not shutil.which('numactl'):
        raise RuntimeError('numactl not found')
    info = sp.check_output([ncu, '--list-sections'], text=True, stderr=sp.STDOUT)
    if 'MemoryWorkloadAnalysis' not in info:
        raise RuntimeError('MemoryWorkloadAnalysis section missing from ncu; inspect ncu --list-sections')
    apps = sp.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    if any(s.strip().isdigit() for s in apps.splitlines()):
        raise RuntimeError('Other GPU compute process present; retry on an idle GPU')
    outputs.mkdir(parents=True, exist_ok=True)
    lock = (outputs/'.tma_sweep.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = Path(tempfile.mkdtemp(prefix=f'tma_targeted_profile_{stamp}_', dir=outputs))
    (out/'plan.json').write_text(json.dumps(dict(source_sweep=str(sweep), source_sha256=manifest['cuda_sha256'],
                ncu=ncu, ncu_version=sp.check_output([ncu, '--version'], text=True), cases=cases), indent=2))
    with (out/'ncu_sections.txt').open('w') as f:
        f.write(info)
    print('Outputs:', out, flush=True)
    completed = []
    try:
        for index, (source, rep, mode, sample, skip) in enumerate(cases, 1):
            name = f'{source}_rep{rep+1}_{mode}_sample{sample}'
            folder = out/name
            folder.mkdir()
            prefix = folder/'tma'
            cmd = ['numactl', '--cpunodebind=0', '--membind=0', ncu,
                   '--replay-mode', 'application', '--cache-control', 'none', '--clock-control', 'none',
                   '--section', 'MemoryWorkloadAnalysis', '--kernel-name-base', 'function',
                   '--kernel-name', 'regex:tma_shared', '--launch-skip', str(skip),
                   '--launch-count', '1', '--export', str(prefix),
                   str(binary), str(1<<25), '128', '128', source, '128', str(rep%3), str(folder/'rows.csv')]
            env = dict(os.environ, TMA_CONTROLLED_PROTOCOL=mode, TMA_CONTROLLED_CASE_DIR=str(folder),
                       TMA_SAMPLE_WARMUP='10', TMA_SAMPLE_COUNT='6')
            (folder/'command.json').write_text(json.dumps(dict(argv=cmd, env={k:env[k] for k in
                      ['TMA_CONTROLLED_PROTOCOL','TMA_CONTROLLED_CASE_DIR','TMA_SAMPLE_WARMUP','TMA_SAMPLE_COUNT']}), indent=2))
            started = time.monotonic()
            with (folder/'ncu.log').open('w') as log:
                rc = sp.run(cmd, env=env, stdout=log, stderr=sp.STDOUT).returncode
            report = folder/'tma.ncu-rep'
            if rc or not report.is_file() or not report.stat().st_size:
                raise RuntimeError(f'{name}: profiler failed or missing .ncu-rep (exit {rc}); inspect {folder}/ncu.log')
            # Keep the original target's correctness, placement and sampler
            # checks, but never use profiled timings as unprofiled measurements.
            import run_tma_sampled_sweep as sampled
            rows, raw = sampled.read_case(folder, (1<<25,128,128,source),10,6,isolated=True)
            if any(x['status']!='ok' or x['cpu_residency_verified']!='1' or x['cpu_node']!='0' for x in rows):
                raise RuntimeError(f'{name}: correctness/NUMA check failed')
            meta = json.loads((folder/'conditioning.json').read_text())
            if not (meta['protocol']==mode and meta['eviction_calls']==18 and
                    meta['minimum_covered_sms']==meta['expected_sms']>0 and
                    meta['eviction_bytes']>=8*meta['l2_bytes']>0 and
                    meta['primers_per_sample']==int(mode=='self_primed')):
                raise RuntimeError(f'{name}: conditioning incomplete')
            with (folder/'ncu_raw.csv').open('w') as rawcsv:
                export = sp.run([ncu, '--import', str(report), '--page', 'raw', '--csv'],
                                stdout=rawcsv, stderr=sp.PIPE, text=True)
            if export.returncode or not (folder/'ncu_raw.csv').stat().st_size:
                raise RuntimeError(f'{name}: ncu report CSV export failed: {export.stderr}')
            record = dict(case=name, source=source, allocation=rep+1, protocol=mode,
                          selected_sample=sample, matching_launches_skipped=skip,
                          profiled_process_event_ms=float(next(r['kernel_ms'] for r in raw
                               if r['kernel']=='tma_shared' and int(r['sample_index'])==sample)),
                          report=str(report), seconds=round(time.monotonic()-started,3), status='ok')
            completed.append(record)
            (out/'status.json').write_text(json.dumps(dict(completed=completed,expected=len(cases)),indent=2))
            filled=index*30//len(cases)
            print(f'Progress: [{"#"*filled}{"-"*(30-filled)}] {index}/{len(cases)} {name}',flush=True)
    finally:
        (out/'status.json').write_text(json.dumps(dict(completed=completed,expected=len(cases)),indent=2))
        bundle = out/'reports.tar.gz'
        with tarfile.open(bundle, 'w:gz') as archive:
            for child in sorted(out.iterdir()):
                if child != bundle:
                    archive.add(child, arcname=child.name)
        print('Outputs:', out, flush=True)
        print('Shareable reports archive:', bundle, flush=True)
    print('Profiled reports are for traffic/cache diagnosis; use the original sweep CSV for timings.',flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
