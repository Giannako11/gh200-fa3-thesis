#!/usr/bin/env python3
"""Full GH200 sweep with GPU eviction, isolated kernels, and independent allocations.

Two separately reported protocols: gpu_evicted and self_primed. This is explicit
GPU cache conditioning, not a hardware invalidation or CPU cache flush claim.
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

import run_tma_sampled_sweep as sampled

PROTOCOLS = ['gpu_evicted', 'self_primed']
EXTRA = ['cache_protocol', 'replicate_count', 'replicate_median_min_ms',
         'replicate_median_max_ms', 'replicate_median_cv']
FIELDS = sampled.FIELDS + EXTRA
CASE_FILES = ['rows.csv', 'samples.csv', 'conditioning.json', 'command.json', 'worker.log']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def controlled_source(original, measurement, conditioning, controlled):
    source = sampled.sampled_source(original, measurement)
    start = source.index('template<int H, int W>\nvoid measure(')
    end = source.index('void save(', start)
    source = source[:start] + conditioning + '\n' + controlled + '\n' + source[end:]
    hook = '        if (argc == 2 && std::string(argv[1]) == "--header")'
    if source.count(hook) != 1:
        raise ValueError('Unexpected main entry point')
    return source.replace(hook, '        if (argc == 3 && std::string(argv[1]) == "--cache-check") '
                          'return controlled_cache_check(argv[2]);\n' + hook)


def validate_probe(path):
    p = json.loads(path.read_text())
    a, b = p['self_primed_samples_ms'], p['gpu_evicted_samples_ms']
    if any(len(v) != 12 or any(not math.isfinite(x) or x <= 0 for x in v) for v in [a, b]):
        raise ValueError('Incomplete/invalid cache probe samples')
    for key, value in [('self_primed_median_ms', statistics.median(a)),
                       ('gpu_evicted_median_ms', statistics.median(b)),
                       ('evicted_over_primed', statistics.median(b)/statistics.median(a))]:
        if not math.isclose(p[key], value, rel_tol=1e-7):
            raise ValueError('Cache probe statistic mismatch')
    if not (p['probe_separates_conditions'] is True and
            statistics.median(b) > 1.05*statistics.median(a) and
            sampled.quantile(b, .25) > sampled.quantile(a, .75) and
            p['minimum_covered_sms'] == p['expected_sms'] > 0 and
            p['eviction_bytes'] >= 8*p['l2_bytes'] > 0 and
            0 < 2*p['probe_bytes'] <= p['l2_bytes'] and
            p['not_a_hardware_invalidation_proof'] is True):
        raise ValueError('Eviction sanity check inconclusive; full sweep not started')
    return p


def validate_case(folder, config, protocol, replicate, warmup, samples):
    rows, raw = sampled.read_case(folder, config, warmup, samples, isolated=True)
    if any(r['status'] != 'ok' or (config[3] != 'hbm' and r['cpu_residency_verified'] != '1')
           or r['cpu_node'] != '0' for r in rows):
        raise ValueError('Correctness/CPU placement failed: ' + str(folder))
    for r in rows:
        k = sampled.KERNELS.index(r['kernel'])
        if int(r['first_sample_position']) != (k-replicate) % 3:
            raise ValueError('Replicate kernel block ordering mismatch')
    c = json.loads((folder/'conditioning.json').read_text())
    calls = 3*samples
    if not (c['protocol'] == protocol and c['eviction_bytes'] >= 8*c['l2_bytes'] > 0 and
            c['block_bytes'] >= 512*1024 and c['blocks'] >= 8*c['expected_sms'] > 0 and
            c['eviction_bytes'] == c['blocks']*c['block_bytes'] and
            c['minimum_covered_sms'] == c['expected_sms'] and c['passes'] == 2 and
            c['eviction_calls'] == calls and 2*calls <= c['eviction_kernel_launches'] <= 6*calls and
            c['primers_per_sample'] == int(protocol == 'self_primed') and
            c['hardware_invalidation_claim'] is False and c['cpu_cache_flush'] is False):
        raise ValueError('Invalid/incomplete GPU conditioning metadata')
    return rows, raw


def pool_cases(cases, protocol):
    """Pool equal sample counts, retaining the dispersion of replicate medians."""
    pooled = []
    for kernel in sampled.KERNELS:
        rr = [next(r for r in rows if r['kernel'] == kernel) for rows, raw in cases]
        times = [float(r['kernel_ms']) for rows, raw in cases for r in raw if r['kernel'] == kernel]
        medians = [float(r['kernel_ms']) for r in rr]
        r = dict(rr[0])
        r.update(cache_protocol=protocol, replicate_count=len(cases), sample_count=len(times),
                 first_sample_position=-1, first_sample_ms=times[0],
                 kernel_ms=statistics.median(times), min_ms=min(times), max_ms=max(times),
                 p25_ms=sampled.quantile(times, .25), p75_ms=sampled.quantile(times, .75),
                 p95_ms=sampled.quantile(times, .95), mean_ms=statistics.mean(times),
                 stddev_ms=statistics.stdev(times), cv=statistics.stdev(times)/statistics.mean(times),
                 replicate_median_min_ms=min(medians), replicate_median_max_ms=max(medians),
                 replicate_median_cv=statistics.stdev(medians)/statistics.mean(medians),
                 message='Pooled independent allocations; page counts summed across replicates; IQR is dispersion, not CI')
        r['effective_input_gb_s'] = int(r['input_bytes'])/(r['kernel_ms']*1e6)
        for phase in sampled.PHASES:
            for field in ['pages', 'on_cpu', 'unknown', 'query_errno']:
                r[field+'_'+phase] = sum(int(x[field+'_'+phase]) for x in rr)
        pooled.append(r)
    return pooled


def case_folder(out, key, configs):
    protocol, rep, index = key
    n, h, w, source = configs[index]
    return out/'cases'/protocol/f'rep{rep+1:02d}'/f'{n}_{h}x{w}_{source}'


def checkpoint(out, records, pooled, total, configs, replicates, samples):
    for protocol in PROTOCOLS:
        dest = out/protocol
        dest.mkdir(exist_ok=True)
        rows = [r for (p, index), group in sorted(pooled.items()) if p == protocol for r in group]
        sampled.write_csv(dest/'results.csv', FIELDS, rows)
        done = sum(k[0] == protocol for k in records)
        (dest/'summary.md').write_text(
            f'# {protocol}\n\n- Validated workers: {done} / {len(configs)*replicates}\n'
            f'- Complete pooled rows: {len(rows)} / {len(configs)*3}\n'
            f'- Samples per complete row: {samples*replicates} from {replicates} fresh allocations\n'
            f'- Rows with replicate median CV > 5%: {sum(float(r["replicate_median_cv"])>.05 for r in rows)}\n'
            f'- Rows with pooled sample CV > 10%: {sum(float(r["cv"])>.10 for r in rows)}\n\n'
            'All included rows passed output checks, placement snapshots and conditioning checks.\n'
            'Correctness PASS does not assert timing stability. See per-replicate rows and raw samples.\n')
    passed = len(records) == total and len(pooled) == 2*len(configs)
    (out/'summary.md').write_text(
        '# Controlled TMA sweep\n\n'
        f'- Validated worker processes: {len(records)} / {total}\n'
        f'- Validated timing samples: {len(records)*3*samples} / {total*3*samples}\n'
        f'- Complete pooled rows: {len(pooled)*3} / {2*len(configs)*3}\n'
        f'- Overall output/placement/conditioning: {"PASS" if passed else "INCOMPLETE_OR_FAILED"}\n\n'
        'gpu_evicted: eviction -> timed target.\n'
        'self_primed: eviction -> one untimed SAME target -> timed target.\n'
        'Each kernel is sampled in an isolated block. Block order cycles across independent allocations.\n'
        'Events include the full copy kernel and its HBM output writes. Conditioning/primer are excluded.\n'
        'GPU eviction is empirical conditioning, not guaranteed hardware invalidation. Grace CPU caches are not flushed.\n'
        'Page placement is checked at four snapshots per kernel, not continuously.\n'
        'Medians/IQR pool equal samples from independent allocations. IQR is not a confidence interval.\n'
        'Effective input GB/s is not physical DRAM or NVLink bandwidth. No outliers are deleted.\n')
    return passed


def rebuild_exports(out, records, configs):
    """Authoritative case files repair any interrupted aggregate write on resume."""
    for protocol in PROTOCOLS:
        dest = out/protocol
        dest.mkdir(exist_ok=True)
        for name, fields in [('samples.csv', sampled.RAW_FIELDS), ('replicate_results.csv', sampled.FIELDS)]:
            target = dest/name
            temp = target.with_suffix('.csv.tmp')
            with temp.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['cache_protocol', 'replicate']+fields)
                writer.writeheader()
                for key in sorted(records):
                    if key[0] != protocol:
                        continue
                    folder = case_folder(out, key, configs)
                    original = sampled.load_csv(folder/('rows.csv' if name.startswith('replicate') else name), fields)
                    writer.writerows(dict(cache_protocol=protocol, replicate=key[1]+1, **r) for r in original)
            temp.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='One 1 MiB tile, all allocations, both conditions, all replicates')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--replicates', type=int, default=3, help='Multiple of 3, fresh processes/allocations')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--skip-plots', action='store_true')
    parser.add_argument('--no-telemetry', action='store_true')
    args = parser.parse_args()
    if args.warmup < 1 or args.samples < 6 or args.samples % 6 or args.replicates < 3 or args.replicates % 3:
        parser.error('Require warmup>=1, samples>=6 divisible by 6, replicates>=3 divisible by 3')
    pairs = [(1<<20, 32, 32)] if args.smoke else sampled.configurations()
    configs = [(*pair, s) for pair in pairs for s in sampled.SOURCES]
    work = [(p, rep, i) for i in range(len(configs)) for rep in range(args.replicates)
            for p in (PROTOCOLS if (i+rep)%2 == 0 else PROTOCOLS[::-1])]
    print(f'Plan: {len(configs)} allocation configs x 2 conditions x {args.replicates} replicates = '
          f'{len(work)} fresh workers; {3*len(work)*args.samples} raw samples; '
          f'{len(configs)*3} pooled rows/condition. Warmup={args.warmup}, samples={args.samples}/replicate.', flush=True)
    if args.plan:
        return 0
    repo = Path(__file__).resolve().parents[2]
    outputs = repo/'thesis/outputs'
    source_dir = repo/'thesis/microbenchmarks/tma_sweep'
    source_names = ['tma_sweep.cu', 'tma_sampled_measure.inc', 'tma_controlled_conditioning.inc', 'tma_controlled_measure.inc']
    dependencies = [source_dir/n for n in source_names] + [Path(__file__),
                    Path(__file__).with_name('run_tma_sampled_sweep.py'),
                    Path(__file__).with_name('run_malloc_bound_sweep.py'),
                    Path(__file__).with_name('run_malloc_diagnostics.py')]
    plot = repo/'thesis/scripts/plots/tma_plots_first.py'
    generated = controlled_source(*(p.read_text() for p in dependencies[:4]))
    if not args.skip_plots:
        sp.run([sys.executable, str(plot), '--help'], stdout=sp.DEVNULL, check=True)
    for cmd in ['nvidia-smi', 'numactl']:
        if not shutil.which(cmd):
            raise RuntimeError(cmd+' not found')
    outputs.mkdir(parents=True, exist_ok=True)
    lock = (outputs/'.tma_sweep.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    apps = sp.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    if any(x.strip().isdigit() for x in apps.splitlines()):
        raise RuntimeError('GPU compute process detected; start on an idle GPU')
    candidates = [Path(os.environ.get('CONDA_PREFIX', '/nonexistent'))/'cuda-12.8.1/bin/nvcc', Path('/usr/local/cuda-12.8/bin/nvcc')]
    nvcc = next((str(p) for p in candidates if p.is_file()), shutil.which('nvcc'))
    if not nvcc:
        raise RuntimeError('Activate fa3: nvcc not found')
    manifest = dict(protocol='isolated_gpu_conditioned_v1', configs=configs, warmup=args.warmup,
                    samples=args.samples, replicates=args.replicates, telemetry=not args.no_telemetry,
                    cuda_sha256=hashlib.sha256(generated.encode()).hexdigest(),
                    dependencies={p.name:digest(p) for p in dependencies+[plot]},
                    compiler=sp.check_output([nvcc, '--version'], text=True),
                    gpu=sp.check_output(['nvidia-smi', '--query-gpu=uuid,driver_version', '--format=csv,noheader'], text=True),
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    cuda_module_loading=os.environ.get('CUDA_MODULE_LOADING', 'driver default'))
    manifest = json.loads(json.dumps(manifest))
    if args.resume:
        out = args.resume.resolve()
        if out.parent != outputs.resolve() or not out.name.startswith('tma_controlled_sweep_'):
            raise ValueError('Invalid controlled resume folder')
        if json.loads((out/'manifest.json').read_text()) != manifest:
            raise ValueError('Resume source/protocol/GPU/compiler mismatch')
    else:
        stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        out = Path(tempfile.mkdtemp(prefix=f'tma_controlled_sweep_{stamp}_', dir=outputs))
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
        (out/'controlled.cu').write_text(generated)
        for p in dependencies+[plot]:
            shutil.copyfile(p, out/p.name)
        (out/'base_commit.txt').write_text(sp.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True))
    if digest(out/'controlled.cu') != manifest['cuda_sha256']:
        raise ValueError('Archived CUDA source mismatch')
    if any(digest(out/name) != value for name, value in manifest['dependencies'].items()):
        raise ValueError('Archived dependency mismatch')
    print(f'Outputs: {out}', flush=True)
    records, pooled = {}, {}
    active = telemetry = None
    telemetry_log = None
    exit_code = 1
    started = time.monotonic()

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGHUP, interrupt)

    def collect(protocol, index):
        keys = [(protocol, rep, index) for rep in range(args.replicates)]
        if (protocol, index) not in pooled and all(k in records for k in keys):
            cases = [(records[k], sampled.load_csv(case_folder(out, k, configs)/'samples.csv', sampled.RAW_FIELDS)) for k in keys]
            pooled[protocol, index] = pool_cases(cases, protocol)

    try:
        # Resume validates hashes AND semantic content; stale appends cannot pass.
        for key in work:
            folder = case_folder(out, key, configs)
            if not (folder/'done.json').exists():
                continue
            done = json.loads((folder/'done.json').read_text())
            if done['returncode'] != 0 or any(digest(folder/n) != done[n] for n in CASE_FILES):
                raise ValueError('Checkpoint content mismatch: '+str(folder))
            rows, raw = validate_case(folder, configs[key[2]], key[0], key[1], args.warmup, args.samples)
            records[key] = rows
            collect(key[0], key[2])
        checkpoint(out, records, pooled, len(work), configs, args.replicates, args.samples)
        with (out/'environment.txt').open('a') as log:
            for cmd in [['uname', '-a'], ['nvidia-smi', '-q'], ['numactl', '--hardware'], ['numactl', '--show']]:
                log.write('\nCOMMAND '+repr(cmd)+'\n'); log.flush()
                sp.run(cmd, stdout=log, stderr=sp.STDOUT, check=False)
            for name in ['/proc/sys/kernel/numa_balancing',
                         '/sys/module/nvidia_uvm/parameters/uvm_perf_access_counter_migration_enable']:
                log.write('\n'+name+'\n')
                try:
                    log.write(Path(name).read_text())
                except OSError as error:
                    log.write(str(error)+'\n')
        binary = out/'controlled'
        toolkit = Path(nvcc).resolve().parents[1]
        print('Compiling controlled CUDA benchmark...', flush=True)
        with (out/'build.log').open('w') as log:
            sp.run([nvcc, '-std=c++17', '-O3', '-lineinfo', '-arch=sm_90a', str(out/'controlled.cu'),
                    '-L'+str(toolkit/'lib64/stubs'), '-lcuda', '-o', str(binary)], stdout=log, stderr=sp.STDOUT, check=True)
        prefix = ['numactl', '--cpunodebind=0', '--membind=0', str(binary)]
        # Run on every invocation, including resume; no bypass for an inconclusive probe.
        probe_id = str(time.time_ns())
        probe = out/('cache_check_'+probe_id+'.json')
        print('Checking GPU eviction conditioning before the sweep...', flush=True)
        with (out/('cache_check_'+probe_id+'.log')).open('w') as log:
            active = sp.Popen(prefix+['--cache-check', str(probe)], stdout=log, stderr=sp.STDOUT, start_new_session=True)
            rc = active.wait(timeout=180); active = None
        if rc:
            raise RuntimeError('Cache probe failed/inconclusive. Inspect '+str(log.name))
        validate_probe(probe)
        if not args.no_telemetry:
            telemetry_log = (out/('telemetry_'+probe_id+'.csv')).open('w')
            telemetry = sp.Popen(['nvidia-smi', '--query-gpu=timestamp,uuid,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,memory.used',
                                  '--format=csv', '--loop=1'], stdout=telemetry_log, stderr=sp.STDOUT)
        env = dict(os.environ, TMA_SAMPLE_WARMUP=str(args.warmup), TMA_SAMPLE_COUNT=str(args.samples))
        for key in work:
            protocol, rep, index = key
            config = configs[index]
            n, h, w, source = config
            folder = case_folder(out, key, configs)
            if key not in records:
                if folder.exists():
                    folder.rename(folder.with_name(folder.name+'.failed_'+str(time.time_ns())))
                folder.mkdir(parents=True)
                cmd = prefix+[str(n), str(h), str(w), source, '128', str(rep%3), str(folder/'rows.csv')]
                (folder/'command.json').write_text(json.dumps(dict(argv=cmd, protocol=protocol, replicate=rep+1,
                                                                 warmup=args.warmup, samples=args.samples,
                                                                 started_utc=dt.datetime.now(dt.timezone.utc).isoformat())))
                worker_start = time.monotonic()
                worker_env = dict(env, TMA_CONTROLLED_PROTOCOL=protocol, TMA_CONTROLLED_CASE_DIR=str(folder))
                with (folder/'worker.log').open('w') as log:
                    active = sp.Popen(cmd, env=worker_env, stdout=log, stderr=sp.STDOUT, start_new_session=True)
                    rc = active.wait(timeout=900); active = None
                if rc:
                    raise RuntimeError(f'Worker exit {rc}: {folder}; stopping with all diagnostics retained')
                rows, raw = validate_case(folder, config, protocol, rep, args.warmup, args.samples)
                done = dict(returncode=rc, finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                            elapsed_seconds=time.monotonic()-worker_start,
                            **{name:digest(folder/name) for name in CASE_FILES})
                temp = folder/'done.json.tmp'; temp.write_text(json.dumps(done)); temp.replace(folder/'done.json')
                records[key] = rows
                collect(protocol, index)
                if len(records)%12 == 0:
                    checkpoint(out, records, pooled, len(work), configs, args.replicates, args.samples)
            count = len(records); filled = count*30//len(work)
            print(f'Progress: [{"#"*filled}{"-"*(30-filled)}] {100*count//len(work):3d}% '
                  f'({count}/{len(work)} workers) {protocol} rep={rep+1} {n}B {h}x{w} {source} '
                  f'elapsed={int(time.monotonic()-started)}s', flush=True)
        passed = checkpoint(out, records, pooled, len(work), configs, args.replicates, args.samples)
        exit_code = 0 if passed else 1
        if not args.skip_plots:
            for protocol in PROTOCOLS:
                dest = out/'plots'/protocol
                dest.parent.mkdir(exist_ok=True)
                with (out/(protocol+'_plots.log')).open('w') as log:
                    result = sp.run([sys.executable, str(plot), str(out/protocol/'results.csv'), '--sampled',
                                     '--output-dir', str(dest)], stdout=log, stderr=sp.STDOUT)
                if result.returncode:
                    exit_code = 2
                print(f'{protocol} plots: {dest} (exit {result.returncode})', flush=True)
        if telemetry is not None and telemetry.poll() is not None:
            print('Telemetry stopped early; inspect telemetry CSV. Clock stability is unverified.', flush=True)
    except BaseException as error:
        exit_code = 130 if isinstance(error, KeyboardInterrupt) else 1
        (out/'failure.txt').write_text(type(error).__name__+': '+str(error)+'\n')
        raise
    finally:
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGKILL); active.wait()
        if telemetry is not None:
            was_running = telemetry.poll() is None
            if was_running:
                telemetry.terminate()
                try:
                    telemetry.wait(timeout=10)
                except sp.TimeoutExpired:
                    telemetry.kill(); telemetry.wait()
            (out/('telemetry_status_'+probe_id+'.json')).write_text(json.dumps(dict(running_until_end=was_running, returncode=telemetry.returncode)))
        if telemetry_log is not None:
            telemetry_log.close()
        rebuild_exports(out, records, configs)
        checkpoint(out, records, pooled, len(work), configs, args.replicates, args.samples)
        (out/'status.json').write_text(json.dumps(dict(exit_code=exit_code)))
        archive = out.with_suffix('.tar.gz')
        with tarfile.open(archive, 'w:gz') as tar:
            for path in out.rglob('*'):
                if path.is_file() and path != out/'controlled':
                    tar.add(path, arcname=str(Path(out.name)/path.relative_to(out)))
        print((out/'summary.md').read_text(), flush=True)
        print(f'Outputs: {out}\nSEND_THIS_FILE: {archive}', flush=True)
    return exit_code


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
