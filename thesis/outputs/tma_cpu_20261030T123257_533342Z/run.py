"""Compile and supervise isolated TMA probes using only the Python standard library."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def parse_worker(text, returncode):
    stages = [line[6:] for line in text.splitlines() if line.startswith('STAGE ')]
    row = {'status': 'worker_error', 'stage': stages[-1] if stages else 'startup', 'exit_code': returncode}
    results = [line[7:] for line in text.splitlines() if line.startswith('RESULT ')]
    if returncode == 0 and len(results) == 1:
        result = json.loads(results[0])
        if result.get('status') != 'correctness_pass':
            raise ValueError('Worker returned an unexpected status')
        row.update(result)
    return row


def verdict(rows):
    controls = [r for r in rows if r['source'] == 'hbm']
    if not controls or any(r['status'] != 'correctness_pass' for r in controls):
        return 'hbm_control_failed_no_host_tma_conclusion'
    host_tma = [r for r in rows if r['source'] != 'hbm' and r['method'] == 'tma']
    if len(host_tma) != 2:
        return 'incomplete'
    if all(r['status'] == 'correctness_pass' and r.get('cpu_residency_verified_before_and_after') for r in host_tma):
        return 'host_tma_correct_cpu_residency_observed_at_checkpoints'
    if any(r['status'] == 'correctness_pass' for r in host_tma):
        return 'some_host_tma_correct_check_per_allocation_residency'
    return 'host_tma_not_demonstrated_inspect_errors_not_a_universal_hardware_verdict'


def supervise(binary, source, method, size, repetitions, folder, timeout):
    folder.mkdir()
    row = dict(source=source, method=method, bytes=size, repetitions=repetitions)
    with (folder / 'worker.log').open('w') as log:
        process = subprocess.Popen([str(binary), source, method, str(size), str(repetitions)],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            # Only our child is terminated; never touch other users' GPU jobs.
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                row.update(status='worker_unresponsive', stage='see_worker_log', exit_code=None)
                save(folder / 'result.json', row)
                return row
            row.update(status='timeout' if isinstance(exc, subprocess.TimeoutExpired) else 'interrupted',
                       stage='see_worker_log', exit_code=process.returncode)
            save(folder / 'result.json', row)
            return row
    row.update(parse_worker((folder / 'worker.log').read_text(), code))
    if row['status'] == 'correctness_pass':
        if (row['source'], row['method'], row['bytes'], row['repetitions']) != (source, method, size, repetitions):
            raise ValueError('Worker result does not match requested trial')
        # Source bytes / kernel time; NOT measured DRAM traffic or a pure TMA bandwidth.
        ms = row['median_kernel_ms']
        row['effective_source_gb_s'] = size / (ms * 1e6) if ms > 0 else None
    save(folder / 'result.json', row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--toolkit', type=Path, required=True)
    parser.add_argument('--bench', action='store_true', help='After smoke passes, add 1MiB/64MiB repeated reads')
    parser.add_argument('--timeout', type=int, default=45, help='Seconds per worker')
    args = parser.parse_args()
    if not 1 <= args.timeout <= 600:
        parser.error('--timeout must be 1..600')
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    out = repo / 'thesis/outputs' / ('tma_cpu_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(parents=True)
    log = (out / 'run.log').open('w', buffering=1)
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    print('Outputs:', out, flush=True)
    status = 2
    try:
        # Repository-wide TMA lock is held for compile and all worker lifetimes.
        import fcntl
        with (repo / 'thesis/outputs/.tma-cpu.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
            if any(line.strip().isdigit() for line in apps.splitlines()):
                raise RuntimeError('GPU compute process detected; no probes started')
            (out / 'gpu-before.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
            (out / 'numa-topology.txt').write_text(subprocess.check_output(['lscpu'], text=True))
            sources = ['tma_cpu.cu', 'run.py', 'README.md']
            hashes = {}
            for name in sources:
                shutil.copy2(here / name, out / name)
                hashes[name] = hashlib.sha256((here / name).read_bytes()).hexdigest()
            runner = repo / 'thesis/scripts/run_tma_cpu.sh'
            shutil.copy2(runner, out / 'runner-used.sh')
            nvcc = args.toolkit / 'bin/nvcc'
            info = subprocess.check_output([str(nvcc), '--version'], text=True)
            if 'release 12.8,' not in info:
                raise RuntimeError('Expected CUDA Toolkit 12.8')
            build = here / 'build'
            build.mkdir(exist_ok=True)
            binary, ptx = build / 'tma_cpu', build / 'tma_cpu.ptx'
            flags = [str(nvcc), '-std=c++17', '-O3', '-lineinfo', '-arch=sm_90a', str(here / 'tma_cpu.cu')]
            commands = [flags + ['-ptx', '-o', str(ptx)], flags + ['-L'+str(args.toolkit/'lib64/stubs'), '-lcuda', '-o', str(binary)]]
            save(out / 'manifest.json', {'toolkit': str(args.toolkit), 'nvcc': info, 'commands': commands,
                 'source_sha256': hashes, 'commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
                 'argv': sys.argv, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                 'note': 'Fresh process per trial; no input staging in HBM for host sources. GPU exclusivity checked at start only.'})
            with (out / 'build.log').open('w') as log:
                for command in commands:
                    print('Compiling:', ' '.join(command), flush=True)
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=180)
            if not re.search(r'cp\.async\.bulk\.tensor\.2d', ptx.read_text()):
                raise RuntimeError('TMA instruction absent in generated PTX')
            (out / 'tma-instruction.txt').write_text('\n'.join(line for line in ptx.read_text().splitlines() if 'cp.async.bulk.tensor' in line)+'\n')
            rows = []
            phases = [(4096, 3)] + ([(1024*1024, 20), (64*1024*1024, 20)] if args.bench else [])
            cases = [('hbm', 'loads'), ('hbm', 'tma'), ('pinned', 'loads'), ('pinned', 'tma'), ('registered', 'loads'), ('registered', 'tma')]
            total = len(phases)*len(cases)
            stop = False
            for size, reps in phases:
                for source, method in cases:
                    print(f'[{len(rows)+1}/{total}] {source} {method} bytes={size}', flush=True)
                    row = supervise(binary, source, method, size, reps, out/f'{size}_{source}_{method}', args.timeout)
                    rows.append(row)
                    save(out / 'results.json', rows)
                    fields = list(dict.fromkeys(k for r in rows for k in r))
                    with (out / 'results.csv').open('w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fields)
                        writer.writeheader(); writer.writerows(rows)
                    print('  ->', row['status'], flush=True)
                    if row['status'] in ('timeout', 'interrupted', 'worker_unresponsive') or (isinstance(row.get('exit_code'), int) and row['exit_code'] < 0) or (source == 'hbm' and row['status'] != 'correctness_pass'):
                        stop = True; break
                # Performance runs require every smoke correctness case to pass first.
                if stop or any(r['status'] != 'correctness_pass' for r in rows):
                    break
            smoke = [r for r in rows if r['bytes'] == 4096]
            status = 0 if len(smoke) == 6 and all(r['status'] == 'correctness_pass' for r in rows) and len(rows) == total else 2
            save(out / 'summary.json', {'verdict': verdict(smoke), 'trials': len(rows), 'exit_code': status,
                 'benchmark_requested': args.bench,
                 'note': 'Correct copy is distinct from CPU page residency. Checkpoint residency is not continuous traffic proof. Timing includes output stores, and repeated reads may hit GPU caches.'})
    except Exception:
        traceback.print_exc()
        status = 2
    finally:
        (out / 'status.txt').write_text(f'exit_code={status}\n')
        with (repo / 'thesis/CHANGELOG.MD').open('a') as f:
            f.write(f'\n- TMA CPU run: `{out.relative_to(repo)}`; exit_code={status}.\n')
        print('Outputs:', out, flush=True)
    return status


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    sys.exit(main())
