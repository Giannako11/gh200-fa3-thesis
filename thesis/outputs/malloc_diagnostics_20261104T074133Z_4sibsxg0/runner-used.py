#!/usr/bin/env python3
"""Isolated GH200 placement diagnostics; reuses the existing three TMA kernels.

Run in fa3. Writes only a new thesis/outputs/malloc_diagnostics_* directory.
Does not modify the benchmark source, machine settings, or existing results.
The resulting diagnostic CSV is intentionally separate from sweep/plot inputs.
"""
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess as sp
import sys
import tarfile
import tempfile


def instrument(source):
    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError("Benchmark source differs: expected exactly one " + repr(old))
        source = source.replace(old, new)

    replace('#include <linux/mempolicy.h>', '#include <linux/mempolicy.h>\n#include <map>')
    helper = r'''
// Query both the calling thread policy and the allocation's effective policy.
void diag_policy(void* address) {
    int mode = -1;
    unsigned long mask[16] = {};
    errno = 0;
    long rc = syscall(SYS_get_mempolicy, &mode, mask, 1024UL, address,
                      address ? MPOL_F_ADDR : 0);
    int error = errno;
    std::cout << "DIAG_POLICY scope=" << (address ? "address" : "thread")
              << " rc=" << rc << " errno=" << (rc < 0 ? error : 0)
              << " mode=" << mode << " nodes=";
    for (unsigned i=0; i<1024; ++i)
        if (mask[i/(8*sizeof(unsigned long))] & (1UL << (i%(8*sizeof(unsigned long)))))
            std::cout << i << ':';
    std::cout << std::endl;
}
'''
    replace('struct Location {', helper + '\nstruct Location {')
    replace('    Location r;\n', '    Location r;\n    std::map<int,size_t> node_counts, negative_status;\n')
    replace('                if (n < 0) ++r.unknown;\n                else if (unsigned(n) == node) ++r.on_cpu;',
            '                if (n < 0) { ++r.unknown; ++negative_status[n]; }\n'
            '                else { ++node_counts[n]; if (unsigned(n) == node) ++r.on_cpu; }')
    replace('    return r;\n}\n\nstruct Memory {',
            '    std::cout << "DIAG_LOCATION pages=" << r.pages << " nodes=";\n'
            '    for (const auto& v : node_counts) std::cout << v.first << ":" << v.second << ";";\n'
            '    std::cout << " negative_status=";\n'
            '    for (const auto& v : negative_status) std::cout << v.first << ":" << v.second << ";";\n'
            '    std::cout << " query_errno=" << r.error << std::endl;\n'
            '    return r;\n}\n\nstruct Memory {')
    replace('                host = static_cast<uint32_t*>(std::malloc(bytes));\n                if (!host) throw std::runtime_error("malloc failed");', r'''
                std::string mode = std::getenv("MALLOC_DIAG_MODE") ? std::getenv("MALLOC_DIAG_MODE") : "plain";
                if (mode == "plain" || mode == "taskbind") {
                    host = static_cast<uint32_t*>(std::malloc(bytes));
                    if (!host) throw std::runtime_error("malloc failed");
                } else {
                    void* allocation = nullptr;
                    int rc = posix_memalign(&allocation, size_t(sysconf(_SC_PAGESIZE)), bytes);
                    if (rc) throw std::runtime_error(std::string("posix_memalign: ")+std::strerror(rc));
                    host = static_cast<uint32_t*>(allocation);
                }
                if (mode == "rangebind") {
                    if (node >= 8*sizeof(unsigned long)) throw std::runtime_error("CPU node exceeds diagnostic mask");
                    unsigned long mask = 1UL << node;
                    bound = syscall(SYS_mbind, host, bytes, MPOL_BIND, &mask,
                                    8*sizeof(unsigned long), 0) == 0;
                    if (!bound) throw std::runtime_error(std::string("diagnostic mbind: ")+std::strerror(errno));
                }
                if (mode == "preferred") {
                    check(cudaMemAdvise(host, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
                    std::cout << "DIAG_PREFERRED_HOST success=1" << std::endl;
                }
''')
    replace('            m.allocate(node);',
            '            diag_policy(nullptr);\n            m.allocate(node);\n'
            '            if (m.host) diag_policy(m.host);')
    replace('        if (m.host) row.before = locate(m.host, m.bytes, node);',
            '        if (m.host) { stage("before_gpu"); diag_policy(m.host); row.before = locate(m.host, m.bytes, node); }')
    replace('        verify(m);\n        if (m.host) row.after = locate(m.host, m.bytes, node);',
            '        if (m.host) { stage("after_gpu_before_verification"); row.after = locate(m.host, m.bytes, node); }\n'
            '        verify(m);\n'
            '        if (m.host) { stage("after_verification"); (void)locate(m.host, m.bytes, node); }')
    return source


def summarize(out, cases, expected):
    records = []
    for c in cases:
        path = out / c['case'] / 'rows.csv'
        if path.exists():
            with path.open() as f:
                for r in csv.DictReader(f):
                    records.append(dict(diag_mode=c['mode'], case=c['case'], **r))
    if records:
        with (out / 'diagnostic_results.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)
    lines = ['# Malloc placement diagnostics', '',
             f'Completed workers: {len(cases)} / {expected}',
             'Residency in the CSV is checked BEFORE GPU and IMMEDIATELY AFTER GPU, before CPU verification.',
             'worker.log also records all NUMA node IDs, negative page statuses, policies and post-verification residency.', '',
             '| Mode | Correct / recorded | CPU residency verified / correct host rows |',
             '|---|---:|---:|']
    for mode in dict.fromkeys(c['mode'] for c in cases):
        rr = [r for r in records if r['diag_mode'] == mode]
        ok = [r for r in rr if r['status'] == 'ok']
        host = [r for r in ok if r['source'] != 'hbm']
        n = sum(r['cpu_residency_verified'] == '1' for r in host)
        lines.append(f'| {mode} | {len(ok)} / {len(rr)} | {n} / {len(host)} |')
    lines += ['', 'Worker return codes: ' + json.dumps(cases), '',
              'plain/taskbind use original malloc. aligned is the alignment control.',
              'rangebind = aligned allocation + explicit per-range mbind, with no CUDA registration.',
              'preferred = aligned allocation + CUDA preferred CPU location, with no CUDA registration.',
              'pinned/registered/hbm are controls. Every worker uses a fresh process.',
              'Single launches and rotating method order: this is a placement diagnostic, not final performance evidence.',
              'A mode that returns an error is recorded as failed; no silent fallback is used.',
              'Do not merge this CSV into the original sweep or its plots.']
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')
    (out / 'cases.json').write_text(json.dumps(cases, indent=2))


def main():
    repo_setting = os.environ.get('MALLOC_DIAG_REPO')
    repo = Path(repo_setting).resolve() if repo_setting else Path(__file__).resolve().parents[2]
    original = repo / 'thesis/microbenchmarks/tma_sweep/tma_sweep.cu'
    if '--check-source' in sys.argv:
        instrument(original.read_text())
        print('SOURCE_INSTRUMENTATION_OK')
        return
    outputs = repo / 'thesis/outputs'
    outputs.mkdir(exist_ok=True)
    lock = (outputs / '.tma_sweep.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for cmd in ['nvidia-smi', 'numactl']:
        if not shutil.which(cmd):
            raise RuntimeError('Missing command: ' + cmd)
    apps = sp.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    if any(x.strip().isdigit() for x in apps.splitlines()):
        raise RuntimeError('GPU compute process detected; diagnostics not started')
    candidates = [Path(os.environ.get('CONDA_PREFIX', '/nonexistent')) / 'cuda-12.8.1/bin/nvcc',
                  Path('/usr/local/cuda-12.8/bin/nvcc')]
    nvcc = next((str(p) for p in candidates if p.is_file()), shutil.which('nvcc'))
    if not nvcc:
        raise RuntimeError('Activate fa3: CUDA nvcc was not found')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = Path(tempfile.mkdtemp(prefix=f'malloc_diagnostics_{stamp}_', dir=outputs))
    print(f'Outputs: {out}', flush=True)
    cases = []
    modes = [('plain', 'malloc'), ('taskbind', 'malloc'), ('aligned', 'malloc'),
             ('rangebind', 'malloc'), ('preferred', 'malloc'),
             ('pinned', 'pinned'), ('registered', 'registered'), ('hbm', 'hbm')]
    sizes = [1 << 20, 64 << 20, 1 << 30, 16 << 30]
    tiles = [(32, 32), (128, 128)]
    expected = len(modes)*len(sizes)*len(tiles)
    active = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try:
        shutil.copyfile(original, out / 'original.cu')
        shutil.copyfile(__file__, out / 'runner-used.py')
        (out / 'original.sha256').write_text(hashlib.sha256(original.read_bytes()).hexdigest() + '\n')
        (out / 'diagnostic.cu').write_text(instrument(original.read_text()))
        with (out / 'environment.txt').open('w') as log:
            for cmd in [['uname', '-a'], ['nvidia-smi', '-q'], ['numactl', '--hardware'],
                        ['numactl', '--show'], ['numactl', '--membind=0', 'numactl', '--show'],
                        [nvcc, '--version'], ['git', '-C', str(repo), 'rev-parse', 'HEAD']]:
                log.write('\nCOMMAND ' + repr(cmd) + '\n'); log.flush()
                sp.run(cmd, stdout=log, stderr=sp.STDOUT, check=False)
            for name in ['/proc/sys/kernel/numa_balancing', '/proc/meminfo',
                         '/sys/module/nvidia_uvm/parameters/uvm_perf_access_counter_migration_enable']:
                log.write('\n' + name + '\n')
                try: log.write(Path(name).read_text())
                except OSError as e: log.write(str(e) + '\n')
        binary = out / 'diagnostic'
        toolkit = Path(nvcc).resolve().parents[1]
        print('Compiling diagnostic copy of the existing kernels...', flush=True)
        with (out / 'build.log').open('w') as log:
            sp.run([nvcc, '-std=c++17', '-O3', '-lineinfo', '-arch=sm_90a',
                    str(out / 'diagnostic.cu'), '-L' + str(toolkit / 'lib64/stubs'),
                    '-lcuda', '-o', str(binary)], stdout=log, stderr=sp.STDOUT, check=True)
        for n in sizes:
            for h, w in tiles:
                for mode, source in modes:
                    i = len(cases)
                    name = f'{i+1:03d}_{n}_{h}x{w}_{mode}'
                    folder = out / name
                    folder.mkdir()
                    env = dict(os.environ, MALLOC_DIAG_MODE=mode)
                    cmd = [str(binary), str(n), str(h), str(w), source, '128', str(i % 3), str(folder / 'rows.csv')]
                    if mode == 'taskbind': cmd = ['numactl', '--membind=0'] + cmd
                    (folder / 'command.json').write_text(json.dumps(cmd))
                    print(f'Progress: {i}/{expected} | {n//(1<<20)} MiB tile={h}x{w} mode={mode}', flush=True)
                    with (folder / 'worker.log').open('w') as log:
                        active = sp.Popen(cmd, env=env, stdout=log, stderr=sp.STDOUT, start_new_session=True)
                        try:
                            rc = active.wait(timeout=180)
                        except sp.TimeoutExpired:
                            os.killpg(active.pid, signal.SIGKILL); active.wait(); rc = 124
                        active = None
                    cases.append(dict(case=name, mode=mode, returncode=rc))
                    summarize(out, cases, expected)
                    if rc == 3 or rc == 124 or rc < 0:
                        raise RuntimeError(f'GPU fault, timeout or signal in {name}; stopping safely')
        print(f'Progress: {expected}/{expected} COMPLETE', flush=True)
    finally:
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGKILL)
            active.wait()
        summarize(out, cases, expected)
        archive = out.with_suffix('.tar.gz')
        with tarfile.open(archive, 'w:gz') as tar:
            for p in out.rglob('*'):
                if p.is_file() and p != out / 'diagnostic':
                    tar.add(p, arcname=str(Path(out.name) / p.relative_to(out)))
        print((out / 'summary.md').read_text(), flush=True)
        print(f'Outputs: {out}\nSEND_THIS_FILE: {archive}', flush=True)


if __name__ == '__main__':
    main()
