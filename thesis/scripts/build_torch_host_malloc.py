#!/usr/bin/env python3
"""Build stock and opt-in patched PyTorch in isolated prefixes; never pip-install into fa3."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess as sp
import sys
import tempfile
import time
from torch_host_malloc_common import (ALLOCATOR_BLOB, PACKAGE, PATCH, REPO,
    TORCH_COMMIT, TORCH_REF, capture, clean_environment, conda_executable,
    cuda_home, default_root, digest, progress, runtime_environment, write_json)
from torch_host_malloc_disk import DiskSpaceError, run_guarded


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=default_root())
    p.add_argument('--cuda-home')
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--min-free-gib', type=int, default=30)
    p.add_argument('--reserve-free-gib', type=int, default=8,
                   help='Stop this build when free space falls below this reserve')
    p.add_argument('--plan', action='store_true')
    args = p.parse_args()
    if args.jobs < 1 or args.reserve_free_gib < 1 or args.min_free_gib <= args.reserve_free_gib:
        p.error('Require jobs>=1 and min-free-gib > reserve-free-gib >=1')
    root = args.root.expanduser().resolve()
    original_prefix = Path(sys.prefix).resolve()
    if root == REPO or REPO in root.parents or root == original_prefix or original_prefix in root.parents:
        p.error('build root must be outside this repo and outside the active environment')
    print('External build root:', root)
    print('Pinned upstream:', TORCH_REF, TORCH_COMMIT)
    print('Environments: build_env, stock_env, fa3_host_malloc (all under build root)')
    print(f'Disk policy: start with >= {args.min_free_gib} GiB free; stop below {args.reserve_free_gib} GiB free')
    if args.plan:
        return 0
    if platform.system() != 'Linux' or platform.machine() != 'aarch64':
        raise RuntimeError('Build on the Linux aarch64 Grace2 host with fa3 active')
    if os.environ.get('THESIS_HOST_MALLOC') not in (None, '0'):
        raise RuntimeError('Start the build from the original fa3 environment, without host-malloc enabled')
    if os.environ.get('LD_PRELOAD'):
        raise RuntimeError('Unset LD_PRELOAD before building the isolated experiment')
    from inspect_torch_host_malloc import inspect
    original = inspect(args.cuda_home)
    if not Path(original['torch_path']).resolve().is_relative_to(original_prefix):
        raise RuntimeError('Active torch is not installed inside the active environment; activate original fa3')
    if not original['cuda_available'] or 'GH200' not in original.get('gpu_name', ''):
        raise RuntimeError('Original environment cannot access the GH200')
    toolkit = cuda_home(args.cuda_home)
    conda = conda_executable()
    owner = dict(schema=1, upstream=TORCH_COMMIT, patch_sha256=digest(PATCH),
                 helper_sha256=digest(PACKAGE/'ThesisHostMalloc.h'),
                 original_prefix=str(original_prefix), cuda_home=str(toolkit), jobs=args.jobs,
                 python_version=f'{sys.version_info.major}.{sys.version_info.minor}')
    if root.exists() and any(root.iterdir()):
        if not (root/'owner.json').is_file() or json.loads((root/'owner.json').read_text()) != owner:
            raise RuntimeError('Existing build root has different ownership/configuration. Choose a fresh --root')
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'build.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_json(root/'owner.json', owner)
    write_json(root/'original_environment.json', original)
    if shutil.disk_usage(root).free < args.min_free_gib * (1 << 30):
        raise RuntimeError(f'Need at least {args.min_free_gib} GiB free in build root filesystem')
    env = runtime_environment(toolkit)
    env.update(CUDA_HOME=str(toolkit), CUDA_PATH=str(toolkit),
        PATH=str(toolkit/'bin')+os.pathsep+env['PATH'],
        USE_CUDA='1', USE_CUDNN='0', USE_DISTRIBUTED='0', USE_NCCL='0',
        USE_MPI='0', USE_MAGMA='0', USE_MKLDNN='0', USE_KINETO='1',
        BUILD_TEST='0', TORCH_CUDA_ARCH_LIST='9.0', MAX_JOBS=str(args.jobs),
        CMAKE_BUILD_PARALLEL_LEVEL=str(args.jobs), PYTORCH_BUILD_VERSION='2.11.0+thesis',
        PYTORCH_BUILD_NUMBER='1')
    source = root/'pytorch'
    log = root/'build.log'
    disk_policy = dict(entry_free_gib=args.min_free_gib, reserve_free_gib=args.reserve_free_gib,
                       poll_seconds=0.5)
    status = dict(state='building', started=dt.datetime.now(dt.timezone.utc).isoformat(),
                  disk_policy=disk_policy, steps=[])
    write_json(root/'build_status.json', status)
    count = 0
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    previous = {}
    for sig in [signal.SIGTERM, signal.SIGHUP]:
        old = signal.getsignal(sig)
        if sig == signal.SIGHUP and old == signal.SIG_IGN:
            continue  # Preserve nohup's disconnect protection.
        previous[sig] = old
        signal.signal(sig, interrupt)

    def run(argv, cwd=root, run_env=env):
        nonlocal count
        count += 1
        print(f'Build command {count}: {argv[0]} ... (log: {log})', flush=True)
        with log.open('a') as f:
            f.write('\nCOMMAND '+json.dumps([str(x) for x in argv])+'\n'); f.flush()
            run_guarded(argv, cwd=cwd, env=run_env, log=f, root=root,
                        reserve_bytes=args.reserve_free_gib*(1 << 30))
        status['steps'].append([str(x) for x in argv])
        write_json(root/'build_status.json', status)

    try:
        progress(0, 6, 'prepare source')
        if not source.exists():
            run(['git', 'clone', '--branch', TORCH_REF, '--depth', '1',
                 'https://github.com/pytorch/pytorch.git', source])
        if capture(['git', 'rev-parse', 'HEAD'], cwd=source).strip() != TORCH_COMMIT:
            raise RuntimeError('Source commit differs from audited pinned commit')
        changed = set(capture(['git', 'diff', '--name-only'], cwd=source).splitlines())
        untracked = set(capture(['git', 'ls-files', '--others', '--exclude-standard'], cwd=source).splitlines())
        if changed - {'c10/cuda/CUDACachingAllocator.cpp'} or untracked - {'c10/cuda/ThesisHostMalloc.h'}:
            raise RuntimeError('Source tree contains unrelated changes; refusing to alter it')
        patched = (source/'c10/cuda/ThesisHostMalloc.h').exists()
        if patched:
            run(['git', 'apply', '--reverse', '--check', PATCH], cwd=source)
            if digest(source/'c10/cuda/ThesisHostMalloc.h') != owner['helper_sha256']:
                raise RuntimeError('Patched header mismatch')
            # Reverse --check alone would overlook unrelated edits outside
            # the patch hunks in this same cpp file. Compare the entire file.
            with tempfile.TemporaryDirectory() as temp:
                expected = Path(temp)/'c10/cuda/CUDACachingAllocator.cpp'
                expected.parent.mkdir(parents=True)
                expected.write_text(capture(['git','show',TORCH_COMMIT+':c10/cuda/CUDACachingAllocator.cpp'],cwd=source))
                run(['git','apply',PATCH],cwd=Path(temp))
                if digest(expected) != digest(source/'c10/cuda/CUDACachingAllocator.cpp'):
                    raise RuntimeError('Unrelated changes inside the patched allocator cpp')
        elif changed:
            raise RuntimeError('Unknown allocator modifications in source tree')
        else:
            blob = capture(['git', 'hash-object', 'c10/cuda/CUDACachingAllocator.cpp'], cwd=source).strip()
            if blob != ALLOCATOR_BLOB:
                raise RuntimeError('Audited allocator source blob mismatch')
            run(['git', 'apply', '--check', PATCH], cwd=source)
        run(['git', 'submodule', 'sync', '--recursive'], cwd=source)
        run(['git', 'submodule', 'update', '--init', '--recursive', '--depth', '1'], cwd=source)
        progress(1, 6, 'prepare isolated environments')
        for name in ['build_env', 'stock_env', 'fa3_host_malloc']:
            prefix = root/name
            if not prefix.exists():
                run([conda, 'create', '-y', '--override-channels', '-c', 'conda-forge', '--prefix', prefix,
                     'python='+owner['python_version'], 'pip'])
            if not (prefix/'conda-meta/history').is_file():
                raise RuntimeError('Incomplete or non-conda prefix: '+str(prefix))
        build_python = root/'build_env/bin/python'
        env.update(CONDA_PREFIX=str(root/'build_env'), CONDA_DEFAULT_ENV=str(root/'build_env'),
                   CMAKE_PREFIX_PATH=str(root/'build_env'),
                   LD_LIBRARY_PATH=str(root/'build_env/lib')+os.pathsep+env['LD_LIBRARY_PATH'],
                   PATH=str(root/'build_env/bin')+os.pathsep+env['PATH'])
        # Use the pinned source's build requirements, with no build isolation.
        run([build_python, '-m', 'pip', 'install', '-r', source/'requirements-build.txt', 'wheel'])
        runtime = ['filelock', 'typing_extensions>=4.10', 'sympy>=1.13.3',
                   'networkx', 'jinja2', 'fsspec', 'numpy', 'packaging', 'psutil']
        for name in ['stock_env', 'fa3_host_malloc']:
            run([root/name/'bin/python', '-m', 'pip', 'install', *runtime])
        wheels = {}
        for index, (mode, name) in enumerate([('stock', 'stock_env'), ('host', 'fa3_host_malloc')], 2):
            dest = root/'wheels'/mode
            dest.mkdir(parents=True, exist_ok=True)
            existing = list(dest.glob('torch-*.whl'))
            checkpoint = dest/'complete.json'
            checkpoint_data = json.loads(checkpoint.read_text()) if checkpoint.is_file() else {}
            valid = (len(existing) == 1 and checkpoint_data.get('sha256') == digest(existing[0]) and
                     checkpoint_data.get('mode') == mode and checkpoint_data.get('upstream') == TORCH_COMMIT)
            if not valid:
                if existing or checkpoint.exists():
                    raise RuntimeError('Incomplete wheel output; inspect it and choose a fresh --root')
                if mode == 'stock' and patched:
                    run(['git', 'apply', '--reverse', PATCH], cwd=source); patched = False
                if mode == 'host' and not patched:
                    run(['git', 'apply', PATCH], cwd=source); patched = True
                if mode == 'host' and digest(source/'c10/cuda/ThesisHostMalloc.h') != owner['helper_sha256']:
                    raise RuntimeError('Patch header differs from the reviewed helper')
                run([build_python, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
                     '--wheel-dir', dest, '.'], cwd=source)
                existing = list(dest.glob('torch-*.whl'))
                if len(existing) != 1:
                    raise RuntimeError('Expected exactly one torch wheel')
                write_json(checkpoint, dict(sha256=digest(existing[0]), mode=mode, upstream=TORCH_COMMIT))
            run([root/name/'bin/python', '-m', 'pip', 'install', '--no-deps', '--force-reinstall', existing[0]])
            wheels[mode] = dict(path=str(existing[0]), sha256=digest(existing[0]), python=str(root/name/'bin/python'))
            progress(index+1, 6, mode+' wheel installed')
        progress(4, 6, 'verify installations with feature OFF')
        for mode, wheel in wheels.items():
            py = Path(wheel['python'])
            verify_env=dict(env,LD_LIBRARY_PATH=str(py.parent.parent/'lib')+os.pathsep+runtime_environment(toolkit)['LD_LIBRARY_PATH'])
            result = json.loads(capture([py, '-c',
                'import torch,json,sys; print(json.dumps(dict(prefix=sys.prefix,version=torch.__version__,commit=torch.version.git_version,cuda=torch.cuda.is_available(),path=torch.__file__)))'],
                env=verify_env, cwd=root).strip().splitlines()[-1])
            if not result['cuda'] or result['commit'] != TORCH_COMMIT or Path(result['prefix']).resolve() == original_prefix:
                raise RuntimeError('Installed wheel verification failed: '+str(result))
            write_json(root/(mode+'_installation.json'), result)
            (root/(mode+'_pip_freeze.txt')).write_text(capture([py, '-m', 'pip', 'freeze'], env=verify_env, cwd=root))
        progress(5, 6, 'record build provenance')
        for path, before in original['original_files_sha256'].items():
            if digest(path) != before:
                raise RuntimeError('Original environment file changed during build: '+path)
        status['original_fingerprint_unchanged'] = True
        write_json(root/'manifest.json', dict(**owner, wheels=wheels, build_environment={k:env[k] for k in
            ['USE_CUDA','USE_CUDNN','USE_DISTRIBUTED','USE_NCCL','USE_MPI','USE_MAGMA','USE_MKLDNN','USE_KINETO',
             'BUILD_TEST','TORCH_CUDA_ARCH_LIST','MAX_JOBS','CMAKE_BUILD_PARALLEL_LEVEL','PYTORCH_BUILD_VERSION']},
            disk_policy=disk_policy, created=dt.datetime.now(dt.timezone.utc).isoformat()))
        status['state'] = 'complete'
        progress(6, 6, 'build complete; GH200 runtime validation still required')
        print('Run: bash thesis/scripts/run_torch_host_malloc.sh --root '+str(root))
    except BaseException as e:
        status.update(state='stopped_low_disk' if isinstance(e, DiskSpaceError) else 'failed', error=repr(e))
        raise
    finally:
        write_json(root/'build_status.json', status)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        lock.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
