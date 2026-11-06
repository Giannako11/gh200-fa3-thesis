"""Shared orchestration helpers; importing this module never imports torch."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

TORCH_REF = 'v2.11.0'
TORCH_COMMIT = '70d99e998b4955e0049d13a98d77ae1b14db1f45'
ALLOCATOR_BLOB = '2ab4effc7853dc0ce62ebc60e6dbb8ced9b72cca'
REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / 'thesis/pytorch_host_malloc'
PATCH = PACKAGE / 'patches/native_host_malloc.patch'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def clean_environment():
    env = dict(os.environ)
    # Runtime experiments must not leak into build/baseline child processes.
    for key in list(env):
        if key.startswith(('THESIS_HOST_MALLOC', 'PYTORCH_', 'PIP_')) or key in {
            'PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE', 'CMAKE_ARGS', 'CMAKE_INSTALL_PREFIX'}:
            env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', PIP_CONFIG_FILE=os.devnull, PIP_USER='0')
    return env


def default_root():
    local = Path('/local') / os.environ.get('USER', 'unknown')
    return (local if local.is_dir() else Path.home()) / 'thesis_host_malloc'


def cuda_home(explicit=None):
    candidates = [explicit, os.environ.get('CUDA_HOME'), os.environ.get('CUDA_PATH')]
    prefix = os.environ.get('CONDA_PREFIX')
    if prefix:
        candidates.append(str(Path(prefix) / 'cuda-12.8.1'))
    nvcc = shutil.which('nvcc')
    if nvcc:
        candidates.append(str(Path(nvcc).resolve().parents[1]))
    candidates += ['/usr/local/cuda-12.8', '/usr/local/cuda']
    for candidate in candidates:
        if candidate and (Path(candidate) / 'bin/nvcc').is_file():
            return Path(candidate).resolve()
    raise RuntimeError('CUDA Toolkit not found: activate fa3 or supply --cuda-home')


def runtime_environment(cuda):
    env = clean_environment()
    paths = [cuda / 'lib64', cuda / 'targets/sbsa-linux/lib', cuda / 'targets/aarch64-linux/lib',
             cuda / 'extras/CUPTI/lib64', Path('/usr/local/nvidia/lib64'), Path('/usr/local/nvidia/lib')]
    # Never put toolkit stubs on the runtime library path.
    # Do not inherit another environment's libtorch or libcudart search paths.
    env['LD_LIBRARY_PATH'] = ':'.join(str(p) for p in paths if p.is_dir())
    return env


def capture(argv, env=None, cwd=None):
    return subprocess.check_output([str(x) for x in argv], text=True, env=env,
                                   cwd=cwd, stderr=subprocess.STDOUT)


def conda_executable():
    candidate = os.environ.get('CONDA_EXE') or shutil.which('conda')
    if not candidate:
        candidate = str(Path.home() / 'miniconda3/bin/conda')
    if not Path(candidate).is_file():
        raise RuntimeError('Conda not found; activate fa3 before building')
    return str(Path(candidate).resolve())


def progress(done, total, label):
    n = done * 30 // total
    print(f'Progress: [{"#"*n}{"-"*(30-n)}] {done}/{total} {label}', flush=True)
