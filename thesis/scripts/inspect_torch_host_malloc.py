#!/usr/bin/env python3
"""Read-only inspection of the active environment; optional JSON output."""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from torch_host_malloc_common import capture, cuda_home, digest, write_json


def inspect(explicit_cuda=None):
    import torch
    result = dict(python=sys.executable, prefix=sys.prefix, python_version=sys.version,
                  machine=platform.machine(), platform=platform.platform(),
                  torch_version=torch.__version__, torch_commit=torch.version.git_version,
                  torch_path=torch.__file__, cuda_build=torch.version.cuda,
                  torch_config=torch.__config__.show(),
                  cxx11_abi=getattr(torch._C, '_GLIBCXX_USE_CXX11_ABI', None),
                  allocator_environment={k: v for k, v in os.environ.items() if
                    k.startswith(('THESIS_HOST_MALLOC', 'PYTORCH_'))},
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    tracked = [Path(torch.__file__), Path(torch.version.__file__), Path(torch._C.__file__)]
    result['original_files_sha256'] = {str(path.resolve()): digest(path) for path in tracked}
    toolkit = cuda_home(explicit_cuda)
    result['cuda_home'] = str(toolkit)
    commands = {'nvcc': [toolkit/'bin/nvcc', '--version'],
                'compiler': [os.environ.get('CXX', 'g++'), '--version'],
                'gpu': ['nvidia-smi', '-q'], 'numa': ['numactl', '--hardware']}
    for key, cmd in commands.items():
        try:
            result[key] = capture(cmd)
        except (OSError, RuntimeError, subprocess.CalledProcessError) as e:
            result[key] = {'error': str(e)}
    for name in ['/proc/sys/kernel/numa_balancing',
                 '/sys/module/nvidia_uvm/parameters/uvm_perf_access_counter_migration_enable']:
        try:
            result[name] = Path(name).read_text().strip()
        except OSError as e:
            result[name] = str(e)
    result['cuda_available'] = torch.cuda.is_available()
    if result['cuda_available']:
        result['gpu_name'] = torch.cuda.get_device_name(0)
        result['capability'] = torch.cuda.get_device_capability(0)
        result['allocator_backend'] = torch.cuda.get_allocator_backend()
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    p.add_argument('--cuda-home')
    args = p.parse_args()
    result = inspect(args.cuda_home)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))
