"""Run block-sparse attention with HBM and Grace DRAM K/V on GH200."""

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile

from run_pytorch_custom_alloc import run_logged
from torch_host_malloc_common import (
    REPO, TORCH_COMMIT, capture, cuda_home, default_root, digest, progress,
    runtime_environment, write_json,
)

THESIS = REPO / "thesis"
BENCH = THESIS / "microbenchmarks/sparse_attention_placement"
ALLOC = THESIS / "pytorch_custom_alloc"


def plot_worker(candidates):
    for python in dict.fromkeys(candidates):
        if python.is_file() and subprocess.run(
            [str(python), "-c", "import matplotlib"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
        ).returncode == 0:
            return python
    return None


def run(args, root, build, python, cuda, out):
    env = runtime_environment(cuda)
    nvcc = str(cuda / "bin/nvcc")
    flags = [nvcc, "-std=c++17", "-O3", "--shared", "--cudart=shared",
             "-Xcompiler=-fPIC,-pthread"]
    allocator = out / "grace_allocator.so"
    kernel = out / "block_sparse_decode.so"
    sources = [ALLOC / "grace_allocator.cpp", BENCH / "block_sparse_decode.cu"]
    compile_allocator = [*flags, str(sources[0]), "-o", str(allocator)]
    compile_kernel = [*flags, "-arch=sm_90", str(sources[1]), "-o", str(kernel)]
    worker = BENCH / "bench_sparse_attention_placement.py"
    command = ["numactl", "--cpunodebind=0", "--membind=0", str(python), "-u",
               str(worker), "--allocator", str(allocator), "--kernel", str(kernel),
               "--out", str(out)]
    if args.smoke:
        command.append("--smoke")
    files = [*sources, ALLOC / "grace_pool.py", worker, BENCH / "plot.py",
             Path(__file__), THESIS / "scripts/run_sparse_attention_placement.sh"]
    write_json(out / "manifest.json", {
        "git_commit": capture(["git", "rev-parse", "HEAD"], cwd=REPO).strip(),
        "files_sha256": {str(path.relative_to(REPO)): digest(path) for path in files},
        "stock_root": str(root),
        "stock_build_manifest_sha256": digest(root / "manifest.json"),
        "cuda_home": str(cuda),
        "compile_commands": [compile_allocator, compile_kernel],
        "worker_command": command,
    })

    run_logged(["nvidia-smi"], out / "gpu.log", env)
    progress(1, 3, "compile allocator and decode kernel")
    run_logged(compile_allocator, out / "compile_allocator.log", env)
    run_logged(compile_kernel, out / "compile_kernel.log", env)
    progress(2, 3, "validate and benchmark both K/V placements")
    run_logged(command, out / "worker.log", env)
    if json.loads((out / "result.json").read_text())["status"] != "PASS":
        raise RuntimeError("benchmark worker did not report PASS")

    plotting_python = plot_worker([
        Path(sys.executable), Path(build["original_prefix"]) / "bin/python", python,
    ])
    if plotting_python:
        run_logged([str(plotting_python), str(BENCH / "plot.py"), str(out)],
                   out / "plot.log", os.environ)
        progress(3, 3, "PASS; both plots saved")
        return "PASS"

    print("Matplotlib unavailable; run --plots-only from an environment with matplotlib.",
          flush=True)
    progress(3, 3, "PASS benchmark; plot generation pending")
    return "pending"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--cuda-home", type=Path)
    parser.add_argument("--smoke", action="store_true", help="only test 16/64 blocks")
    parser.add_argument("--plan", action="store_true", help="show the steps without GPU work")
    parser.add_argument("--plots-only", type=Path, metavar="RUN_DIR",
                        help="regenerate both plots without CUDA")
    args = parser.parse_args()
    if args.plots_only:
        plot_script = BENCH / "plot.py"
        return subprocess.call([sys.executable, str(plot_script), str(args.plots_only)])
    root = args.root.expanduser().resolve()
    print("Stock PyTorch environment:", root / "stock_env")
    if args.plan:
        print("Compile Grace allocator and direct-read block-sparse CUDA decode.")
        print("Compare both K/V placements for 16/64 blocks." if args.smoke
              else "Compare both K/V placements for 8, 16, 32 and 64 blocks.")
        print("Write latency samples, HBM deltas, validation and plots.")
        return 0

    out = None
    try:
        if os.environ.get("LD_PRELOAD"):
            raise RuntimeError("unset LD_PRELOAD for the isolated benchmark")
        build = json.loads((root / "manifest.json").read_text())
        build_status = json.loads((root / "build_status.json").read_text())
        if build_status["state"] != "complete" or build["upstream"] != TORCH_COMMIT:
            raise RuntimeError("expected a completed step 2 stock PyTorch build")
        python = Path(build["wheels"]["stock"]["python"]).resolve(strict=True)
        if python.parent.parent != (root / "stock_env").resolve(strict=True):
            raise RuntimeError("stock Python resolved outside stock_env")
        cuda = cuda_home(args.cuda_home or build["cuda_home"])
        for command in ("numactl", "nvidia-smi"):
            if not shutil.which(command):
                raise RuntimeError(f"missing command: {command}")
        outputs = THESIS / "outputs"
        outputs.mkdir(exist_ok=True)
        with (outputs / ".tma_sweep.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pids = capture(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
            if any(line.strip().isdigit() for line in pids.splitlines()):
                raise RuntimeError("GPU compute process present; run on an idle GH200")
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out = Path(tempfile.mkdtemp(prefix=f"sparse_attention_placement_{stamp}_", dir=outputs))
            print("Outputs:", out, flush=True)
            _, core_max = resource.getrlimit(resource.RLIMIT_CORE)
            resource.setrlimit(resource.RLIMIT_CORE, (0, core_max))
            plots = run(args, root, build, python, cuda, out)
        write_json(out / "status.json", {"status": "PASS", "plots": plots})
        return 0
    except (OSError, RuntimeError, LookupError, ValueError, subprocess.SubprocessError,
            KeyboardInterrupt) as error:
        if out:
            write_json(out / "status.json", {"status": "FAIL", "error": str(error)})
        print(f"Sparse attention placement run failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
