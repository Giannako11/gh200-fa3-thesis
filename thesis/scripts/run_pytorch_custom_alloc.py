"""Build the small Grace allocator library and validate selective placement on GH200."""

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

from torch_host_malloc_common import (
    REPO,
    TORCH_COMMIT,
    capture,
    cuda_home,
    default_root,
    digest,
    progress,
    runtime_environment,
    write_json,
)

THESIS = REPO / "thesis"
PACKAGE = THESIS / "pytorch_custom_alloc"
WORKER = THESIS / "tests/test_pytorch_custom_alloc.py"


def run_logged(command, log, env):
    with log.open("w") as output:
        subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True,
                       env=env, timeout=180)


def validate(root, cuda, python, output):
    library = output / "grace_allocator.so"
    source = PACKAGE / "grace_allocator.cpp"
    env = runtime_environment(cuda)
    compile_cmd = [
        str(cuda / "bin/nvcc"), "-std=c++17", "-O2", "--shared", "--cudart=shared",
        "-Xcompiler=-fPIC,-Wall,-Wextra,-pthread", str(source), "-o", str(library),
    ]
    worker_cmd = [
        shutil.which("numactl"), "--cpunodebind=0", "--membind=0", str(python), "-u",
        str(WORKER), str(library), str(output / "result.json"),
    ]
    files = [source, PACKAGE / "grace_pool.py", WORKER, Path(__file__),
             THESIS / "scripts/run_pytorch_custom_alloc.sh"]
    write_json(output / "manifest.json", {
        "stock_root": str(root),
        "stock_torch_commit": TORCH_COMMIT,
        "stock_build_manifest_sha256": digest(root / "manifest.json"),
        "cuda_home": str(cuda),
        "git_commit": capture(["git", "rev-parse", "HEAD"], cwd=REPO).strip(),
        "files_sha256": {str(path.relative_to(REPO)): digest(path) for path in files},
        "compile_command": compile_cmd,
        "worker_command": worker_cmd,
    })

    run_logged([shutil.which("nvidia-smi")], output / "gpu.log", env)
    progress(1, 3, "compile Grace allocator")
    run_logged(compile_cmd, output / "compile.log", env)
    progress(2, 3, "run GH200 placement, GPU correctness and release checks")
    run_logged(worker_cmd, output / "worker.log", env)
    result = json.loads((output / "result.json").read_text())
    if result["status"] != "PASS":
        raise RuntimeError("Worker did not report PASS")
    progress(3, 3, "PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root(),
                        help="step 2 build root containing stock_env")
    parser.add_argument("--cuda-home", type=Path, help="override the step 2 CUDA Toolkit")
    parser.add_argument("--plan", action="store_true", help="show the steps without running")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    print("Stock PyTorch environment:", root / "stock_env")
    if args.plan:
        print("1. Build grace_allocator.so with the CUDA Toolkit.")
        print("2. Verify HBM and Grace placement, mixed CUDA arithmetic and pool release.")
        print("3. Save run manifest, logs and results under thesis/outputs/.")
        return 0

    output = None
    try:
        if os.environ.get("LD_PRELOAD"):
            raise RuntimeError("unset LD_PRELOAD for this isolated test")
        build = json.loads((root / "manifest.json").read_text())
        status = json.loads((root / "build_status.json").read_text())
        if status["state"] != "complete" or build["upstream"] != TORCH_COMMIT:
            raise RuntimeError("expected a completed step 2 source build")
        python = Path(build["wheels"]["stock"]["python"]).resolve(strict=True)
        if python.parent.parent != (root / "stock_env").resolve(strict=True):
            raise RuntimeError("stock Python is outside the step 2 stock_env")
        cuda = cuda_home(args.cuda_home or build["cuda_home"])

        for command in ("numactl", "nvidia-smi"):
            if not shutil.which(command):
                raise RuntimeError(f"missing command: {command}")
        outputs = THESIS / "outputs"
        outputs.mkdir(exist_ok=True)
        # Share the lock used by the TMA and step 2 runners on Grace2.
        with (outputs / ".tma_sweep.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pids = capture(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
            if any(line.strip().isdigit() for line in pids.splitlines()):
                raise RuntimeError("GPU compute process present; run on an idle GH200")

            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output = Path(tempfile.mkdtemp(prefix=f"torch_custom_alloc_{stamp}_", dir=outputs))
            print("Outputs:", output, flush=True)
            # Fatal callback errors are logged to stderr. Avoid generating CUDA
            # worker core dumps on the shared filesystem.
            _, core_max = resource.getrlimit(resource.RLIMIT_CORE)
            resource.setrlimit(resource.RLIMIT_CORE, (0, core_max))
            validate(root, cuda, python, output)
    except (OSError, RuntimeError, LookupError, ValueError, subprocess.SubprocessError,
            KeyboardInterrupt) as error:
        if output:
            write_json(output / "status.json", {"status": "FAIL", "error": str(error)})
        print(f"Step 3 run failed: {error}", file=sys.stderr)
        return 1

    write_json(output / "status.json", {"status": "PASS"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
