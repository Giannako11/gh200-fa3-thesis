"""One matrix multiply config on one backend, supervised by run.py."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from time import perf_counter

from grid import Case


class Timing:
    def __init__(self, device, stream):
        self.device = device
        self.stream = stream
        self.sample = None

    def emit(self, **fields):
        print(json.dumps({"device": self.device, "sample": self.sample, **fields}),
              file=self.stream, flush=True)

    @contextmanager
    def stage(self, name):
        self.emit(event="start", stage=name)
        start = perf_counter()
        try:
            yield
        except BaseException:
            self.emit(event="error", stage=name, seconds=perf_counter() - start)
            raise
        else:
            self.emit(event="done", stage=name, seconds=perf_counter() - start)


def matrix(np, rows, cols, pct, rng):
    """Generate a reproducible FP32 matrix with an exact requested nnz count."""
    if pct == 0:
        return np.zeros((rows, cols), dtype=np.float32)
    if pct == 100:
        return np.ones((rows, cols), dtype=np.float32)
    target = (rows * cols * pct + 50) // 100
    per_row, extra = divmod(target, rows)
    dense = np.zeros((rows, cols), dtype=np.float32)
    for row in range(rows):
        count = per_row + (row < extra)
        if count <= cols // 2:
            dense[row, rng.choice(cols, count, replace=False, shuffle=False)] = 1
        else:
            dense[row].fill(1)
            dense[row, rng.choice(cols, cols - count, replace=False, shuffle=False)] = 0
    return dense


def sparse_a(case):
    return case.strategy in {"csr_csr", "csr_dense"}


def sparse_b(case):
    return case.strategy in {"csr_csr", "dense_csr"}


def prepare_a(case, a, compress, contiguous):
    if case.strategy == "dense_csr":
        return contiguous(a.T)
    return compress(a) if sparse_a(case) else a


def prepare_b(case, b, compress):
    if case.strategy == "dense_csr":
        return compress(b.T)
    return compress(b) if sparse_b(case) else b


def prepare_reference(np, a, b):
    rows = np.linspace(0, a.shape[0] - 1, min(a.shape[0], 8), dtype=np.int64)
    cols = np.linspace(0, b.shape[1] - 1, min(b.shape[1], 8), dtype=np.int64)
    # At most K terms of value 0 or 1. This reference represents every sum exactly.
    return rows, cols, a[rows, :] @ b[:, cols]


def check_result(np, output, rows, cols, expected):
    sample = output[rows[:, None], cols[None, :]].cpu().numpy()
    if not np.array_equal(sample, expected):
        raise RuntimeError(f"reference mismatch: maximum sampled error {np.max(np.abs(sample - expected))}")


def allocation_check(torch, case, device):
    free, _ = torch.cuda.mem_get_info()
    a_bytes = case.m * case.k * 4
    b_bytes = case.k * case.n * 4
    c_bytes = case.m * case.n * 4
    # B and C persist in HBM even for the Grace path. Allow additional
    # output/CSR materialization on Hopper. Backend workspace is separate.
    required = b_bytes + c_bytes if device == "grace" else (
        a_bytes + b_bytes + c_bytes + (
            b_bytes + c_bytes if case.strategy == "csr_csr" else 0
        )
    )
    if free < required:
        raise MemoryError(f"relevant HBM buffers need {required / 2**30:.2f} GiB, "
                          f"with {free / 2**30:.2f} GiB available")
    # Guard the large input and temporary allocations on a shared host.
    # CPU library scratch can add to this estimate.
    available = next(line for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    host_free = int(available.split()[1]) * 1024
    host_required = 2 * (a_bytes + b_bytes + c_bytes) if device == "grace" else 2 * (a_bytes + b_bytes)
    if host_free < host_required:
        raise MemoryError(f"host budget estimate {host_required / 2**30:.2f} GiB, "
                          f"with {host_free / 2**30:.2f} GiB available")


def run(args, timing):
    case = Case(*args.case)
    with timing.stage("imports_and_device_init"):
        import numpy as np
        import torch

        if not torch.cuda.is_available() or "GH200" not in torch.cuda.get_device_name(0):
            raise RuntimeError("expected the GH200 CUDA environment")
        torch.set_num_threads(args.threads)
        torch.backends.cuda.matmul.allow_tf32 = False
        if args.device == "grace":
            from scipy.sparse import csr_matrix

        torch.cuda.init()
        allocation_check(torch, case, args.device)

    with timing.stage("generate_a"):
        a_host = matrix(np, case.m, case.k, case.a_pct, np.random.default_rng(711))
    with timing.stage("generate_b"):
        b_host = matrix(np, case.k, case.n, case.b_pct, np.random.default_rng(911))
    with timing.stage("prepare_reference"):
        rows, cols, expected = prepare_reference(np, a_host, b_host)

    hopper = args.device == "hopper"
    with timing.stage("prepare_resident_inputs"):
        b_gpu = torch.from_numpy(b_host).to("cuda")
        if hopper:
            # Preprocess a static weight on the host. This is setup, before
            # per-inference timing and the HBM-to-HBM measurement boundary.
            if sparse_a(case):
                from scipy.sparse import csr_matrix
                a = csr_matrix(a_host)
                a_gpu = torch.sparse_csr_tensor(
                    torch.from_numpy(a.indptr).to("cuda"),
                    torch.from_numpy(a.indices).to("cuda"),
                    torch.from_numpy(a.data).to("cuda"), size=a.shape, device="cuda",
                )
                del a
            else:
                a_gpu = prepare_a(case, torch.from_numpy(a_host).to("cuda"), None,
                                  lambda tensor: tensor.contiguous())
        else:
            a_cpu = prepare_a(case, a_host, csr_matrix, np.ascontiguousarray)
            activation_cpu = torch.empty_like(torch.from_numpy(b_host))
            output_gpu = torch.empty((case.m, case.n), device="cuda", dtype=torch.float32)
        torch.cuda.synchronize()
        del a_host, b_host

    with torch.inference_mode():
        for i in range(args.warmups + args.repeats):
            timing.sample = i - args.warmups + 1  # <=0 denotes warmup
            with timing.stage("end_to_end"):
                if hopper:
                    output = run_hopper(torch, case, a_gpu, b_gpu, timing)
                else:
                    output = run_grace(np, torch, case, a_cpu, b_gpu, activation_cpu,
                                       output_gpu, csr_matrix, timing)
            if i == 0 or i == args.warmups + args.repeats - 1:
                with timing.stage("validate"):
                    check_result(np, output, rows, cols, expected)
            del output
    timing.sample = None
    timing.emit(event="pass", stage="result")


def run_grace(np, torch, case, a, b_gpu, activation_cpu, output_gpu, compress, timing):
    with timing.stage("transfer_hbm_to_grace"):
        activation_cpu.copy_(b_gpu, non_blocking=False)
    with timing.stage("prepare_b"):
        b = prepare_b(case, activation_cpu.numpy(), compress)
    with timing.stage("compute"):
        # SpMM uses sparse on the left, including the transpose identity.
        c = b @ a if case.strategy == "dense_csr" else a @ b
    with timing.stage("materialize_output"):
        if case.strategy == "csr_csr":
            c = c.toarray()
        if case.strategy == "dense_csr":
            c = c.T
        source = torch.from_numpy(np.ascontiguousarray(c))
    with timing.stage("transfer_grace_to_hbm"):
        output_gpu.copy_(source, non_blocking=False)
        torch.cuda.synchronize()
    return output_gpu


def run_hopper(torch, case, a, b, timing):
    # Stage timings use synchronization so timeout logs identify a concrete
    # step. Record device compute separately from synchronized host wall time.
    with timing.stage("prepare_b"):
        b = prepare_b(case, b, lambda t: t.contiguous().to_sparse_csr())
        torch.cuda.synchronize()
    with timing.stage("compute"):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        if case.strategy == "dense_dense":
            c = torch.mm(a, b)
        elif case.strategy == "dense_csr":
            c = torch.sparse.mm(b, a)
        else:
            c = torch.sparse.mm(a, b)
        end.record()
        end.synchronize()
        timing.emit(event="device_time", stage="compute", seconds=start.elapsed_time(end) / 1000)
    with timing.stage("materialize_output"):
        if case.strategy == "csr_csr":
            c = c.to_dense()
        if case.strategy == "dense_csr":
            c = c.T.contiguous()
        torch.cuda.synchronize()
    return c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, nargs=5, required=True)
    parser.add_argument("--device", choices=("grace", "hopper"), required=True)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--steps", type=Path, required=True)
    args = parser.parse_args()
    with args.steps.open("w", buffering=1) as stream:
        run(args, Timing(args.device, stream))


if __name__ == "__main__":
    main()
