"""Validate and time direct reads of block-selected K/V on one GH200."""

import argparse
import csv
import ctypes
import os
from pathlib import Path
import random
import statistics
import sys

THESIS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(THESIS / "scripts"))
sys.path.insert(0, str(THESIS / "pytorch_custom_alloc"))
from torch_host_malloc_common import TORCH_COMMIT, write_json

HEADS, TOKENS, DIM, BLOCK = 8, 8192, 128, 128
TOTAL_BLOCKS = TOKENS // BLOCK
WARMUPS, SAMPLES = 10, 30
SEED = 20261115
PLACEMENTS = ("hbm", "grace")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Decode:
    def __init__(self, library):
        from torch import cuda, empty, float32

        self.library = ctypes.CDLL(str(library))
        self.launch = self.library.sparse_decode
        ptr = ctypes.c_void_p
        self.launch.argtypes = (
            (ptr,) * 4 + (ctypes.c_int,) * 3 + (ptr,) * 5
            + (ctypes.POINTER(ctypes.c_float),)
        )
        self.launch.restype = ctypes.c_int
        self.scratch = (
            empty((HEADS, TOTAL_BLOCKS), dtype=float32, device="cuda"),
            empty((HEADS, TOTAL_BLOCKS), dtype=float32, device="cuda"),
            empty((HEADS, TOTAL_BLOCKS, DIM), dtype=float32, device="cuda"),
        )
        self.output = empty((HEADS, DIM), dtype=float32, device="cuda")
        self.stream = cuda.current_stream().cuda_stream

    def __call__(self, query, kv, blocks, *, timed=False):
        elapsed = ctypes.c_float()
        k, v = kv
        status = self.launch(
            query.data_ptr(), k.data_ptr(), v.data_ptr(), blocks.data_ptr(),
            HEADS, TOKENS, blocks.numel(), *(t.data_ptr() for t in self.scratch),
            self.output.data_ptr(), self.stream, ctypes.byref(elapsed) if timed else None,
        )
        require(status == 0, f"CUDA decode launch/timing failed with runtime status {status}")
        if timed:
            require(elapsed.value > 0, "non-positive CUDA event time")
            return elapsed.value * 1000
        return self.output


def reference(torch, query, kv, block_ids):
    token_ids = [token for block in block_ids for token in range(block * BLOCK, (block + 1) * BLOCK)]
    index = torch.tensor(token_ids, dtype=torch.int64, device="cuda")
    k, v = (tensor.index_select(1, index).float() for tensor in kv)
    logits = (k * query.float().unsqueeze(1)).sum(dim=2) * DIM ** -0.5
    return (v * logits.softmax(dim=1).unsqueeze(2)).sum(dim=1)


def paired_times(decode, query, kv, blocks):
    for _ in range(WARMUPS):
        for place in PLACEMENTS:
            decode(query, kv[place], blocks)

    rows = []
    for pair in range(SAMPLES):
        order = PLACEMENTS if pair % 2 == 0 else PLACEMENTS[::-1]
        for position, place in enumerate(order, start=1):
            rows.append({
                "selected_blocks": blocks.numel(),
                "pair": pair + 1,
                "position_in_pair": position,
                "kv_placement": place,
                "latency_us": decode(query, kv[place], blocks, timed=True),
            })
    return rows


def experiment(args, result):
    import torch
    from grace_pool import GracePool

    require(torch.version.git_version == TORCH_COMMIT, "expected the step 2 stock PyTorch build")
    require(torch.cuda.device_count() == 1 and "GH200" in torch.cuda.get_device_name(0),
            "requires one visible GH200")
    require(torch.cuda.get_allocator_backend() == "native", "requires PyTorch native CUDA allocator")
    require(os.environ.get("THESIS_HOST_MALLOC") in (None, "0"), "global host allocation is enabled")
    torch.cuda.set_device(0)
    dtype = torch.bfloat16
    pool = GracePool(args.allocator)
    shape = (HEADS, TOKENS, DIM)
    tensor_bytes = HEADS * TOKENS * DIM * 2
    kv_bytes = 2 * tensor_bytes
    expected_pages = tensor_bytes // os.sysconf("SC_PAGE_SIZE")
    result.update({
        "torch_version": torch.__version__, "torch_commit": torch.version.git_version,
        "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "seed": SEED, "queries_per_head": 1, "heads": HEADS, "head_dim": DIM,
        "tokens": TOKENS, "tokens_per_block": BLOCK, "kv_dtype": "bfloat16",
        "kv_bytes": kv_bytes, "warmups_per_placement": WARMUPS, "sample_pairs": SAMPLES,
        "cache_protocol": "warmed buffers, alternating order, no eviction",
    })

    with torch.inference_mode():
        decode = Decode(args.kernel)
        query = torch.empty((HEADS, DIM), dtype=dtype, device="cuda")
        query.uniform_(-1, 1, generator=torch.Generator(device="cuda").manual_seed(SEED))
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        before = torch.cuda.mem_get_info()[0]
        hbm = tuple(torch.empty(shape, dtype=dtype, device="cuda") for _ in range(2))
        after_hbm = torch.cuda.mem_get_info()[0]
        grace = tuple(pool.empty_on_grace(shape, dtype=dtype) for _ in range(2))
        after_grace = torch.cuda.mem_get_info()[0]
        hbm_delta = {"hbm": before - after_hbm, "grace": after_hbm - after_grace}
        pages_before = [pool.verified_pages(tensor) for tensor in grace]
        result.update(hbm_delta_bytes=hbm_delta, grace_pages_before=pages_before, grace_node=0)
        require(hbm_delta == {"hbm": kv_bytes, "grace": 0} and all(map(pool.is_hbm, hbm)),
                f"unexpected K/V HBM allocation deltas: {hbm_delta}")
        require(pages_before == [expected_pages] * 2, "K/V pages not verified on Grace")

        for location in (hbm, grace):
            generator = torch.Generator(device="cuda").manual_seed(SEED + 1)
            for tensor in location:
                tensor.uniform_(-1, 1, generator=generator)
            del tensor
        del location
        require(all(torch.equal(h, g) for h, g in zip(hbm, grace)),
                "K/V values differ across placements")
        result["matching_kv_inputs"] = True

        kv = {"hbm": hbm, "grace": grace}
        selection = list(range(TOTAL_BLOCKS))
        random.Random(SEED).shuffle(selection)
        selections = result["selected_block_ids"] = {}
        results = []
        samples = []
        for i, count in enumerate(args.counts, start=1):
            picked = sorted(selection[:count])
            selections[str(count)] = picked
            blocks = torch.tensor(picked, dtype=torch.int32, device="cuda")
            expected = reference(torch, query, hbm, picked)
            baseline = decode(query, hbm, blocks).clone()
            candidate = decode(query, grace, blocks)
            require(torch.equal(candidate, baseline), f"{count} blocks: outputs differ by placement")
            torch.testing.assert_close(baseline, expected, atol=2e-5, rtol=3e-4)
            largest_error = (baseline - expected).abs().max().item()

            case_samples = paired_times(decode, query, kv, blocks)
            samples.extend(case_samples)
            for place in PLACEMENTS:
                times = [s["latency_us"] for s in case_samples if s["kv_placement"] == place]
                p25, _, p75 = statistics.quantiles(times, n=4, method="inclusive")
                results.append({
                    "selected_blocks": count,
                    "selected_percent": 100 * count / TOTAL_BLOCKS,
                    "touched_kv_bytes": kv_bytes * count // TOTAL_BLOCKS,
                    "kv_placement": place,
                    "kv_bytes": kv_bytes,
                    "kv_hbm_delta_bytes": hbm_delta[place],
                    "median_us": statistics.median(times),
                    "p25_us": p25,
                    "p75_us": p75,
                    "samples": len(times),
                    "max_abs_error_vs_fp32": largest_error,
                })
            write_csv(args.out / "results.csv", results)
            write_csv(args.out / "samples.csv", samples)
            print(f"PASS [{i}/{len(args.counts)}] {count}/{TOTAL_BLOCKS} blocks", flush=True)

        pages_after = [pool.verified_pages(tensor) for tensor in grace]
        result["grace_pages_after"] = pages_after
        require(pages_after == pages_before, "Grace K/V placement changed during GPU work")
        torch.cuda.synchronize()
        del kv, grace
        pool.close()
        result["grace_live_bytes_after_close"] = pool.live_bytes()
        require(result["grace_live_bytes_after_close"] == 0, "Grace K/V allocation retained at teardown")

    lines = [
        "# GH200 block-selected decode attention placement", "",
        "- Status: PASS. Both placements matched each other and the FP32 reference.",
        f"- K/V physical HBM delta: {hbm_delta['hbm'] / 2**20:g} MiB in HBM, "
        f"{hbm_delta['grace'] / 2**20:g} MiB in Grace.",
        f"- Grace node 0 before and after: {expected_pages} pages per K/V tensor.",
        "- Timing uses CUDA events around the two decode kernels, warmed buffers, "
        "alternating order, and no explicit cache eviction.", "",
        "| Blocks read | HBM median (µs) | Grace median (µs) | Grace / HBM |",
        "|---:|---:|---:|---:|",
    ]
    for h, g in zip(results[::2], results[1::2]):
        lines.append(f"| {h['selected_blocks']}/{TOTAL_BLOCKS} | {h['median_us']:.3f} | "
                     f"{g['median_us']:.3f} | {g['median_us'] / h['median_us']:.3f}× |")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocator", type=Path, required=True)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="run only the 25% case")
    args = parser.parse_args()
    args.counts = (16,) if args.smoke else (8, 16, 32, 64)
    result = {"status": "FAIL"}
    try:
        experiment(args, result)
    except Exception as error:
        result["error"] = str(error)
        raise
    else:
        result["status"] = "PASS"
    finally:
        write_json(args.out / "result.json", result)


if __name__ == "__main__":
    main()
