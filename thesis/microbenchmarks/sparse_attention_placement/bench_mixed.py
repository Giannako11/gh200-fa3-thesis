"""Compare mixed K/V placement at a fixed block-sparse decode workload."""

import argparse
import os
from pathlib import Path
import random
import statistics

from bench_sparse_attention_placement import (
    BLOCK, DIM, HEADS, SEED, TORCH_COMMIT, WARMUPS, Decode, reference, require,
    write_csv, write_json,
)

CACHE_BLOCKS = 128
ACTIVE_BLOCKS = 64
TOKENS = CACHE_BLOCKS * BLOCK
BLOCK_BYTES = 2 * HEADS * BLOCK * DIM * 2  # K and V, two bytes per BF16
MIB = 2**20
SAMPLES_PER_PASS = 20

# Each row specifies total Grace-resident blocks and the subset read by decode.
POLICIES = {
    "all_hbm": (0, 0),
    "balanced_25": (32, 16),
    "balanced_50": (64, 32),
    "balanced_75": (96, 48),
    "all_grace": (128, 64),
    "active_hbm": (64, 0),
    "active_grace": (64, 64),
}


def layout(selected, inactive, grace_count, grace_active):
    grace_ids = set(selected[:grace_active] + inactive[:grace_count - grace_active])
    return ([block for block in range(CACHE_BLOCKS) if block not in grace_ids],
            sorted(grace_ids))


def block_addresses(shards, layout_ids, selected):
    """Pack K and V address tables in head-major, selected-block order."""
    readers = {
        logical_id: (shard, slot)
        for shard, ids in zip(shards, layout_ids)
        for slot, logical_id in enumerate(ids)
    }
    tables = [[], []]
    for head in range(HEADS):
        for logical_id in selected:
            shard, slot = readers[logical_id]
            require(shard is not None, "selected block has no backing storage")
            for part, table in enumerate(tables):
                offset = part * shard.stride(0) + head * shard.stride(1)
                offset += slot * BLOCK * shard.stride(2)
                table.append(shard.data_ptr() + offset * shard.element_size())
    return tables


def packed_copy(torch, source, shard, ids):
    if shard is None:
        return
    index = torch.tensor(ids, dtype=torch.int64, device="cuda")
    packed = source.view(2, HEADS, CACHE_BLOCKS, BLOCK, DIM).index_select(2, index)
    packed = packed.flatten(2, 3)
    # Use a CUDA pointwise kernel for the write to pageable Grace memory.
    torch.mul(packed, 1, out=shard)
    require(torch.equal(packed, shard), "stored K/V values differ from source")


def sample_policy(torch, pool, decode, source, query, selected, blocks, expected,
                  policy, layout_ids, pass_number, baseline):
    hbm_ids, grace_ids = layout_ids
    hbm_shape = (2, HEADS, len(hbm_ids) * BLOCK, DIM)
    grace_shape = (2, HEADS, len(grace_ids) * BLOCK, DIM)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = torch.cuda.mem_get_info()[0]
    hbm = torch.empty(hbm_shape, dtype=source.dtype, device="cuda") if hbm_ids else None
    after_hbm = torch.cuda.mem_get_info()[0]
    grace = pool.empty_on_grace(grace_shape, dtype=source.dtype) if grace_ids else None
    after_grace = torch.cuda.mem_get_info()[0]

    delta = before - after_hbm
    require(delta == len(hbm_ids) * BLOCK_BYTES and (hbm is None or pool.is_hbm(hbm)),
            f"{policy}: unexpected HBM K/V allocation delta {delta}")
    require(after_hbm == after_grace, f"{policy}: Grace allocation changed HBM use")
    pages = len(grace_ids) * BLOCK_BYTES // os.sysconf("SC_PAGE_SIZE")
    if grace is not None:
        require(pool.verified_pages(grace) == pages, f"{policy}: Grace placement verification failed")

    packed_copy(torch, source, hbm, hbm_ids)
    packed_copy(torch, source, grace, grace_ids)
    tables = tuple(torch.tensor(t, dtype=torch.int64, device="cuda")
                   for t in block_addresses((hbm, grace), layout_ids, selected))
    output = decode(query, tables, blocks)
    torch.testing.assert_close(output, expected, atol=2e-5, rtol=3e-4)
    error = (output - expected).abs().max().item()
    if baseline is None:
        baseline = output.clone()
    else:
        require(torch.equal(output, baseline), f"{policy}: output differs from all-HBM baseline")

    for _ in range(WARMUPS):
        decode(query, tables, blocks)
    times = [decode(query, tables, blocks, timed=True) for _ in range(SAMPLES_PER_PASS)]
    if grace is not None:
        require(pool.verified_pages(grace) == pages, f"{policy}: post-kernel placement changed")
    torch.cuda.synchronize()

    del grace
    pool.close()
    require(pool.live_bytes() == 0, f"{policy}: custom pool did not release all memory")
    samples = [{"policy": policy, "pass": pass_number, "sample": i, "latency_us": value}
               for i, value in enumerate(times, start=1)]
    return samples, {"hbm_bytes": delta, "grace_pages": pages,
                     "max_abs_error_vs_fp32": error, "live_bytes_after_close": 0}, baseline


def experiment(args, result):
    import torch
    from grace_pool import GracePool

    require(torch.version.git_version == TORCH_COMMIT, "expected the step 2 stock PyTorch build")
    require(torch.cuda.device_count() == 1 and "GH200" in torch.cuda.get_device_name(0),
            "requires one visible GH200")
    require(torch.cuda.get_allocator_backend() == "native", "requires PyTorch native CUDA allocator")
    require(os.environ.get("THESIS_HOST_MALLOC") in (None, "0"), "global host allocation is enabled")
    torch.cuda.set_device(0)

    order = list(range(CACHE_BLOCKS))
    random.Random(SEED).shuffle(order)
    active_order, inactive = order[:ACTIVE_BLOCKS], order[ACTIVE_BLOCKS:]
    selected = sorted(active_order)
    policies = ("all_hbm", "balanced_50", "all_grace") if args.smoke else tuple(POLICIES)
    # Reverse the order in the second pass so each policy's mean position is equal.
    sequences = (policies,) if args.smoke else (policies, tuple(reversed(policies)))
    layouts = {name: layout(active_order, inactive, *POLICIES[name]) for name in policies}
    checks = result["checks"] = {name: [] for name in policies}
    result.update(gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
                  torch_commit=torch.version.git_version, torch_cuda=torch.version.cuda,
                  cache_blocks=CACHE_BLOCKS, selected_block_ids=selected,
                  layout_ids=layouts, cache_bytes=CACHE_BLOCKS * BLOCK_BYTES, seed=SEED,
                  sequences=sequences, warmups_per_policy=WARMUPS,
                  samples_per_pass=SAMPLES_PER_PASS)

    samples = []
    with torch.inference_mode():
        decode = Decode(args.kernel, tokens=TOKENS, indirect=True)
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        query = torch.empty((HEADS, DIM), dtype=torch.bfloat16, device="cuda")
        source = torch.empty((2, HEADS, TOKENS, DIM), dtype=query.dtype, device="cuda")
        query.uniform_(-1, 1, generator=generator)
        source.uniform_(-1, 1, generator=generator)
        blocks = torch.tensor(selected, dtype=torch.int32, device="cuda")
        expected = reference(torch, query, (source[0], source[1]), selected)

        baseline = None
        for pass_number, sequence in enumerate(sequences, start=1):
            for i, name in enumerate(sequence, start=1):
                times, observation, baseline = sample_policy(
                    torch, GracePool(args.allocator), decode, source, query, selected, blocks,
                    expected, name, layouts[name], pass_number, baseline,
                )
                samples.extend(times)
                checks[name].append(observation)
                write_csv(args.out / "samples.csv", samples)
                print(f"PASS pass {pass_number}/{len(sequences)} [{i}/{len(sequence)}] {name}",
                      flush=True)

    rows = []
    for name in policies:
        grace_count, grace_active = POLICIES[name]
        times = [s["latency_us"] for s in samples if s["policy"] == name]
        p25, median, p75 = statistics.quantiles(times, n=4, method="inclusive")
        deltas = {c["hbm_bytes"] for c in checks[name]}
        require(len(deltas) == 1, f"{name}: HBM allocation differed across passes")
        delta = deltas.pop()
        rows.append({"policy": name, "grace_resident_blocks": grace_count,
                     "grace_active_blocks": grace_active, "active_blocks": ACTIVE_BLOCKS,
                     "kv_hbm_bytes": delta, "hbm_saved_bytes": CACHE_BLOCKS * BLOCK_BYTES - delta,
                     "median_us": median, "p25_us": p25, "p75_us": p75, "samples": len(times)})
    write_csv(args.out / "results.csv", rows)
    lines = [
        "# GH200 mixed K/V placement", "",
        "- Status: PASS. All layouts matched the FP32 reference and NUMA placement checks.",
        "- Fixed workload: 64 of 128 blocks read, 64 MiB K/V cache, one query per head.",
        "- GPU events cover the two decode kernels. Warmed buffers, no explicit eviction.",
        "- HBM savings refer to the experimental K/V allocation. The HBM reference "
        "source and scratch are outside the allocation deltas.", "",
        "| Policy | Grace resident | Grace read | HBM saved (MiB) | Median (µs) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['policy']} | {row['grace_resident_blocks']}/128 | "
                     f"{row['grace_active_blocks']}/64 | {row['hbm_saved_bytes'] / MIB:g} | "
                     f"{row['median_us']:.3f} |")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocator", type=Path, required=True)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = {"status": "FAIL", "mode": "mixed"}
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
