# GH200 sparse attention K/V placement

One-token decode uses eight heads, head dimension 128, 8,192 K/V tokens and
BF16 inputs. A fixed seeded block order selects 8, 16, 32, or 64 of the 64
blocks of 128 tokens. Both placements use the same sorted block IDs and
the same Q, K and V values. Q, block IDs, output, and softmax scratch live
in Hopper HBM.

The baseline holds the entire 32 MiB K/V cache in Hopper HBM. The other
case holds the entire cache in Grace DRAM using the step 3 pool. The sparse
CUDA kernel reads selected K/V blocks directly. Each CTA computes a block's
FP32 softmax partial, and a second kernel merges the partials into the output.
The reference uses PyTorch FP32 on the HBM inputs and is run outside timing.
The CUDA path is a small placement experiment with a fixed block size.

## Run on Grace2

From the repository root on an idle GH200, using the stock build from step 2:

```bash
bash thesis/scripts/run_sparse_attention_placement.sh --smoke
bash thesis/scripts/run_sparse_attention_placement.sh
```

The smoke run measures 16 out of 64 blocks. The full run includes 12.5%, 25%,
50%, and 100%. Pass `--root` if the step 2 build uses a custom location.
The runner builds two small shared libraries in the output directory. It
uses the existing stock PyTorch environment and installs no packages.

## Outputs

Each run creates `thesis/outputs/sparse_attention_placement_*` containing:

- `result.json`: matched inputs and outputs, K/V HBM deltas, before/after
  Grace placement, block IDs, and release check.
- `samples.csv`, `results.csv`, and `summary.md`: raw paired samples, median
  and IQR by placement, maximum absolute reference error, and latency ratio.
- `attention_latency.png`: median and IQR of the two-kernel decode time vs.
  the fraction of blocks read, with a line for each K/V placement.
- `kv_hbm_usage.png`: physical HBM delta during creation of the fixed K/V
  allocation. It is independent of how many blocks are read later.
- `manifest.json`, `status.json`, and logs for reproducing or diagnosing the run.

Timing uses GPU events around the two kernels after 10 warmups per placement
and block count, followed by 30 HBM/Grace sample pairs in alternating order.
It describes warmed buffers and includes the partial merge. Allocations,
data initialization, validation, and plotting happen outside timing.

The plots use Matplotlib from the launching Python or an existing step 2
environment. If it is unavailable, keep the run folder and generate the
plots from any Python environment with Matplotlib and access to the files:

```bash
bash thesis/scripts/run_sparse_attention_placement.sh --plots-only /path/to/run
```

## Mixed K/V placement

```bash
bash thesis/scripts/run_sparse_attention_placement.sh --mixed --smoke
bash thesis/scripts/run_sparse_attention_placement.sh --mixed
```

The mixed experiment uses a 64 MiB K/V cache (16,384 tokens) and reads the
same 64 of 128 blocks for every policy. The two endpoints and all
intermediate placements use the same CUDA entry point. Per-block pointers
in HBM select the packed HBM or Grace shard directly. The kernel reads those
shards during attention, with no staging into HBM in the timed region.

| Policy | K/V resident in Grace | Active blocks read from Grace |
|---|---:|---:|
| All HBM | 0/128 | 0/64 |
| Balanced 25% | 32/128 | 16/64 |
| Balanced 50% | 64/128 | 32/64 |
| Balanced 75% | 96/128 | 48/64 |
| All Grace | 128/128 | 64/64 |
| Active blocks in HBM | 64/128 | 0/64 |
| Active blocks in Grace | 64/128 | 64/64 |

The full run uses two passes with reversed policy order and fresh backing
allocations per policy, 10 warmups and 20 timed samples per pass. The smoke
run uses one pass with both endpoints and balanced 50%. Buffer values, block
addresses and the FP32 reference are prepared before timing. The worker
checks physical HBM allocation deltas, Grace residency before and after GPU
work, output against the reference, and complete Grace allocator teardown.

Output is under `thesis/outputs/sparse_attention_mixed_*`. The two plots show
the latency vs. HBM saved across balanced policies, and the latency at a
fixed 32 MiB HBM saving as the number of active blocks on Grace changes.
Use the full run for the three-way comparison at equal capacity. The HBM
reference source used to populate the experimental cache and check outputs
remains in the process and is excluded from the experimental K/V deltas.
The fixed block list models known access patterns and does not include a
runtime policy for predicting or moving hot blocks.
