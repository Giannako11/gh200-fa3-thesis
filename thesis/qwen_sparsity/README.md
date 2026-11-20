# Qwen activation sparsity distributions

Measure the existing Qwen2.5-32B-Instruct BF16/FA3 model on GH200 without
changing its weights, allocator, attention implementation or numerical values.
This is diagnostic inference, not a Grace execution policy or an inference speed benchmark.

## Fixed full experiment

| Setting | Full run |
|---|---|
| Distinct questions | 300 |
| Input lengths, including chat framing | 1,024 / 2,048 / 4,096 / 8,192 / 16,384 / 32,768 / 65,536 tokens |
| Generations | **2,100**; the same questions at all seven lengths |
| Generation | Batch 1, raw-logit greedy argmax without sampling/logit processors, up to 64 new tokens, normal EOS stopping |
| Model | Existing local Qwen2.5-32B-Instruct, all weights in Hopper HBM |
| Configuration | Existing baseline config when available; otherwise validated local checkpoint with explicit runtime YaRN4/128K config |
| Activations | MLP input, input to `down_proj` after the gated activation, Q/K/V projection outputs; all 64 layers |
| Prefill sampling | 16 evenly spaced token rows, including the first and last |
| Decode sampling | Steps 1, 2, 4, 8, 16, 32, 48, 63, when generation reaches them |
| Static weights | Exact zero/nonfinite counts for all 2D parameters, scanned once per experiment in bounded chunks |

One activation observation is one token's feature vector at one instrumented
layer/module, not an entire prefill tensor. Prefill observations describe token
positions inside the input. Decode observations describe already-generated
tokens fed back to the model; the final predicted token has no activation until
it is fed back. No full attention-probability matrix or activation dump is created.

For each sampled row: exact-zero count, nonfinite count, mean absolute value,
RMS, maximum magnitude, counts with `abs(x) <= 1e-5, 1e-4, 1e-3, 1e-2`, magnitude
histogram, and fully zero contiguous feature blocks of widths 16/64/256.
Trailing incomplete blocks are excluded. These are **1D feature blocks**, not
evidence of empty 2D matrix tiles. BF16 values are inspected using FP32 reductions.
Near-zero statistics never prune values or imply free, exact sparsity.

## Prompts and scope

The runner downloads the public [LongBench v2 dataset](https://huggingface.co/datasets/zai-org/LongBench-v2)
once: pinned revision `ac35867b9c9436dc81c4c7535fe640ba7cf47877`, SHA256
`15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2`.
The JSON is approximately **444 MiB**, saved outside Git under
`$HOME/.cache/thesis_qwen_sparsity/longbench-v2.json` (override with `--dataset-cache`).
It does not install packages or download a second model/checkpoint.

Selection is deterministic (seed 711), with round-robin domains among eligible
questions. Sources must contain enough genuine text for the largest input;
short sources are skipped, never repeated or padded. Questions sharing the same
source context are labelled with its hash. Prepared token IDs and their checksums
are retained in the output, so resumes do not retokenize or fetch the dataset.

The first 65,536 context tokens form the source excerpt. Each input length uses
a head+tail crop of that excerpt, preserving the question, options and chat
delimiters in an exact token budget. Framing and context are tokenized separately.
This may remove answer evidence; **this is not a LongBench task-accuracy result**.
Results describe this long-source, domain-stratified subset and this generation
protocol, not a universal probability distribution for Qwen. Short real-world
questions may have a different distribution. Early EOS and missing/OOM cases
are recorded; no synthetic decode samples fill the gaps.

## Run on grace2

Wait for the current matrix sweep to finish. The shared repository lock prevents
the two runners from overlapping. Use the original `fa3` environment; the shell
wrapper activates that existing environment if necessary. Required versions
match the baseline: PyTorch 2.11.0+cu128, Transformers 4.57.1, FA3 3.0.1.
NumPy and Matplotlib must already be available. Everything stays under `thesis/`;
there are no package installs, builds or changes to original model code.

In tmux, run the four-generation smoke (two prompts × 1K/2K, up to four new tokens),
then the full configuration:

```bash
cd ~/repos/gh200-fa3-thesis
conda activate fa3
bash thesis/scripts/run_qwen_sparsity.sh --smoke &&
bash thesis/scripts/run_qwen_sparsity.sh
```

The model is loaded once per invocation, not for every prompt. Each invocation
performs two short validation forwards, checking identical logits with sampling
disabled/enabled and actual FA3 calls. These are not extra full generations.
The smoke uses separate output files and does not replace any of the 2,100 full
cases. Weight scanning is omitted in the smoke. The full run has no automatic
wall-time limit; `--max-runs N` optionally ends cleanly after N attempted cases.

Model weights and tokenizer are read offline from the existing cache (default
`$HOME/models/huggingface`, override with `--cache-dir`). A successful historical
baseline in `thesis/outputs/qwen_baseline_*` is reused when available. It is not
required: the runner can find the existing `models--Qwen--Qwen2.5-32B-Instruct`
cache, its optional `hub/` parent, or a local Qwen checkpoint. For an exact path:

```bash
bash thesis/scripts/run_qwen_sparsity.sh --smoke --model-dir /path/to/checkpoint
```

`--model-dir` accepts the directory containing `config.json` and the weight files,
or the Hugging Face model cache directory containing `refs/` and `snapshots/`.
Without old baseline files, the runner validates Qwen32B's architecture, BF16
configuration, tokenizer files and every listed safetensors shard. It creates the
explicit YaRN-factor-4/128K config in memory, leaving the cached config unchanged.
The manifest records that this came from a local checkpoint, not a historical
successful run. Snapshot revision, config/tokenizer hashes and file size/mtime
records are retained and checked on resume. Large shard bytes are not fully
hashed by this preflight; the actual loader/runtime validation still has to pass.
Ambiguous cached revisions without a local `refs/main` need an exact `--model-dir`.
An explicit `--baseline-dir` still requires a valid baseline and is not ignored.
Other compute processes cause preflight to stop; the runner never kills them.
The dataset download uses HTTPS independently of offline model loading.

## Pause and resume

Press **Ctrl+C**, wait for the saved-result message and the process to return.
You can then close tmux/SSH. This releases the model/GPU; it is not a suspended
process holding GPU memory.

```bash
bash thesis/scripts/run_qwen_sparsity.sh --resume /absolute/path/to/qwen_sparsity_<timestamp>_<unique>
```

Only committed PASS cases with matching config and file hashes are skipped.
An interrupted/incomplete generation restarts from its beginning. Its old files
are moved into a timestamped attempt folder; completed runs are not repeated.
The same logic recovers after an abrupt process kill or lost terminal. A stop
during prompt preparation restarts that CPU preparation, with no model runs lost.
Code snapshots are saved inside the run; the shell resume command selects the
original snapshot, even after later repository updates. Environment/GPU,
configuration and prompt checksums must still match.

OOM is recorded per case with the original requested length. Other cases
continue, with no automatic context reduction. OOM cases are retained and not
retried by default; add `--retry-failed` when deliberately retrying them.
An invocation ending with OOM records returns exit code 2, not an all-PASS result.
Import, placement or validation errors stop the invocation and remain retryable.
At 64K, generation and instrumentation require additional memory beyond prefill;
capacity success in the earlier prefill-only experiment is not a guarantee.

## Outputs and plots

The output directory is printed immediately. It contains:

- `manifest.json`, `runner_source/`, `prompts/`: exact protocol, original code,
  source IDs and reproducible input tokens. No model weights are copied.
- `run.log`, `runs.csv`, `summary.json` / `.md`: progress, status, wall times,
  prefill/decode times and peak memory for each generation.
- `cases/<id>/observations.npz`: compressed metrics and layer/token/shape metadata.
  Full activations are not retained. `generation.json` stores generated token IDs,
  text and stopping reason. `steps.jsonl` records time per forward;
  `progress.json` records the current stage.
- `weights.json` / `.csv`: static weight zero counts and shapes.
- `distributions.csv`, `survival.csv`, `per_prompt.csv`, `layer_summary.csv`,
  `analysis.json`: probability tables, uncertainty and sampling definitions.
- `plots/`: exact-zero and near-zero distributions, probability of sparsity at
  least S, layer/context mean heatmaps, layer × sparsity-frequency heatmaps for
  all seven lengths, magnitude distributions, zero feature blocks and diagnostic
  timings. Gray/blank areas are missing observations, not zero sparsity.

Probabilities give **equal total weight to each available prompt** within each
context length, phase and tensor kind, with equal sampled-row weight inside it.
Confidence intervals resample source documents as clusters; questions sharing
a context stay together. They are pointwise 95% percentile intervals from 400
bootstrap samples, not a claim that correlated layer/token observations are
independent. Fewer than two source clusters cannot establish an interval.

Plots are rebuilt after a normal completion, clean limit or Ctrl+C, using only
successful generations. An OOM-biased partial run is not the complete distribution.
For CPU-only regeneration:

```bash
python thesis/qwen_sparsity/plot.py /absolute/path/to/output
# or, inside fa3:
bash thesis/scripts/run_qwen_sparsity.sh --plot-only /absolute/path/to/output
```

Timing includes hooks, reductions, synchronization and diagnostic I/O. It helps
estimate this experiment's duration, not compare raw Qwen throughput. The next
step is to replay representative real operations on Grace/Hopper, including
format conversions and transfers; sparsity frequency alone does not decide placement.

## Simple frequency histograms from existing captures

For a single-chart view of the most common sparsity range, without another
model run:

```bash
python thesis/qwen_sparsity/plot_frequency.py /absolute/path/to/output
```

This creates two supplementary PNGs and accompanying CSV/JSON under `plots/`:

- `sparsity_frequency_per_generation.png`: one value per generation, calculated
  as all exact zeros divided by all measured activation elements. This is an
  element-weighted summary of the stored samples, not an exhaustive measurement
  of every activation in inference. Each generation has equal histogram weight.
- `sparsity_frequency_encountered_vectors.png`: sparsity of an encountered
  sampled token/layer/tensor feature vector. Each generation has equal total
  weight, with equal weight per sampled vector inside it. Exact zero sparsity
  is a separate point mass; the positive tail uses fine percentage-point bins.

The first chart answers how the average sampled sparsity varies between runs;
the second answers which individual sampled vectors are commonly encountered.
They must not be conflated for a per-tensor placement policy. Both use actual
empirical frequencies, without imposing a normal distribution. Report the modal
interval with its bin width; it is not an exact point-probability estimate.

## Validation

CPU tests check metrics against NumPy, hook equivalence, exact weight counts,
2,100-case scheduling, prompt balancing, source-cluster uncertainty, atomic
resume/corruption detection, full generation bookkeeping and early EOS using a
small CPU model. These do not establish GH200 runtime correctness; run the smoke
on grace2 before starting the full experiment.

```bash
python -m unittest discover -s thesis/tests -p test_qwen_sparsity.py -v
```
