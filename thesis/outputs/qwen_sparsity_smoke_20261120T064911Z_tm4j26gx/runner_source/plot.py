"""Prompt-balanced empirical distributions; bootstrap resamples whole prompts."""

from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from protocol import BLOCKS, KINDS, MAG_EDGES, METRICS, THRESHOLDS, atomic_json, complete_result

BINS = np.arange(0, 105, 5)
CUTS = np.arange(0, 105, 5)


def prompt_distribution(percentages):
    """A prompt contributes unit mass regardless of its sampled token count."""
    percentages = np.asarray(percentages, dtype=float)
    hist = np.histogram(percentages, bins=BINS)[0].astype(float)
    return hist / len(percentages)


def bootstrap_mean(values, draws=400, seed=711, clusters=None):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    estimate = values.mean(axis=0)
    clusters = list(range(len(values))) if clusters is None else list(clusters)
    if len(clusters) != len(values):
        raise ValueError("Cluster labels must align with prompt observations")
    labels = sorted(set(clusters))
    if len(labels) == 1:
        # One source document cannot establish between-document uncertainty.
        return estimate, np.full_like(estimate, np.nan), np.full_like(estimate, np.nan)
    group_sums = np.stack([values[np.asarray(clusters) == label].sum(axis=0) for label in labels])
    group_counts = np.asarray([clusters.count(label) for label in labels])
    indices = rng.integers(0, len(labels), size=(draws, len(labels)))
    sampled = group_sums[indices].sum(axis=1) / group_counts[indices].sum(axis=1).reshape((-1,) + (1,) * (values.ndim - 1))
    return estimate, *np.quantile(sampled, [0.025, 0.975], axis=0)


def write_csv(path, rows, fields):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def analyze(out):
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text())
    histograms, survival, magnitudes, block_means = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    layers, layer_hists = defaultdict(list), defaultdict(list)
    clusters = defaultdict(list)
    per_prompt, timings = [], []
    completed, observations = 0, 0
    print("Building prompt-balanced distribution tables from saved PASS generations", flush=True)
    for case in manifest["cases"]:
        folder = out / "cases" / case["id"]
        if not (folder / "result.json").exists():
            continue
        row = complete_result(folder, manifest["identity"], case)
        if row is None:
            continue
        if row.get("identity") != manifest["identity"]:
            raise RuntimeError(f"Mismatched result identity: {folder}")
        completed += 1
        if completed % 50 == 0:
            print(f"Distribution analysis: {completed} generations read", flush=True)
        timings.append(row)
        with np.load(folder / "observations.npz", allow_pickle=False) as saved:
            if tuple(saved["metric_names"]) != METRICS or tuple(saved["kinds"]) != KINDS:
                raise RuntimeError("Metric schema changed")
            data, meta = saved["metrics"].astype(np.float64), saved["metadata"]
        observations += len(meta)
        width = meta[:, 5]
        for phase_id, phase in enumerate(("prefill", "decode")):
            for kind_id, kind in enumerate(KINDS):
                mask = (meta[:, 1] == kind_id) & (meta[:, 2] == phase_id)
                if not mask.any():
                    continue
                local, dims, layer_ids = data[mask], width[mask], meta[mask, 0]
                key = (case["input_tokens"], phase, kind)
                clusters[key].append(manifest["prompts"][case["prompt_index"]]["context_sha256"])
                exact = 100 * local[:, METRICS.index("zero_count")] / dims
                measured = {"exact_zero": exact}
                for threshold in THRESHOLDS:
                    measured[f"abs_le_{threshold:g}"] = 100 * local[:, METRICS.index(f"small_{threshold:g}_count")] / dims
                for metric, percentages in measured.items():
                    histograms[(*key, metric)].append(prompt_distribution(percentages))
                survival[key].append(np.asarray([(exact >= cut).mean() for cut in CUTS]))
                mag = local[:, [METRICS.index(f"magnitude_bin_{i}") for i in range(len(MAG_EDGES) + 1)]]
                magnitudes[key].append((mag / dims[:, None]).mean(axis=0))
                for block in BLOCKS:
                    full = dims // block
                    valid = full > 0
                    fractions = 100 * local[valid, METRICS.index(f"zero_blocks_{block}")] / full[valid]
                    if len(fractions):
                        block_means[(*key, block)].append(float(fractions.mean()))
                per_prompt.append({"prompt_index": case["prompt_index"], "domain": case["domain"],
                                   "input_tokens": key[0], "phase": phase, "kind": kind,
                                   "sampled_rows": int(mask.sum()), "width": int(dims[0]),
                                   "mean_exact_zero_pct": float(exact.mean()),
                                   **{f"mean_{name}_pct": float(values.mean()) for name, values in measured.items() if name != "exact_zero"}})
                for layer in np.unique(layer_ids):
                    selected = exact[layer_ids == layer]
                    layers[(*key, int(layer))].append(float(selected.mean()))
                    layer_hists[(*key, int(layer))].append(prompt_distribution(selected))
    distribution_rows = []
    for key, values in sorted(histograms.items()):
        mean, low, high = bootstrap_mean(values, clusters=clusters[key[:3]])
        for index in range(len(BINS) - 1):
            distribution_rows.append(dict(zip(("input_tokens", "phase", "kind", "metric"), key),
                    bin_low_pct=int(BINS[index]), bin_high_pct=int(BINS[index + 1]),
                    probability=float(mean[index]), ci_low=float(low[index]), ci_high=float(high[index]),
                    prompts=len(values)))
    write_csv(out / "distributions.csv", distribution_rows,
              ["input_tokens", "phase", "kind", "metric", "bin_low_pct", "bin_high_pct", "probability", "ci_low", "ci_high", "prompts"])
    survival_rows = []
    for key, values in sorted(survival.items()):
        mean, low, high = bootstrap_mean(values, clusters=clusters[key])
        for index, cut in enumerate(CUTS):
            survival_rows.append(dict(zip(("input_tokens", "phase", "kind"), key),
                    sparsity_at_least_pct=int(cut), probability=float(mean[index]),
                    ci_low=float(low[index]), ci_high=float(high[index]), prompts=len(values)))
    write_csv(out / "survival.csv", survival_rows,
              ["input_tokens", "phase", "kind", "sparsity_at_least_pct", "probability", "ci_low", "ci_high", "prompts"])
    write_csv(out / "per_prompt.csv", per_prompt,
              ["prompt_index", "domain", "input_tokens", "phase", "kind", "sampled_rows", "width", "mean_exact_zero_pct",
               *(f"mean_abs_le_{t:g}_pct" for t in THRESHOLDS)])
    layer_rows = [dict(zip(("input_tokens", "phase", "kind", "layer"), key),
                       mean_exact_zero_pct=float(np.mean(values)), prompts=len(values))
                  for key, values in sorted(layers.items())]
    write_csv(out / "layer_summary.csv", layer_rows,
              ["input_tokens", "phase", "kind", "layer", "mean_exact_zero_pct", "prompts"])
    if (out / "weights.json").exists():
        weights = json.loads((out / "weights.json").read_text())["parameters"]
        write_csv(out / "weights.csv", weights, ["name", "shape", "dtype", "elements", "zero_count", "nonfinite_count", "sparsity_pct"])
    atomic_json(out / "analysis.json", {"completed_generations": completed, "expected_generations": len(manifest["cases"]),
                "sampled_token_layer_tensor_observations": observations,
                "probability_definition": "equal weight per observed prompt at fixed context length, phase and tensor kind; equal sampled row/layer weight within a prompt",
                "uncertainty": "95% pointwise percentile cluster bootstrap, 400 resamples of source context hashes; questions sharing a context stay together; correlated token rows are not independent trials",
                "histogram_bins": "[0,5), [5,10), ..., [95,100]; finite-width bins are not exact point probabilities",
                "selection_scope": "length-eligible, domain round-robin LongBench-v2 subset; not all inference workloads",
                "threshold_note": "Near-zero counts are descriptive. Nothing is pruned or moved to CPU.",
                "missing_note": "Only PASS generations contribute. OOM/partial/failed generations are excluded; missing results are not zero-filled.",
                "dtype": "BF16 activations, FP32 metric calculations", "block_definition": manifest["protocol"]["block_definition"]})
    return manifest, histograms, survival, magnitudes, block_means, layers, layer_hists, timings, clusters


def plot(out):
    out = Path(out)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    manifest, hist, survival, magnitudes, blocks, layers, layer_hists, timings, clusters = analyze(out)
    if not timings:
        print("No completed generations to plot yet.", flush=True)
        return
    plots = out / "plots"
    plots.mkdir(exist_ok=True)
    lengths = manifest["protocol"]["lengths"]
    colours = {length: plt.get_cmap("viridis")(i / max(1, len(lengths) - 1)) for i, length in enumerate(lengths)}

    def panels(title):
        fig, axes = plt.subplots(len(KINDS), 2, figsize=(13, 15), squeeze=False, constrained_layout=True)
        fig.suptitle(title + "\nPrompt-balanced; available PASS runs only", fontsize=13)
        for r, kind in enumerate(KINDS):
            for c, phase in enumerate(("prefill", "decode")):
                axes[r, c].set_title(f"{kind} — {phase}", fontsize=10)
        return fig, axes

    def save(fig, name):
        fig.savefig(plots / name, dpi=150)
        plt.close(fig)
        print(f"Saved plot: {name}", flush=True)

    for metric in ("exact_zero", "abs_le_0.001"):
        fig, axes = panels("Sparsity distribution: " + metric)
        for r, kind in enumerate(KINDS):
            for c, phase in enumerate(("prefill", "decode")):
                ax = axes[r, c]
                for length in lengths:
                    values = hist.get((length, phase, kind, metric))
                    if values:
                        ax.stairs(100 * np.mean(values, axis=0), BINS, color=colours[length], label=f"{length // 1024}K (n={len(values)})")
                ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="Sparsity (%)", ylabel="Probability per 5-point bin (%)")
                ax.grid(alpha=.2)
                if r == 0:
                    ax.legend(fontsize=7)
        save(fig, f"distribution_{metric}.png")

    fig, axes = panels("Probability of sparsity at least S — exact zeros")
    for r, kind in enumerate(KINDS):
        for c, phase in enumerate(("prefill", "decode")):
            ax = axes[r, c]
            for length in lengths:
                values = survival.get((length, phase, kind))
                if values:
                    mean, low, high = bootstrap_mean(values, clusters=clusters[(length, phase, kind)])
                    ax.plot(CUTS, 100 * mean, color=colours[length], label=f"{length // 1024}K")
                    ax.fill_between(CUTS, 100 * low, 100 * high, color=colours[length], alpha=.12)
            ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="S (%)", ylabel="P(sparsity ≥ S) (%)")
            ax.grid(alpha=.2)
            if r == 0:
                ax.legend(fontsize=7)
    save(fig, "survival_exact_zero.png")

    fig, axes = panels("Mean exact-zero sparsity by layer and context length")
    n_layers = manifest["config"]["num_hidden_layers"]
    for r, kind in enumerate(KINDS):
        for c, phase in enumerate(("prefill", "decode")):
            matrix = np.full((len(lengths), n_layers), np.nan)
            for i, length in enumerate(lengths):
                for layer in range(n_layers):
                    values = layers.get((length, phase, kind, layer))
                    if values:
                        matrix[i, layer] = np.mean(values)
            ax = axes[r, c]
            image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", vmin=0, vmax=100, cmap="magma")
            ax.set(yticks=range(len(lengths)), yticklabels=[f"{t // 1024}K" for t in lengths], xlabel="Layer", ylabel="Input length")
            fig.colorbar(image, ax=ax, label="Mean sparsity (%)")
    save(fig, "layer_mean_exact_zero.png")

    for length in lengths:
        fig, axes = panels(f"Layer × sparsity distribution — {length // 1024}K, exact zeros")
        for r, kind in enumerate(KINDS):
            for c, phase in enumerate(("prefill", "decode")):
                matrix = np.full((len(BINS) - 1, n_layers), np.nan)
                for layer in range(n_layers):
                    values = layer_hists.get((length, phase, kind, layer))
                    if values:
                        matrix[:, layer] = 100 * np.mean(values, axis=0)
                ax = axes[r, c]
                image = ax.imshow(np.ma.masked_invalid(matrix), origin="lower", aspect="auto", extent=(-.5, n_layers - .5, 0, 100), vmin=0, vmax=100, cmap="viridis")
                ax.set(xlabel="Layer", ylabel="Sparsity bin (%)")
                fig.colorbar(image, ax=ax, label="Probability per bin (%)")
        save(fig, f"layer_distribution_T{length:05d}.png")

    fig, axes = panels("Activation magnitude distribution — first bin includes exact zeros")
    labels = ["≤1e-8", "≤1e-6", "≤1e-4", "≤1e-3", "≤1e-2", "≤0.1", "≤1", "≤10", "≤100", ">100"]
    for r, kind in enumerate(KINDS):
        for c, phase in enumerate(("prefill", "decode")):
            ax = axes[r, c]
            for length in lengths:
                values = magnitudes.get((length, phase, kind))
                if values:
                    ax.plot(range(len(labels)), 100 * np.mean(values, axis=0), color=colours[length], label=f"{length // 1024}K")
            ax.set(xticks=range(len(labels)), xticklabels=labels, ylabel="Element fraction per interval (%)")
            ax.tick_params(axis="x", rotation=45, labelsize=7)
            ax.grid(alpha=.2)
            if r == 0:
                ax.legend(fontsize=7)
    save(fig, "magnitude_distribution.png")

    fig, axes = panels("Fully zero 1D feature blocks — not 2D matrix tiles")
    for r, kind in enumerate(KINDS):
        for c, phase in enumerate(("prefill", "decode")):
            ax = axes[r, c]
            for block in BLOCKS:
                means = [np.mean(blocks[(length, phase, kind, block)]) if blocks.get((length, phase, kind, block)) else np.nan for length in lengths]
                ax.plot(range(len(lengths)), means, marker="o", label=f"width {block}")
            ax.set(xticks=range(len(lengths)), xticklabels=[f"{t // 1024}K" for t in lengths], ylim=(0, 100), ylabel="Fully zero blocks (%)")
            ax.grid(alpha=.2)
            if r == 0:
                ax.legend(fontsize=7)
    save(fig, "zero_feature_blocks.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, field, title in zip(axes, ("wall_seconds", "prefill_profiled_seconds"), ("Whole profiled generation", "Profiled prefill")):
        values = [[row[field] for row in timings if row["input_tokens"] == length] for length in lengths]
        ax.boxplot([v if v else [np.nan] for v in values])
        ax.set_xticks(range(1, len(lengths) + 1), [f"{t // 1024}K" for t in lengths])
        ax.set(title=title, ylabel="Seconds (instrumented)", yscale="log", xlabel="Input length")
        ax.grid(alpha=.2)
    save(fig, "profiling_times.png")
    print(f"Plots and distribution tables: {out}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    plot(parser.parse_args().output)
