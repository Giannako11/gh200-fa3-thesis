"""Plot measured full-sweep timings from device_times.csv."""

import argparse
import csv
import json
from pathlib import Path

from grid import FULL_DENSITIES, SHAPES


def seconds(row, key):
    if row is None or row["status"] != "pass":
        return None
    value = row.get(f"median_{key}_seconds")
    if not value:
        return None
    total = float(value)
    return total if total > 0 else None


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    identity = json.loads((out / "manifest.json").read_text())["identity"]
    if identity.get("mode") != "full5" or identity.get("smoke"):
        raise ValueError("heatmaps require a non-smoke --full run")

    results = {}
    with (out / "device_times.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            shape = tuple(int(row[c]) for c in ("m", "k", "n"))
            a, b = int(row["a_pct"]), int(row["b_pct"])
            if shape not in SHAPES or a not in FULL_DENSITIES or b not in FULL_DENSITIES:
                raise ValueError("device_times.csv includes a case outside the full 5% sweep")
            key = shape, a, b, row["device"]
            if key in results:
                raise ValueError(f"duplicate benchmark row: {key}")
            results[key] = row

    labels = {
        "compute": ("Compute only", "log₂(Hopper GPU event time / Grace compute time)"),
        "transfers": ("Grace transfers only", "Grace round-trip copy time (ms)"),
        "end_to_end": ("End-to-end", "log₂(Hopper / Grace end-to-end time)"),
    }
    for kind, (title, label) in labels.items():
        matrices = []
        for shape in SHAPES:
            size = len(FULL_DENSITIES)
            grid = np.full((size, size), np.nan)
            for ai, a in enumerate(FULL_DENSITIES):
                for bi, b in enumerate(FULL_DENSITIES):
                    cpu = results.get((shape, a, b, "grace"))
                    gpu = results.get((shape, a, b, "hopper"))
                    if kind == "transfers":
                        elapsed = seconds(cpu, "transfer")
                        if elapsed is not None:
                            grid[bi, ai] = elapsed * 1000
                    else:
                        grace_time = seconds(cpu, kind)
                        hopper_time = seconds(gpu, "cuda_compute" if kind == "compute" else kind)
                        if grace_time and hopper_time:
                            grid[bi, ai] = np.log2(hopper_time / grace_time)
            matrices.append(grid)

        measured = np.concatenate([grid[np.isfinite(grid)] for grid in matrices])
        if kind == "transfers":
            lo = 0
            hi = max(float(measured.max()), 1e-6) if measured.size else 1
            cmap = plt.get_cmap("viridis").copy()
        else:
            hi = max(float(np.abs(measured).max()), 1) if measured.size else 1
            lo = -hi
            cmap = plt.get_cmap("RdBu").copy()
        cmap.set_bad("#e3e5e8")
        fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
        ticks = [5, 25, 50, 75, 100]
        for ax, shape, grid in zip(axes.flat, SHAPES, matrices):
            shown = ax.imshow(grid, origin="lower", extent=(2.5, 102.5, 2.5, 102.5),
                              interpolation="nearest", cmap=cmap, vmin=lo, vmax=hi)
            ax.set_title(" × ".join(map(str, shape))
                         + f"  ({np.isfinite(grid).sum()}/400)", fontsize=10)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
        fig.supxlabel("A density (%)")
        fig.supylabel("B density (%)")
        note = "Gray: missing or failed"
        if kind != "transfers":
            note += " · Positive log₂ ratios favor Grace"
        fig.suptitle(f"{title} · GH200\n{note}", fontsize=14)
        fig.colorbar(shown, ax=axes, label=label, shrink=.85)
        fig.savefig(out / f"heatmap_{kind}.png", dpi=180)
        plt.close(fig)
    print(f"Saved three measured heatmaps to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    plot(args.run_dir)


if __name__ == "__main__":
    main()
