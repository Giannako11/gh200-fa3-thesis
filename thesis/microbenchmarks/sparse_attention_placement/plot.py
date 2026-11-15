"""Recreate the latency and HBM allocation plots from a successful GH200 run."""

import argparse
import csv
import json
from pathlib import Path

LABELS = {"hbm": "K/V in Hopper HBM", "grace": "K/V in Grace DRAM"}
COLORS = {"hbm": "#176385", "grace": "#bd552a"}


def plot(run):
    if json.loads((run / "result.json").read_text())["status"] != "PASS":
        raise ValueError("plots require a successful placement and correctness result")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (run / "results.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))

    series = {place: sorted((row for row in rows if row["kv_placement"] == place),
                            key=lambda row: int(row["selected_blocks"])) for place in LABELS}
    x = [float(row["selected_percent"]) for row in series["hbm"]]
    grace_x = [float(row["selected_percent"]) for row in series["grace"]]
    if not x or len(x) != len(set(x)) or x != grace_x or len(rows) != 2 * len(x):
        raise ValueError("expected one HBM and one Grace measurement per block count")

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for place, cases in series.items():
        median = [float(row["median_us"]) for row in cases]
        lower = [m - float(row["p25_us"]) for m, row in zip(median, cases)]
        upper = [float(row["p75_us"]) - m for m, row in zip(median, cases)]
        ax.errorbar(x, median, yerr=[lower, upper], label=LABELS[place], color=COLORS[place],
                    marker="o", capsize=4, linewidth=1.7)
    ax.set(xlabel="K/V blocks read (%)", ylabel="Decode latency (µs)",
           title="Block-sparse decode latency by K/V placement")
    ax.set_xticks(x, [f"{value:g}%" for value in x])
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.24)
    ax.legend(title="Median and IQR · warmed buffers", frameon=False)
    fig.savefig(run / "attention_latency.png", dpi=200)
    plt.close(fig)

    measured_mib = []
    for cases in series.values():
        deltas = {int(row["kv_hbm_delta_bytes"]) for row in cases}
        if len(deltas) != 1:
            raise ValueError("expected a fixed K/V allocation across block counts")
        measured_mib.append(deltas.pop() / 2**20)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    bars = ax.bar(list(LABELS.values()), measured_mib, width=0.55,
                  color=list(COLORS.values()))
    ax.bar_label(bars, labels=[f"{mib:g} MiB" for mib in measured_mib], padding=6)
    ax.set(ylabel="Increase in HBM usage (MiB)", title="HBM used by the 32 MiB K/V cache")
    ax.set_ylim(0, max(measured_mib + [1]) * 1.18)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.24)
    fig.savefig(run / "kv_hbm_usage.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    plot(parser.parse_args().run)


if __name__ == "__main__":
    main()
