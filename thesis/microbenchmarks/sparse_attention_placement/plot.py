"""Recreate the latency and HBM allocation plots from a successful GH200 run."""

import argparse
import csv
import json
from pathlib import Path

LABELS = {"hbm": "K/V in Hopper HBM", "grace": "K/V in Grace DRAM"}
COLORS = {"hbm": "#176385", "grace": "#bd552a"}


def plot(run):
    validation = json.loads((run / "result.json").read_text())
    if validation["status"] != "PASS":
        raise ValueError("plots require a successful placement and correctness result")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (run / "results.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))

    if validation.get("mode") == "mixed":
        plot_mixed(plt, run, rows)
        return

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


def plot_mixed(plt, run, rows):
    balanced = sorted((row for row in rows if not row["policy"].startswith("active_")),
                      key=lambda row: int(row["hbm_saved_bytes"]))
    equal_capacity = sorted((row for row in rows if int(row["grace_resident_blocks"]) == 64),
                            key=lambda row: int(row["grace_active_blocks"]))

    def draw(cases, x, xlabel, title, filename):
        median = [float(row["median_us"]) for row in cases]
        lower = [m - float(row["p25_us"]) for m, row in zip(median, cases)]
        upper = [float(row["p75_us"]) - m for m, row in zip(median, cases)]
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.errorbar(x, median, yerr=[lower, upper], color=COLORS["grace"],
                    marker="o", linewidth=1.7, capsize=4)
        ax.set(title=title, xlabel=xlabel, ylabel="Decode latency (µs)")
        ax.set_xticks(x)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.24)
        fig.savefig(run / filename, dpi=200)
        plt.close(fig)

    require_policies = {"all_hbm", "all_grace", "balanced_50"}
    if not require_policies <= {row["policy"] for row in balanced}:
        raise ValueError("mixed result is missing baseline policies")
    draw(balanced, [int(row["hbm_saved_bytes"]) / 2**20 for row in balanced],
         "HBM saved by K/V placement (MiB)", "Mixed K/V placement · median and IQR",
         "mixed_latency_vs_hbm_saved.png")
    draw(equal_capacity, [int(row["grace_active_blocks"]) for row in equal_capacity],
         "Blocks read from Grace DRAM (out of 64)",
         "Decode latency with 32 MiB of HBM saved",
         "mixed_latency_at_equal_capacity.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    plot(parser.parse_args().run)


if __name__ == "__main__":
    main()
