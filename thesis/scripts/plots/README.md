# First TMA plots

Run on a CPU with Python 3.10+, Matplotlib and NumPy:

```bash
python thesis/scripts/plots/tma_plots_first.py thesis/outputs/RUN_FOLDER/results.csv
```

This creates `tma_plots_first/` beside the CSV, with 39 figures in PNG and PDF:

- `barplots/`: 20 figures. KiB, MiB and GiB input sizes with a fixed 32x32 tile,
  plus all seven tiles with a fixed 1 MiB input. Each comparison has four separate
  allocation figures and one combined figure with four panels.
- `speedup/`: 5 figures showing regular shared/TMA and regular direct/TMA time ratios.
- `heatmaps/`: 5 figures showing regular shared/TMA across all input sizes and tiles.
- `host_vs_hbm/`: 4 figures showing host/HBM time ratios, separately by kernel and
  together in a three-panel figure.
- `input_rate/`: 5 figures showing effective input GB/s versus input size, using
  the fixed tile. Four separate allocation figures and one four-panel figure.

Effective input GB/s is recomputed as `input_bytes / (kernel_ms * 1e6)`.
The time includes output stores to HBM; this is not physical DRAM or NVLink
bandwidth. The input-rate plots use a logarithmic input-size axis and a linear
GB/s axis starting at zero, with the same GPU kernel colors as the barplots.

The figures use English labels. GPU kernel colors are consistent in the barplots.
Ratios use a logarithmic scale with a reference at 1x. Separate and combined
versions use the same scales. Empty or failed results stay missing; they are not
replaced by zero. Hatched bars/cells and open markers identify host measurements
without verified CPU page placement before and after the kernel.

`checks.csv` is just for our own checks. `plots_index.csv` lists the figures,
and `plot_notes.txt` records the input path/hash and plotting settings.
Partial sweeps also work; every figure states the recorded and successful counts
against the default full sweep of 1,740 measurements. Duplicate measurements and
mixed GPU thread counts are rejected rather than averaged.

Options:

```bash
python thesis/scripts/plots/tma_plots_first.py results.csv --output-dir my_plots --formats png
python thesis/scripts/plots/tma_plots_first.py results.csv --tile 64x64 --fixed-size 64MiB
```

The fixed input for tile comparison must be at least 64 KiB. Larger fixed tiles
leave the smaller input-size bars empty. Each CSV should come from one run.
The script does not run the benchmark, need a GPU, or change the input CSV.
