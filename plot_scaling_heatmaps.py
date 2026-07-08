#!/usr/bin/env python3
"""
plot_scaling_heatmaps.py
------------------------
Create 4 heatmap figures of the *elementwise* error norms produced by
run_scaling_sweep.sh, with t on the x-axis and s (number of slices) on the y-axis.

The 4 elementwise columns are:
  elem_C2_C1  (FP64 scaled   vs FP64 ref)
  elem_C3_C1  (INT8 scaled   vs FP64 ref)
  elem_C4_C1  (INT8 unscaled vs FP64 ref)
  elem_C3_C4  (INT8 scaled   vs INT8 unscaled)

Usage:
  python3 plot_scaling_heatmaps.py <sweep.csv> [--outdir .] [--linear]

Requires: numpy, pandas, matplotlib.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ELEM_COLS = {
    "elem_C2_C1": "Elementwise rel. error: FP64 scaled vs FP64 ref (C2 vs C1)",
    "elem_C3_C1": "Elementwise rel. error: INT8 scaled vs FP64 ref (C3 vs C1)",
    "elem_C4_C1": "Elementwise rel. error: INT8 unscaled vs FP64 ref (C4 vs C1)",
    "elem_C3_C4": "Elementwise rel. error: INT8 scaled vs INT8 unscaled (C3 vs C4)",
}


def make_heatmap(df, col, title, out_path, linear):
    grid = df.pivot(index="s", columns="t", values=col).sort_index()
    grid = grid.reindex(sorted(grid.columns), axis=1)

    s_vals = grid.index.to_numpy()
    t_vals = grid.columns.to_numpy()
    Z = grid.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6))

    if linear:
        norm = None
    else:
        # Log scale: mask non-positive (e.g. exact-zero FP64 comparison) entries.
        Zpos = Z[np.isfinite(Z) & (Z > 0)]
        if Zpos.size:
            norm = LogNorm(vmin=Zpos.min(), vmax=Zpos.max())
        else:
            norm = None
        Z = np.ma.masked_less_equal(Z, 0.0)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="lightgray")  # masked / zero entries

    # Cell-centered edges so ticks land on the actual s/t values.
    def edges(v):
        v = np.asarray(v, dtype=float)
        if v.size == 1:
            return np.array([v[0] - 0.5, v[0] + 0.5])
        mid = (v[:-1] + v[1:]) / 2
        return np.concatenate([[v[0] - (mid[0] - v[0])], mid,
                               [v[-1] + (v[-1] - mid[-1])]])

    mesh = ax.pcolormesh(edges(t_vals), edges(s_vals), Z,
                         cmap=cmap, norm=norm, shading="flat")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("elementwise relative error" + ("" if linear else " (log scale)"))

    ax.set_xlabel("t  (scaling exponent range: r_k in [-t, t])")
    ax.set_ylabel("s  (number of slices / moduli)")
    ax.set_title(title)
    ax.set_xticks(t_vals)
    ax.set_yticks(s_vals)
    ax.set_xticklabels([str(int(t)) for t in t_vals], rotation=90, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV produced by run_scaling_sweep.sh")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--linear", action="store_true",
                    help="use a linear color scale instead of log")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.csv))[0]

    for col, title in ELEM_COLS.items():
        if col not in df.columns:
            print(f"[warn] column {col} not in CSV, skipping")
            continue
        out_path = os.path.join(args.outdir, f"{base}_{col}_heatmap.png")
        make_heatmap(df, col, title, out_path, args.linear)


if __name__ == "__main__":
    main()
