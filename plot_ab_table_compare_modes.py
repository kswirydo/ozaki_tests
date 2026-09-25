#!/usr/bin/env python3
"""
Overlay accurate vs fast ab_table_benchmark runs on one log–log diagonal plot.

Eight Ozaki series (4 methods × 2 modes), plus red reference at log10(err) = log10(1e-16):
  cond(A) = cond(B) = 10^k, k = 0..17 (18 points per line).

Usage:
    python3 plot_ab_table_compare_modes.py accurate.csv fast.csv
    python3 plot_ab_table_compare_modes.py accurate.csv fast.csv out/compare
    python3 plot_ab_table_compare_modes.py
        # uses newest ab_table_benchmark_accurate_*.csv and ab_table_benchmark_fast_*.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Shared loaders/constants from single-mode plot script (same directory).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from plot_ab_table_heatmap import (  # noqa: E402
    ERROR_COLUMNS,
    EXPECTED_ROWS,
    MACHINE_ACCURACY,
    apply_shared_accuracy_axes,
    diagonal_errors,
    find_csv,
    load_csv,
    log10_relative_error,
)

mpl.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "legend.fontsize": 9,
    }
)


def plot_both_modes(
    grids_acc: dict[str, np.ndarray],
    meta_acc: dict,
    grids_fast: dict[str, np.ndarray],
    meta_fast: dict,
    out_path: str,
    *,
    machine_ref: float = MACHINE_ACCURACY,
) -> None:
    k = np.arange(18, dtype=float)
    cond = np.power(10.0, k)
    log10_ref = float(np.log10(machine_ref))

    fig, ax = plt.subplots(figsize=(10.5, 7))

    for col, (label, color) in ERROR_COLUMNS.items():
        for grids, mode_name, linestyle, marker in (
            (grids_acc, "accurate", "-", "o"),
            (grids_fast, "fast", "--", "s"),
        ):
            err = diagonal_errors(grids[col])
            log_err = log10_relative_error(err)
            ok = np.isfinite(log_err)
            if not np.any(ok):
                continue
            ax.semilogx(
                cond[ok],
                log_err[ok],
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=6 if linestyle == "-" else 5,
                linewidth=2,
                label=f"{label} ({mode_name})",
                clip_on=False,
            )

    ax.semilogx(
        [cond[0], cond[-1]],
        [log10_ref, log10_ref],
        color="red",
        linewidth=2.5,
        linestyle="-",
        label=f"Machine accuracy ($\\log_{{10}}={log10_ref:g}$)",
        zorder=10,
    )

    ax.set_title("GEMMul8 Ozaki II emulation error: accurate vs fast (cond A = cond B)")
    ax.set_xlabel(r"Condition number  $\mathrm{cond}(A)=\mathrm{cond}(B)$")
    ax.set_ylabel(
        r"$\log_{10}$ relative Frobenius error  "
        r"$\|C_{\mathrm{ozaki}}-C_{\mathrm{FP64}}\|_F / \|C_{\mathrm{FP64}}\|_F$"
    )
    ax.grid(True, which="major", alpha=0.4, linestyle="--")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, framealpha=0.95)
    apply_shared_accuracy_axes(ax, cond=cond)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _check_mode(meta: dict, expected_fast: int, label: str) -> None:
    fm = meta.get("ozaki_fastmode")
    if fm is None:
        return
    if fm != expected_fast:
        print(
            f"Warning: {label} CSV has ozaki_fastmode={fm}, expected {expected_fast}",
            file=sys.stderr,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "accurate_csv",
        nargs="?",
        help="Accurate-mode CSV (or glob). Default: newest ab_table_benchmark_accurate_*.csv",
    )
    parser.add_argument(
        "fast_csv",
        nargs="?",
        help="Fast-mode CSV (or glob). Default: newest ab_table_benchmark_fast_*.csv",
    )
    parser.add_argument(
        "prefix",
        nargs="?",
        help="Output prefix (default: ab_table_compare_accurate_vs_fast)",
    )
    parser.add_argument("--machine-ref", type=float, default=MACHINE_ACCURACY)
    args = parser.parse_args()

    acc_path = find_csv(args.accurate_csv or "ab_table_benchmark_accurate_*.csv")
    fast_path = find_csv(args.fast_csv or "ab_table_benchmark_fast_*.csv")
    prefix = args.prefix or "ab_table_compare_accurate_vs_fast"
    out_path = f"{prefix}_diagonal_loglog.png"

    grids_acc, meta_acc = load_csv(acc_path)
    grids_fast, meta_fast = load_csv(fast_path)

    print(f"Accurate: {acc_path}", file=sys.stderr)
    print(f"Fast:     {fast_path}", file=sys.stderr)
    _check_mode(meta_acc, 0, "accurate")
    _check_mode(meta_fast, 1, "fast")
    for name, meta in ("accurate", meta_acc), ("fast", meta_fast):
        if meta["rows_read"] != EXPECTED_ROWS:
            print(f"Warning: {name} has {meta['rows_read']}/{EXPECTED_ROWS} rows", file=sys.stderr)

    plot_both_modes(
        grids_acc,
        meta_acc,
        grids_fast,
        meta_fast,
        out_path,
        machine_ref=args.machine_ref,
    )
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
