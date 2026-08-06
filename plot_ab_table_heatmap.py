#!/usr/bin/env python3
"""
Plot ab_table_benchmark CSV: diagonal sweep cond(A) = cond(B).

Default figure (18 points per series):
  • X: condition number (log scale), same on A and B → 1, 10, …, 10^17
  • Y: log10(relative Frobenius error vs FP64), linear axis
  • Four Ozaki curves (INT8 12/16, FP8 10/12) + red reference at 1e-16

Usage:
    python3 plot_ab_table_heatmap.py out.csv
    python3 plot_ab_table_heatmap.py out.csv my_plot --format pdf --dpi 600
    python3 plot_ab_table_heatmap.py out.csv my_plot_diagonal_loglog.jpg --dpi 600
    python3 plot_ab_table_heatmap.py out.csv my_plot --fix-log10-cond-a 2 --format jpg --dpi 600
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
    }
)

LOG_MIN = 0
LOG_MAX = 17
N = LOG_MAX - LOG_MIN + 1

# Shared y/x semantics with plot_benchmark.py accuracy figures (log10 cond index 0..17).
LOG10_COND_K_MIN = LOG_MIN
LOG10_COND_K_MAX = LOG_MAX
LOG10_ERR_YLIM = (-16.5, 8.0)
EXPECTED_ROWS = N * N
MACHINE_ACCURACY = 1e-16

ERROR_COLUMNS = {
    "int8_ozaki12_frob_rel_err": ("INT8 Ozaki II, 12 moduli", "#2166ac"),
    "int8_ozaki16_frob_rel_err": ("INT8 Ozaki II, 16 moduli", "#1b7837"),
    "fp8_ozaki10_frob_rel_err": ("FP8 Ozaki II, 10 moduli", "#e66101"),
    "fp8_ozaki12_frob_rel_err": ("FP8 Ozaki II, 12 moduli", "#7b3294"),
}

_OUTPUT_EXTS = {".png", ".pdf", ".jpg", ".jpeg"}


def resolve_output_path(prefix: str, fmt: str | None, *, stem: str = "diagonal_loglog") -> str:
    """Build output file path from prefix, optional --format, or explicit extension."""
    ext = os.path.splitext(prefix)[1].lower()
    if ext in _OUTPUT_EXTS:
        return prefix
    if fmt:
        fmt = fmt.lower()
        suffix = {"jpeg": "jpeg", "jpg": "jpg", "png": "png", "pdf": "pdf"}[fmt]
        return f"{prefix}_{stem}.{suffix}"
    return f"{prefix}_{stem}.png"


def save_figure(fig, out_path: str, dpi: int) -> None:
    ext = os.path.splitext(out_path)[1].lower()
    kwargs: dict = {"bbox_inches": "tight", "dpi": dpi}
    if ext in (".jpg", ".jpeg"):
        kwargs["format"] = "jpeg"
        kwargs["pil_kwargs"] = {"quality": 95, "optimize": True}
    fig.savefig(out_path, **kwargs)


def find_csv(path: str) -> str:
    if os.path.isfile(path):
        return path
    matches = glob.glob(path)
    if not matches:
        raise FileNotFoundError(f"No CSV found for pattern: {path}")
    return max(matches, key=os.path.getmtime)


def load_csv(csv_path: str) -> tuple[dict[str, np.ndarray], dict]:
    grids = {col: np.full((N, N), np.nan) for col in ERROR_COLUMNS}
    rows_read = 0
    ozaki_fast = None
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for col in ERROR_COLUMNS:
            if col not in (reader.fieldnames or []):
                raise ValueError(f"Missing column {col!r}")
        for row in reader:
            rows_read += 1
            a, b = int(row["log10_cond_A"]), int(row["log10_cond_B"])
            if row.get("ozaki_fastmode", "").strip() != "":
                ozaki_fast = int(float(row["ozaki_fastmode"]))
            for col in ERROR_COLUMNS:
                grids[col][a, b] = float(row[col])
    return grids, {
        "rows_read": rows_read,
        "ozaki_fastmode": ozaki_fast,
        "basename": os.path.basename(csv_path),
    }


def diagonal_errors(grid: np.ndarray) -> np.ndarray:
    return np.array([grid[i, i] for i in range(N)], dtype=float)


def row_errors(grid: np.ndarray, log10_cond_a: int) -> np.ndarray:
    if not (LOG_MIN <= log10_cond_a <= LOG_MAX):
        raise ValueError(f"log10_cond_A must be in [{LOG_MIN}, {LOG_MAX}], got {log10_cond_a}")
    return np.array([grid[log10_cond_a, j] for j in range(N)], dtype=float)


def log10_relative_error(err: np.ndarray) -> np.ndarray:
    out = np.full_like(err, np.nan, dtype=float)
    ok = np.isfinite(err) & (err > 0)
    out[ok] = np.log10(err[ok])
    return out


def _diagonal_xlim(cond: np.ndarray) -> tuple[float, float]:
    """Log-scale x limits with margin so edge markers are not clipped."""
    return (cond[0] / 1.8, cond[-1] * 1.8)


def apply_shared_accuracy_axes(
    ax,
    *,
    cond: np.ndarray | None = None,
    linear_log10_cond: bool = False,
) -> None:
    """Fixed scales so ab_table diagonal plots match plot_benchmark accuracy PNGs."""
    ax.set_ylim(*LOG10_ERR_YLIM)
    if linear_log10_cond:
        ax.set_xlim(LOG10_COND_K_MIN, LOG10_COND_K_MAX)
        ax.set_xticks(range(LOG10_COND_K_MIN, LOG10_COND_K_MAX + 1))
    elif cond is not None:
        ax.set_xscale("log")
        ax.set_xlim(*_diagonal_xlim(cond))


def plot_diagonal_loglog(
    grids: dict[str, np.ndarray],
    meta: dict,
    out_path: str,
    *,
    machine_ref: float = MACHINE_ACCURACY,
    dpi: int = 200,
) -> None:
    """Diagonal sweep: log cond on x, log10(relative error) on y; 18 points per series."""
    k = np.arange(N, dtype=float)
    cond = np.power(10.0, k)
    log10_ref = float(np.log10(machine_ref))

    fig, ax = plt.subplots(figsize=(9, 6.5))

    for col, (label, color) in ERROR_COLUMNS.items():
        err = diagonal_errors(grids[col])
        log_err = log10_relative_error(err)
        ok = np.isfinite(log_err)
        if not np.any(ok):
            continue
        ax.semilogx(
            cond[ok],
            log_err[ok],
            marker="o",
            markersize=7,
            linewidth=2,
            color=color,
            label=label,
            clip_on=False,
        )

    ax.semilogx(
        [cond[0], cond[-1]],
        [log10_ref, log10_ref],
        color="red",
        linewidth=2.5,
        linestyle="-",
        label=f"Machine accuracy ($\\log_{{10}}={log10_ref:g}$)",
        zorder=5,
    )

    mode = "fast mode" if meta.get("ozaki_fastmode") == 1 else "accurate mode"
    ax.set_title(f"GEMMul8 Ozaki II emulation error when cond(A) = cond(B) ({mode})")
    ax.set_xlabel(r"Condition number  $\mathrm{cond}(A)=\mathrm{cond}(B)$")
    ax.set_ylabel(
        r"$\log_{10}$ relative Frobenius error  "
        r"$\|C_{\mathrm{ozaki}}-C_{\mathrm{FP64}}\|_F / \|C_{\mathrm{FP64}}\|_F$"
    )
    ax.grid(True, which="major", alpha=0.4, linestyle="--")
    ax.legend(loc="best", framealpha=0.95)
    apply_shared_accuracy_axes(ax, cond=cond)

    save_figure(fig, out_path, dpi)
    plt.close(fig)


def plot_fixed_a_vs_b_loglog(
    grids: dict[str, np.ndarray],
    meta: dict,
    out_path: str,
    log10_cond_a: int,
    *,
    machine_ref: float = MACHINE_ACCURACY,
    dpi: int = 200,
) -> None:
    """Fixed cond(A)=10^a, sweep cond(B)=10^k for k=0..17."""
    k = np.arange(N, dtype=float)
    cond_b = np.power(10.0, k)
    log10_ref = float(np.log10(machine_ref))

    fig, ax = plt.subplots(figsize=(9, 6.5))

    for col, (label, color) in ERROR_COLUMNS.items():
        err = row_errors(grids[col], log10_cond_a)
        log_err = log10_relative_error(err)
        ok = np.isfinite(log_err)
        if not np.any(ok):
            continue
        ax.semilogx(
            cond_b[ok],
            log_err[ok],
            marker="o",
            markersize=7,
            linewidth=2,
            color=color,
            label=label,
            clip_on=False,
        )

    ax.semilogx(
        [cond_b[0], cond_b[-1]],
        [log10_ref, log10_ref],
        color="red",
        linewidth=2.5,
        linestyle="-",
        label=f"Machine accuracy ($\\log_{{10}}={log10_ref:g}$)",
        zorder=5,
    )

    mode = "fast mode" if meta.get("ozaki_fastmode") == 1 else "accurate mode"
    ax.set_title(
        rf"GEMMul8 Ozaki II emulation error with $\mathrm{{cond}}(A)=10^{{{log10_cond_a}}}$ fixed ({mode})"
    )
    ax.set_xlabel(r"Condition number  $\mathrm{cond}(B)$")
    ax.set_ylabel(
        r"$\log_{10}$ relative Frobenius error  "
        r"$\|C_{\mathrm{ozaki}}-C_{\mathrm{FP64}}\|_F / \|C_{\mathrm{FP64}}\|_F$"
    )
    ax.grid(True, which="major", alpha=0.4, linestyle="--")
    ax.legend(loc="best", framealpha=0.95)
    apply_shared_accuracy_axes(ax, cond=cond_b)

    save_figure(fig, out_path, dpi)
    plt.close(fig)


def print_stats(grids: dict[str, np.ndarray]) -> None:
    for col, (label, _) in ERROR_COLUMNS.items():
        d = diagonal_errors(grids[col])
        ok = d[np.isfinite(d) & (d > 0)]
        if ok.size:
            print(
                f"  {label} (diagonal): min={ok.min():.4e} max={ok.max():.4e} @cond=1: {d[0]:.4e}",
                file=sys.stderr,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="CSV file or glob")
    parser.add_argument(
        "prefix",
        nargs="?",
        help="Output prefix or full path with .pdf/.jpg/.jpeg/.png (default: CSV basename)",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "jpg", "jpeg"),
        help="Image format when prefix has no extension (default: png)",
    )
    parser.add_argument(
        "--fix-log10-cond-a",
        type=int,
        metavar="K",
        help="Plot slice with cond(A)=10^K fixed and cond(B) swept 10^0..10^17 (default: diagonal)",
    )
    parser.add_argument(
        "--machine-ref",
        type=float,
        default=MACHINE_ACCURACY,
        help=f"Y value for red reference line (default: {MACHINE_ACCURACY:g})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Raster resolution for PNG/JPEG (default: 200; PDF uses this for embedded bitmaps)",
    )
    args = parser.parse_args()

    csv_path = find_csv(args.csv)
    prefix = args.prefix or os.path.splitext(csv_path)[0]
    if args.fix_log10_cond_a is not None:
        stem = f"condA1e{args.fix_log10_cond_a}_vs_condB"
    else:
        stem = "diagonal_loglog"
    out_path = resolve_output_path(prefix, args.format, stem=stem)

    grids, meta = load_csv(csv_path)
    print(f"Reading {csv_path}", file=sys.stderr)
    if args.fix_log10_cond_a is None:
        print_stats(grids)
    if meta["rows_read"] != EXPECTED_ROWS:
        print(f"Warning: {meta['rows_read']}/{EXPECTED_ROWS} CSV rows", file=sys.stderr)

    if args.fix_log10_cond_a is not None:
        plot_fixed_a_vs_b_loglog(
            grids,
            meta,
            out_path,
            args.fix_log10_cond_a,
            machine_ref=args.machine_ref,
            dpi=args.dpi,
        )
    else:
        plot_diagonal_loglog(
            grids,
            meta,
            out_path,
            machine_ref=args.machine_ref,
            dpi=args.dpi,
        )
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
