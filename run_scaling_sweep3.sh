#!/usr/bin/env bash
#
# run_scaling_sweep3.sh
# ---------------------
# Sweep the 4th parameter (exponent interval range) of scaling_gemm_gradeTest3
# over 0, 2, 4, ..., 54 on fixed 1024x1024 matrices with a constant seed, and
# summarize the results in two tables:
#   Table 1: Frobenius norms  ||C1|| .. ||C6||   (FP64, INT8 and FP8 variants)
#   Table 2: Relative norm differences (the 7 ratios printed at end of each run)
#
# Same idea as run_scaling_sweep.sh, extended for the FP8 Ozaki-II columns.
#
# Usage:
#   ./run_scaling_sweep3.sh
#
set -euo pipefail

# ---- Configuration ---------------------------------------------------------
N=1024                 # matrix size (N = K = M)
MODULI=16              # INT8 moduli count (2nd arg, constant)
SEED=12345             # RNG seed (3rd arg, constant)
FP8_MODULI=12          # FP8 moduli count (5th arg, constant)
RANGE_START=0          # 4th arg sweep start
RANGE_STEP=2
RANGE_END=54
EXE=./scaling_gemm_gradeTest3

GEMMUL8_PATH="${GEMMUL8_PATH:-/home/kswirydo/GEMMul8}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export LD_LIBRARY_PATH="${GEMMUL8_PATH}/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"

if [[ ! -x "$EXE" ]]; then
    echo "ERROR: executable '$EXE' not found or not executable." >&2
    echo "Build it first with: make scaling_gemm_gradeTest3" >&2
    exit 1
fi

# ---- Detect GPU model (for the CSV file names) -----------------------------
# Capture rocminfo output first (avoid SIGPIPE from an early 'awk exit' under
# 'set -o pipefail'), then take the first AMD Instinct marketing name.
rocm_out="$(rocminfo 2>/dev/null || true)"
GPU_MODEL="$(printf '%s\n' "$rocm_out" \
    | awk -F: '/Marketing Name/ && /Instinct/ && !seen {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; seen=1}')"
[[ -z "$GPU_MODEL" ]] && GPU_MODEL="unknownGPU"
# Sanitize for use in a filename (spaces/slashes -> underscores).
GPU_TAG="$(printf '%s' "$GPU_MODEL" | tr -c 'A-Za-z0-9._-' '_' | sed 's/_\+/_/g; s/^_//; s/_$//')"

DATE_TAG="$(date +%Y%m%d_%H%M%S)"
TAG="${DATE_TAG}_N${N}_seed${SEED}_int8m${MODULI}_fp8m${FP8_MODULI}_${GPU_TAG}"
CSV_NORMS="scaling_sweep3_norms_${TAG}.csv"
CSV_DIFFS="scaling_sweep3_diffs_${TAG}.csv"

# ---- Collect results -------------------------------------------------------
# Temp files hold one aligned row per range value (for the pretty tables).
norms_tbl="$(mktemp)"
diffs_tbl="$(mktemp)"
trap 'rm -f "$norms_tbl" "$diffs_tbl"' EXIT

# CSV headers.
echo "range,||C1||_F,||C2||_F,||C3||_F,||C4||_F,||C5||_F,||C6||_F" > "$CSV_NORMS"
echo "range,|C1-C2|/|C1|,|C1-C3|/|C1|,|C1-C4|/|C1|,|C1-C5|/|C1|,|C1-C6|/|C1|,|C3-C4|/|C4|,|C5-C6|/|C6|" > "$CSV_DIFFS"

# Helper: extract the numeric value at end of the first line containing a
# fixed (literal) substring. Uses index() to avoid regex metacharacter issues
# with the '|' characters in the norm labels.
extract() { awk -v s="$1" 'index($0, s) { print $NF; exit }'; }

for (( r = RANGE_START; r <= RANGE_END; r += RANGE_STEP )); do
    out="$("$EXE" "$N" "$MODULI" "$SEED" "$r" "$FP8_MODULI" 2>/dev/null)"

    c1=$(printf '%s\n' "$out" | extract '||C1||_F =')
    c2=$(printf '%s\n' "$out" | extract '||C2||_F =')
    c3=$(printf '%s\n' "$out" | extract '||C3||_F =')
    c4=$(printf '%s\n' "$out" | extract '||C4||_F =')
    c5=$(printf '%s\n' "$out" | extract '||C5||_F =')
    c6=$(printf '%s\n' "$out" | extract '||C6||_F =')

    d12=$(printf '%s\n' "$out" | extract '||C1 - C2||_F')
    d13=$(printf '%s\n' "$out" | extract '||C1 - C3||_F')
    d14=$(printf '%s\n' "$out" | extract '||C1 - C4||_F')
    d15=$(printf '%s\n' "$out" | extract '||C1 - C5||_F')
    d16=$(printf '%s\n' "$out" | extract '||C1 - C6||_F')
    d34=$(printf '%s\n' "$out" | extract '||C3 - C4||_F')
    d56=$(printf '%s\n' "$out" | extract '||C5 - C6||_F')

    printf '%-8d %-24s %-24s %-24s %-24s %-24s %-24s\n' \
        "$r" "$c1" "$c2" "$c3" "$c4" "$c5" "$c6" >> "$norms_tbl"
    printf '%-8d %-24s %-24s %-24s %-24s %-24s %-24s %-24s\n' \
        "$r" "$d12" "$d13" "$d14" "$d15" "$d16" "$d34" "$d56" >> "$diffs_tbl"

    printf '%d,%s,%s,%s,%s,%s,%s\n' "$r" "$c1" "$c2" "$c3" "$c4" "$c5" "$c6" >> "$CSV_NORMS"
    printf '%d,%s,%s,%s,%s,%s,%s,%s\n' "$r" "$d12" "$d13" "$d14" "$d15" "$d16" "$d34" "$d56" >> "$CSV_DIFFS"
done

# ---- Display raw CSVs first ------------------------------------------------
echo "############################################################################################################################"
echo "# CSV OUTPUT"
echo "############################################################################################################################"
echo
echo ">>> $CSV_NORMS"
cat "$CSV_NORMS"
echo
echo ">>> $CSV_DIFFS"
cat "$CSV_DIFFS"
echo

# ---- Print tables ----------------------------------------------------------
echo "============================================================================================================================"
echo "scaling_gemm_gradeTest3 sweep : N=$N, INT8 moduli=$MODULI, FP8 moduli=$FP8_MODULI, seed=$SEED,"
echo "                               range=${RANGE_START}..${RANGE_END} step ${RANGE_STEP}"
echo "GPU: $GPU_MODEL"
echo "============================================================================================================================"
echo
echo "TABLE 1 : Frobenius norms"
echo "  C1 = A*B (FP64)   C2 = A_scal*B_scal (FP64)   C3/C4 = INT8 scaled/unscaled   C5/C6 = FP8 scaled/unscaled"
echo "----------------------------------------------------------------------------------------------------------------------------"
printf '%-8s %-24s %-24s %-24s %-24s %-24s %-24s\n' \
    "range" "||C1||_F" "||C2||_F" "||C3||_F" "||C4||_F" "||C5||_F" "||C6||_F"
echo "----------------------------------------------------------------------------------------------------------------------------"
cat "$norms_tbl"
echo
echo "TABLE 2 : Relative norm differences"
echo "----------------------------------------------------------------------------------------------------------------------------"
printf '%-8s %-24s %-24s %-24s %-24s %-24s %-24s %-24s\n' \
    "range" "|C1-C2|/|C1|" "|C1-C3|/|C1|" "|C1-C4|/|C1|" "|C1-C5|/|C1|" "|C1-C6|/|C1|" "|C3-C4|/|C4|" "|C5-C6|/|C6|"
echo "         (FP64 scaled)            (INT8 scaled)            (INT8 unscaled)          (FP8 scaled)             (FP8 unscaled)           (INT8 sc vs unsc)        (FP8 sc vs unsc)"
echo "----------------------------------------------------------------------------------------------------------------------------"
cat "$diffs_tbl"
echo "----------------------------------------------------------------------------------------------------------------------------"
echo
echo "CSV files saved:"
echo "  $CSV_NORMS"
echo "  $CSV_DIFFS"
