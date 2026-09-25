#!/usr/bin/env bash
#
# run_scaling_sweep.sh
# --------------------
# Sweep scaling_gemm_gradeTest2 over:
#   - s = number of slices (moduli), 2..20
#   - t = scaling exponent range (r_k in [-t, t]), 0, 2, 4, ..., 60
#
# For each (s, t) it runs the binary and writes a CSV row with all 4 comparisons
# (8 values: 4 elementwise + 4 Frobenius). Columns:
#   s, t,
#   elem_C2_C1, elem_C3_C1, elem_C4_C1, elem_C3_C4,
#   frob_C1_C2, frob_C1_C3, frob_C1_C4, frob_C3_C4
# where (C1 = A*B FP64 ref, C2 = FP64 scaled, C3 = INT8 scaled, C4 = INT8 unscaled):
#   elem_*  = max_ij |Cx - Cref| / |Cref|
#   frob_*  = ||Cref - Cx||_F / ||Cref||_F
#
# Usage:
#   ./run_scaling_sweep.sh [N] [seed] [out.csv]
#   defaults: N=1024, seed=12345,
#             out=Test2_<date>_size<NxN>_N<N>_seed<SEED>.csv
#                 (e.g. Test2_2026-07-07_size1024x1024_N1024_seed12345.csv)

set -euo pipefail

N="${1:-1024}"
SEED="${2:-12345}"
# Default CSV name: Test2_<date>_size<NxN>_N<N>_seed<SEED>.csv
DEFAULT_OUT="Test2_$(date +%Y-%m-%d)_size${N}x${N}_N${N}_seed${SEED}.csv"
OUT="${3:-$DEFAULT_OUT}"

BINARY="./scaling_gemm_gradeTest2"
GEMMUL8_PATH="${GEMMUL8_PATH:-/home/kswirydo/GEMMul8}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export LD_LIBRARY_PATH="${GEMMUL8_PATH}/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"

if [[ ! -x "$BINARY" ]]; then
    echo "$BINARY not found. Build it first: make scaling_gemm_gradeTest2" >&2
    exit 1
fi

# t = 0, 2, 4, ..., 60 ; s = 2..20
T_VALUES=($(seq 0 2 60))
S_VALUES=($(seq 2 20))

total=$(( ${#T_VALUES[@]} * ${#S_VALUES[@]} ))
done=0

echo "s,t,elem_C2_C1,elem_C3_C1,elem_C4_C1,elem_C3_C4,frob_C1_C2,frob_C1_C3,frob_C1_C4,frob_C3_C4" > "$OUT"

# Extract the numeric value after "= " on the last matching line.
extract() {
    # $1 = output text, $2 = grep pattern
    grep -E "$2" <<< "$1" | tail -n 1 | sed -E 's/.*=[[:space:]]*//'
}

for t in "${T_VALUES[@]}"; do
    for s in "${S_VALUES[@]}"; do
        done=$(( done + 1 ))
        echo "[${done}/${total}] s=${s} t=${t} ..."

        if ! out="$("$BINARY" "$N" "$s" "$SEED" "$t" 2>&1)"; then
            echo "[warn] run failed (s=${s}, t=${t})" >&2
            echo "${s},${t},,,,,,,," >> "$OUT"
            continue
        fi

        elem_c2="$(extract "$out" 'max_ij \|C2 - C1\| / \|C1\|')"
        elem_c3="$(extract "$out" 'max_ij \|C3 - C1\| / \|C1\|')"
        elem_c4="$(extract "$out" 'max_ij \|C4 - C1\| / \|C1\|')"
        elem_c34="$(extract "$out" 'max_ij \|C3 - C4\| / \|C4\|')"
        frob_c2="$(extract "$out" '\|\|C1 - C2\|\|_F / \|\|C1\|\|_F')"
        frob_c3="$(extract "$out" '\|\|C1 - C3\|\|_F / \|\|C1\|\|_F')"
        frob_c4="$(extract "$out" '\|\|C1 - C4\|\|_F / \|\|C1\|\|_F')"
        frob_c34="$(extract "$out" '\|\|C3 - C4\|\|_F / \|\|C4\|\|_F')"

        if [[ -z "$elem_c2" || -z "$elem_c3" || -z "$elem_c4" || -z "$elem_c34" \
              || -z "$frob_c2" || -z "$frob_c3" || -z "$frob_c4" || -z "$frob_c34" ]]; then
            echo "[warn] parse failed (s=${s}, t=${t})" >&2
            echo "${s},${t},,,,,,,," >> "$OUT"
            continue
        fi

        echo "${s},${t},${elem_c2},${elem_c3},${elem_c4},${elem_c34},${frob_c2},${frob_c3},${frob_c4},${frob_c34}" >> "$OUT"
    done
done

echo
echo "Done. Wrote ${OUT}"
