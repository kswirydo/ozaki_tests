#!/bin/bash
#
# Run table benchmark (FP8 backend) with GPU power/energy measurement for GEMM phases only.
#
# Usage:
#   ./run_power_benchmark_fp8.sh <size1> [size2] [size3] ...
#   ./run_power_benchmark_fp8.sh 4096 8192 16384
#
# Output files:
#   power_profile_fp8_<timestamp>.txt  - Raw power samples from power_profiler
#   benchmark_log_fp8_<timestamp>.txt  - Benchmark stderr with GEMM phase timestamps
#   benchmark_out_fp8_<timestamp>.txt  - Benchmark stdout (summary table)
#   power_results_fp8_<timestamp>.json - Parsed power/energy results for each GEMM phase
#   power_results_fp8_<timestamp>_energy_chart.png - Energy comparison bar chart
#

set -e

# Configuration
POWER_PROFILER_PATH="${POWER_PROFILER_PATH:-/home/kswirydo/power_analysis-main/power_profiler}"
ROCEMU_PATH="${ROCEMU_PATH:-/home/kswirydo/rocEMU}"
GEMMUL8_PATH="${GEMMUL8_PATH:-/home/kswirydo/GEMMul8/GEMMul8}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

SAMPLING_FREQ=5000     # Hz (samples per second) - 0.2ms between samples
MAX_DURATION=3600      # Max profiling time in seconds (1 hour)

# Timestamp for output files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Output files (with fp8 in name)
POWER_PROFILE="power_profile_fp8_${TIMESTAMP}.txt"
BENCHMARK_LOG="benchmark_log_fp8_${TIMESTAMP}.txt"
BENCHMARK_OUT="benchmark_out_fp8_${TIMESTAMP}.txt"
POWER_RESULTS="power_results_fp8_${TIMESTAMP}.json"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <size1> [size2] [size3] ..."
    echo "Example: $0 4096 8192 16384"
    exit 1
fi

SIZES="$@"
echo "=============================================="
echo "Power Benchmark for rocEMU Table Benchmark"
echo "Backend: FP8 Tensor Cores"
echo "=============================================="
echo "Matrix sizes: $SIZES"
echo "Timestamp: $TIMESTAMP"
echo "Power sampling: ${SAMPLING_FREQ} Hz"
echo ""

# Step 1: Build power profiler if needed
echo "[1/6] Checking power profiler..."
if [ ! -f "${POWER_PROFILER_PATH}/power_profiler" ]; then
    echo "  Building power profiler..."
    pushd "${POWER_PROFILER_PATH}" > /dev/null
    make clean 2>/dev/null || true
    make
    popd > /dev/null
fi
echo "  Power profiler ready: ${POWER_PROFILER_PATH}/power_profiler"

# Step 2: Build table_benchmark_power_fp8 if needed
echo "[2/6] Checking table_benchmark_power_fp8..."
pushd "${ROCEMU_PATH}" > /dev/null
if [ ! -f "table_benchmark_power_fp8" ] || [ "table_benchmark_power_fp8.cu" -nt "table_benchmark_power_fp8" ]; then
    echo "  Building table_benchmark_power_fp8..."
    make table_benchmark_power_fp8
fi
popd > /dev/null
echo "  Benchmark ready: ${ROCEMU_PATH}/table_benchmark_power_fp8"

# Step 3: Start power profiler in background
echo "[3/6] Starting power profiler..."
"${POWER_PROFILER_PATH}/power_profiler" "${ROCEMU_PATH}/${POWER_PROFILE}" ${MAX_DURATION} ${SAMPLING_FREQ} &
PROFILER_PID=$!
echo "  Power profiler PID: ${PROFILER_PID}"
sleep 1  # Give profiler time to initialize

# Step 4: Run the benchmark
echo "[4/6] Running FP8 benchmark..."
echo "  Output: ${BENCHMARK_OUT}"
echo "  Log: ${BENCHMARK_LOG}"

cd "${ROCEMU_PATH}"
export LD_LIBRARY_PATH="${GEMMUL8_PATH}/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH}"

./table_benchmark_power_fp8 $SIZES > "${BENCHMARK_OUT}" 2> "${BENCHMARK_LOG}"
BENCH_EXIT=$?

# Step 5: Stop power profiler
echo "[5/6] Stopping power profiler..."
kill ${PROFILER_PID} 2>/dev/null || true
wait ${PROFILER_PID} 2>/dev/null || true
sleep 1
echo "  Power profile saved: ${POWER_PROFILE}"

# Step 6: Analyze power data for GEMM phases only
echo "[6/6] Analyzing power data..."
python3 "${ROCEMU_PATH}/analyze_gemm_power_fp8.py" \
    --power "${ROCEMU_PATH}/${POWER_PROFILE}" \
    --log "${ROCEMU_PATH}/${BENCHMARK_LOG}" \
    --output "${ROCEMU_PATH}/${POWER_RESULTS}"

ENERGY_CHART="${POWER_RESULTS%.json}_energy_chart.png"

echo ""
echo "=============================================="
echo "FP8 Benchmark completed!"
echo "=============================================="
echo "Files created:"
echo "  - ${POWER_PROFILE}  (raw power samples)"
echo "  - ${BENCHMARK_LOG}  (benchmark log with timestamps)"
echo "  - ${BENCHMARK_OUT}  (benchmark summary table)"
echo "  - ${POWER_RESULTS}  (GEMM power/energy results)"
echo "  - ${ENERGY_CHART}   (energy comparison bar chart)"
echo ""

# Display the performance summary table
echo "Performance Summary (FP8 Backend):"
cat "${BENCHMARK_OUT}"

exit ${BENCH_EXIT}
