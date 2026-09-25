#!/bin/bash
#
# Collect GEMMul8 profiling data using table_benchmark
#
# Runs table_benchmark with various matrix sizes and collects
# profiling data (skipping warmup iterations).
#

# Allow SIZES to be overridden via environment variable
SIZES="${SIZES:-384 512 1024 2048 4096 8192 16384 24576 32768 40960}"
NUM_WARMUP=10  # Must match table_benchmark.cu

# Temporary file for all profile data
RAWDATA=$(mktemp)

echo "GEMMul8 Profiling Data Collection"
echo "=================================="
echo ""

for SIZE in $SIZES; do
    echo "Running benchmark for size $SIZE..."
    
    # Run benchmark and capture stderr (where profiling goes)
    GEMMUL8_PROFILE=1 ./table_benchmark $SIZE 2>&1 | \
        grep -E "(\[GEMMUL8 PROFILE\]|Scaling|INT8 GEMM|Reconstruction|TOTAL)" | \
        sed "s/^/${SIZE} /" >> "$RAWDATA"
done

echo ""
echo "Processing data..."
echo ""

# Use Python to parse and display results
python3 - "$RAWDATA" "$NUM_WARMUP" << 'PYTHON_SCRIPT'
import sys
import re
from collections import defaultdict

rawfile = sys.argv[1]
num_warmup = int(sys.argv[2])

# Parse raw data
data = defaultdict(list)
current_size = None
current_moduli = None
current_entry = {}

with open(rawfile, 'r') as f:
    for line in f:
        line = line.strip()
        
        # Match profile header: SIZE [GEMMUL8 PROFILE] DGEMM MxNxK mod=X ...
        header_match = re.search(r'^(\d+)\s+\[GEMMUL8 PROFILE\].*mod=(\d+)', line)
        if header_match:
            # Save previous entry if exists
            if current_size and current_moduli and current_entry:
                data[(current_size, current_moduli)].append(current_entry)
            
            current_size = header_match.group(1)
            current_moduli = header_match.group(2)
            current_entry = {}
            continue
        
        # Match timing lines
        scale_match = re.search(r'Scaling.*?(\d+\.?\d*)\s*ms', line)
        if scale_match:
            current_entry['scale'] = float(scale_match.group(1))
            continue
            
        gemm_match = re.search(r'INT8 GEMM.*?(\d+\.?\d*)\s*ms', line)
        if gemm_match:
            current_entry['gemm'] = float(gemm_match.group(1))
            continue
            
        recon_match = re.search(r'Reconstruction.*?(\d+\.?\d*)\s*ms', line)
        if recon_match:
            current_entry['recon'] = float(recon_match.group(1))
            continue
            
        total_match = re.search(r'TOTAL.*?(\d+\.?\d*)\s*ms', line)
        if total_match:
            current_entry['total'] = float(total_match.group(1))
            continue

# Save last entry
if current_size and current_moduli and current_entry:
    data[(current_size, current_moduli)].append(current_entry)

# Skip warmup and compute averages
# For each (size, moduli) pair, skip first num_warmup entries
results = {}
for key, entries in data.items():
    # Debug: show how many entries we have
    total_entries = len(entries)
    
    # Skip first num_warmup entries (these are warmup iterations)
    entries = entries[num_warmup:]
    
    if entries:
        n = len(entries)
        
        # Use median instead of mean to filter outliers
        def median(values):
            s = sorted(values)
            mid = len(s) // 2
            if len(s) % 2 == 0:
                return (s[mid-1] + s[mid]) / 2
            return s[mid]
        
        scale_vals = [e.get('scale', 0) for e in entries]
        gemm_vals = [e.get('gemm', 0) for e in entries]
        recon_vals = [e.get('recon', 0) for e in entries]
        total_vals = [e.get('total', 0) for e in entries]
        
        med_scale = median(scale_vals)
        med_gemm = median(gemm_vals)
        med_recon = median(recon_vals)
        med_total = median(total_vals)
        
        results[key] = {
            'scale': med_scale,
            'gemm': med_gemm,
            'recon': med_recon,
            'total': med_total,
            'count': n,
            'total_entries': total_entries,
            'skipped': num_warmup
        }
        
        # Debug output with outlier info
        outliers = sum(1 for v in scale_vals if v > med_scale * 10)
        print(f"  Size {key[0]}, Moduli {key[1]}: {total_entries} total, skipped {num_warmup}, median of {n} ({outliers} outliers)", file=sys.stderr)

sizes = [384, 512, 1024, 2048, 4096, 8192, 16384, 24576, 32768, 40960]

# Table 1: All data (12 and 16 moduli)
print()
print("=" * 105)
print("                               GEMMul8 PROFILING RESULTS")
print(f"                         (Median over timed iterations, {num_warmup} warmup skipped)")
print("=" * 105)
print()
print("TABLE 1: All Configurations (Absolute Times)")
print("+--------+--------+------------+------------+------------+------------+")
print("|  Size  | Moduli | Scale (ms) | GEMM (ms)  | Recon (ms) | Total (ms) |")
print("+--------+--------+------------+------------+------------+------------+")

for size in sizes:
    for moduli in [12, 16]:
        key = (str(size), str(moduli))
        if key in results:
            r = results[key]
            print(f"| {size:6} | {moduli:6} | {r['scale']:10.3f} | {r['gemm']:10.3f} | {r['recon']:10.3f} | {r['total']:10.3f} |")

print("+--------+--------+------------+------------+------------+------------+")

# Table 2: 12 moduli only (Absolute Times)
print()
print("TABLE 2: 12 Moduli Only (Absolute Times)")
print("+--------+------------+------------+------------+------------+")
print("|  Size  | Scale (ms) | GEMM (ms)  | Recon (ms) | Total (ms) |")
print("+--------+------------+------------+------------+------------+")

for size in sizes:
    key = (str(size), '12')
    if key in results:
        r = results[key]
        print(f"| {size:6} | {r['scale']:10.3f} | {r['gemm']:10.3f} | {r['recon']:10.3f} | {r['total']:10.3f} |")

print("+--------+------------+------------+------------+------------+")

# Table 3: 16 moduli only (Absolute Times)
print()
print("TABLE 3: 16 Moduli Only (Absolute Times)")
print("+--------+------------+------------+------------+------------+")
print("|  Size  | Scale (ms) | GEMM (ms)  | Recon (ms) | Total (ms) |")
print("+--------+------------+------------+------------+------------+")

for size in sizes:
    key = (str(size), '16')
    if key in results:
        r = results[key]
        print(f"| {size:6} | {r['scale']:10.3f} | {r['gemm']:10.3f} | {r['recon']:10.3f} | {r['total']:10.3f} |")

print("+--------+------------+------------+------------+------------+")

# Table 4: 12 moduli (Percentage)
print()
print("TABLE 4: 12 Moduli Only (Percentage of Execution)")
print("+--------+------------+------------+------------+")
print("|  Size  | Scale (%)  | GEMM (%)   | Recon (%)  |")
print("+--------+------------+------------+------------+")

for size in sizes:
    key = (str(size), '12')
    if key in results:
        r = results[key]
        # Use sum of components (not median of total) for accurate percentages
        total = r['scale'] + r['gemm'] + r['recon']
        if total <= 0:
            total = 1
        scale_pct = r['scale'] / total * 100
        gemm_pct = r['gemm'] / total * 100
        recon_pct = r['recon'] / total * 100
        print(f"| {size:6} | {scale_pct:10.1f} | {gemm_pct:10.1f} | {recon_pct:10.1f} |")

print("+--------+------------+------------+------------+")

# Table 5: 16 moduli (Percentage)
print()
print("TABLE 5: 16 Moduli Only (Percentage of Execution)")
print("+--------+------------+------------+------------+")
print("|  Size  | Scale (%)  | GEMM (%)   | Recon (%)  |")
print("+--------+------------+------------+------------+")

for size in sizes:
    key = (str(size), '16')
    if key in results:
        r = results[key]
        # Use sum of components (not median of total) for accurate percentages
        total = r['scale'] + r['gemm'] + r['recon']
        if total <= 0:
            total = 1
        scale_pct = r['scale'] / total * 100
        gemm_pct = r['gemm'] / total * 100
        recon_pct = r['recon'] / total * 100
        print(f"| {size:6} | {scale_pct:10.1f} | {gemm_pct:10.1f} | {recon_pct:10.1f} |")

print("+--------+------------+------------+------------+")

print()
print("Legend:")
print("  Scale:  Scaling/splitting phase (FP64 -> INT8 decomposition)")
print("  GEMM:   INT8 Tensor Core GEMM operations")  
print("  Recon:  Reconstruction phase (combine results back to FP64)")
print("=" * 105)

PYTHON_SCRIPT

# Cleanup
rm -f "$RAWDATA"

echo ""
echo "Done!"
