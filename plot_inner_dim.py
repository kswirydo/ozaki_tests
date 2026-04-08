#!/usr/bin/env python3
"""
Plot inner dimension benchmark results.

Parses the summary table output from inner_dim_benchmark and generates:
1. Performance plot: K (inner dim) vs TFLOPS
2. Accuracy plot: K (inner dim) vs Relative Error

Usage:
    python plot_inner_dim.py <results_file>
    python plot_inner_dim.py results_16384.txt
"""

import sys
import os
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_results_file(filepath):
    """Parse the inner_dim_benchmark results file."""
    data = {
        'K': [],
        'N': None,
        'native_tflops': [],
        'ozaki12_tflops': [],
        'ozaki12_error': [],
        'ozaki16_tflops': [],
        'ozaki16_error': [],
    }
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Parse data rows: | 16384x16384x16384 | 70.48 TF | 162.03 TF | 1.46e-12 | 125.34 TF | 4.27e-15 |
            if line.startswith('|') and 'x' in line and 'TF' in line:
                # Remove leading/trailing pipes and split
                parts = [p.strip() for p in line.split('|') if p.strip()]
                
                if len(parts) >= 6:
                    # Parse dimensions (NxKxN)
                    dims_str = parts[0].strip()
                    dims_match = re.match(r'(\d+)x(\d+)x(\d+)', dims_str)
                    if dims_match:
                        N = int(dims_match.group(1))
                        K = int(dims_match.group(2))
                        
                        if data['N'] is None:
                            data['N'] = N
                        
                        data['K'].append(K)
                        
                        # Parse native TFLOPS
                        native_match = re.search(r'([\d.]+)\s*TF', parts[1])
                        if native_match:
                            data['native_tflops'].append(float(native_match.group(1)))
                        
                        # Parse Ozaki 12 TFLOPS
                        ozaki12_perf_match = re.search(r'([\d.]+)\s*TF', parts[2])
                        if ozaki12_perf_match:
                            data['ozaki12_tflops'].append(float(ozaki12_perf_match.group(1)))
                        
                        # Parse Ozaki 12 error
                        ozaki12_err_match = re.search(r'([\d.]+e[+-]?\d+)', parts[3])
                        if ozaki12_err_match:
                            data['ozaki12_error'].append(float(ozaki12_err_match.group(1)))
                        
                        # Parse Ozaki 16 TFLOPS
                        ozaki16_perf_match = re.search(r'([\d.]+)\s*TF', parts[4])
                        if ozaki16_perf_match:
                            data['ozaki16_tflops'].append(float(ozaki16_perf_match.group(1)))
                        
                        # Parse Ozaki 16 error
                        ozaki16_err_match = re.search(r'([\d.]+e[+-]?\d+)', parts[5])
                        if ozaki16_err_match:
                            data['ozaki16_error'].append(float(ozaki16_err_match.group(1)))
    
    return data

def plot_performance(data, output_file):
    """Generate performance plot: K vs TFLOPS."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    K = np.array(data['K'])
    
    # Sort by K for proper line plotting
    sort_idx = np.argsort(K)
    K_sorted = K[sort_idx]
    
    native = np.array(data['native_tflops'])[sort_idx]
    ozaki12 = np.array(data['ozaki12_tflops'])[sort_idx]
    ozaki16 = np.array(data['ozaki16_tflops'])[sort_idx]
    
    # Plot lines
    ax.plot(K_sorted, native, 'o-', color='blue', linewidth=2, markersize=4, 
            label='Native FP64 GEMM')
    ax.plot(K_sorted, ozaki12, 's-', color='green', linewidth=2, markersize=4,
            label='Ozaki-II (12 splits)')
    ax.plot(K_sorted, ozaki16, '^-', color='red', linewidth=2, markersize=4,
            label='Ozaki-II (16 splits)')
    
    ax.set_xlabel('Inner Dimension K', fontsize=12)
    ax.set_ylabel('Performance (TFLOPS)', fontsize=12)
    ax.set_title(f'GEMM Performance: A({data["N"]}×K) × B(K×{data["N"]}) = C({data["N"]}×{data["N"]})', 
                 fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Set reasonable x-axis limits
    ax.set_xlim(min(K_sorted) - 100, max(K_sorted) + 100)
    ax.set_ylim(0, max(max(ozaki12), max(native)) * 1.1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

def plot_accuracy(data, output_file):
    """Generate accuracy plot: K vs Relative Error (log scale)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    K = np.array(data['K'])
    
    # Sort by K for proper line plotting
    sort_idx = np.argsort(K)
    K_sorted = K[sort_idx]
    
    ozaki12_err = np.array(data['ozaki12_error'])[sort_idx]
    ozaki16_err = np.array(data['ozaki16_error'])[sort_idx]
    
    # Machine epsilon for FP64
    machine_eps = 2.2e-16
    
    # Plot lines
    ax.semilogy(K_sorted, ozaki12_err, 's-', color='green', linewidth=2, markersize=4,
                label='Ozaki-II (12 splits)')
    ax.semilogy(K_sorted, ozaki16_err, '^-', color='red', linewidth=2, markersize=4,
                label='Ozaki-II (16 splits)')
    
    # Machine epsilon reference line (Native GEMM assumed to be at machine precision)
    ax.axhline(y=machine_eps, color='blue', linestyle='--', linewidth=2,
               label=f'Machine epsilon ({machine_eps:.1e})')
    
    ax.set_xlabel('Inner Dimension K', fontsize=12)
    ax.set_ylabel('Relative Frobenius Error', fontsize=12)
    ax.set_title(f'GEMM Accuracy: A({data["N"]}×K) × B(K×{data["N"]}) = C({data["N"]}×{data["N"]})', 
                 fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    # Set reasonable x-axis limits
    ax.set_xlim(min(K_sorted) - 100, max(K_sorted) + 100)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_inner_dim.py <results_file>")
        print("Example: python plot_inner_dim.py results_16384.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    
    # Parse data
    print(f"Parsing: {input_file}")
    data = parse_results_file(input_file)
    
    if not data['K']:
        print("Error: No data found in file")
        sys.exit(1)
    
    print(f"Found {len(data['K'])} data points")
    print(f"Outer dimension N = {data['N']}")
    print(f"K range: {min(data['K'])} to {max(data['K'])}")
    
    # Generate output filenames
    base_name = os.path.splitext(input_file)[0]
    perf_output = f"{base_name}_performance.png"
    acc_output = f"{base_name}_accuracy.png"
    
    # Generate plots
    print("\nGenerating plots...")
    plot_performance(data, perf_output)
    plot_accuracy(data, acc_output)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
