#!/usr/bin/env python3
"""
Plot benchmark results from gemm_benchmark or ab_gemm_benchmark CSV output.

Generates 6 figures:
- 3 performance plots (8, 12, 16 moduli): condition number vs TFLOPS
- 3 accuracy plots (8, 12, 16 moduli): condition number vs relative error

Usage:
    python plot_benchmark.py <csv_file>
    python plot_benchmark.py <matrix_folder>  # finds latest CSV in folder
"""

import sys
import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    from plot_ab_table_heatmap import (
        LOG10_COND_K_MAX,
        LOG10_COND_K_MIN,
        LOG10_ERR_YLIM,
        MACHINE_ACCURACY,
    )
except ImportError:
    LOG10_COND_K_MIN = 0
    LOG10_COND_K_MAX = 17
    LOG10_ERR_YLIM = (-16.5, 8.0)
    MACHINE_ACCURACY = 1e-16


def find_latest_csv(folder):
    """Find the most recent benchmark CSV file in the folder."""
    patterns = [
        os.path.join(folder, "ab_gemm_benchmark_*.csv"),
        os.path.join(folder, "gemm_benchmark_*.csv"),
        os.path.join(folder, "simple_gemm_benchmark_*.csv"),
    ]
    
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern))
    
    if not all_files:
        return None
    return max(all_files, key=os.path.getmtime)

def get_matrix_dimensions(folder):
    """Extract matrix dimensions from A_* and B_* filenames in the folder."""
    # Look for A_NxK_*.txt files (ab_gemm_benchmark style)
    a_files = glob.glob(os.path.join(folder, "A_*x*.txt"))
    b_files = glob.glob(os.path.join(folder, "B_*x*.txt"))
    
    # Also look for M_cond_*.txt files (gemm_benchmark style - square matrices)
    m_files = glob.glob(os.path.join(folder, "M_cond_*.txt"))
    
    N, K, M = None, None, None
    
    if a_files:
        basename = os.path.basename(a_files[0])
        match = re.match(r'A_(\d+)x(\d+)', basename)
        if match:
            N = int(match.group(1))
            K = int(match.group(2))
    
    if b_files:
        basename = os.path.basename(b_files[0])
        match = re.match(r'B_(\d+)x(\d+)', basename)
        if match:
            K_check = int(match.group(1))
            M = int(match.group(2))
            if K is None:
                K = K_check
    
    # For square matrices from gemm_benchmark (M_cond_*.txt files)
    if m_files and N is None:
        folder_name = os.path.basename(folder.rstrip('/'))
        # Try patterns like "my_matrices4096square" or "matrices_4096"
        match = re.search(r'(\d+)square', folder_name)
        if match:
            size = int(match.group(1))
            N, K, M = size, size, size
        else:
            match = re.search(r'(\d+)x(\d+)', folder_name)
            if match:
                N = int(match.group(1))
                K = int(match.group(2))
                M = K
            else:
                # Try to read first line of matrix file to get size
                try:
                    with open(m_files[0], 'r') as f:
                        first_line = f.readline().strip().split()
                        cols = len(first_line)
                        # Count lines for rows
                        rows = 1
                        for _ in f:
                            rows += 1
                        N, K, M = rows, cols, cols
                except:
                    pass
    
    return N, K, M

def parse_gemm_benchmark_csv(csv_file):
    """Parse gemm_benchmark CSV which has condition numbers in comment lines."""
    data_rows = []
    current_cond = None
    
    with open(csv_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Check for condition number in comments
            if line.startswith('#'):
                match = re.search(r'Condition number: 10\^(\d+)', line)
                if match:
                    current_cond = int(match.group(1))
                continue
            
            # Skip header line
            if line.startswith('method,'):
                continue
            
            # Parse data line
            parts = line.split(',')
            if len(parts) >= 8 and current_cond is not None:
                try:
                    row = {
                        'log10_cond': current_cond,
                        'method': parts[0],
                        'num_moduli': int(parts[1]) if parts[1] != '0' else 0,
                        'fastmode': parts[2],
                        'time_ms': float(parts[3]),
                        'tflops': float(parts[4]),
                        'frob_rel_err': float(parts[5]),
                        'max_rel_err': float(parts[6]),
                        'avg_rel_err': float(parts[7]),
                    }
                    # Normalize method names
                    if row['method'] == 'rocBLAS':
                        row['method'] = 'rocBLAS_FP64'
                    elif row['method'] == 'Ozaki-II':
                        row['method'] = 'Ozaki-II_EMU'
                    data_rows.append(row)
                except (ValueError, IndexError):
                    continue
    
    return pd.DataFrame(data_rows)

def parse_csv(csv_file):
    """Parse the benchmark CSV file, detecting format automatically."""
    # Check if it's gemm_benchmark format (has condition numbers in comments)
    with open(csv_file, 'r') as f:
        content = f.read()
    
    if 'Condition number: 10^' in content:
        # gemm_benchmark format
        df = parse_gemm_benchmark_csv(csv_file)
    else:
        # ab_gemm_benchmark format - has log10_cond column
        df = pd.read_csv(csv_file, comment='#')
    
    # Extract metadata
    metadata = {}
    with open(csv_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                if ':' in line:
                    parts = line[1:].strip().split(':', 1)
                    if len(parts) == 2:
                        metadata[parts[0].strip()] = parts[1].strip()
            else:
                break
    
    return df, metadata

def plot_performance(df, N, K, M, moduli_list, output_prefix):
    """Generate performance plots for specified moduli counts."""
    
    if M == K and K == N:
        dims_str = f"{N}x{N} (A*A)"
    else:
        dims_str = f"A({N}x{K}) * B({K}x{M}) = C({N}x{M})"
    
    for num_moduli in moduli_list:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Filter data for this moduli count
        ozaki_data = df[(df['method'] == 'Ozaki-II_EMU') & (df['num_moduli'] == num_moduli)]
        rocblas_data = df[df['method'] == 'rocBLAS_FP64']
        
        if ozaki_data.empty:
            print(f"No data for {num_moduli} moduli, skipping...")
            plt.close()
            continue
        
        # Plot rocBLAS (native FP64) as reference line
        if not rocblas_data.empty:
            rocblas_perf = rocblas_data['tflops'].mean()
            ax.axhline(y=rocblas_perf, color='red', linestyle='--', 
                      linewidth=2, label=f'rocBLAS FP64 ({rocblas_perf:.2f} TFLOPS)')
        
        # Plot Ozaki-II for both modes
        for mode, marker, color in [('accurate', 'o', 'blue'), ('fast', 's', 'green')]:
            mode_data = ozaki_data[ozaki_data['fastmode'] == mode]
            if not mode_data.empty:
                x = mode_data['log10_cond'].values
                y = mode_data['tflops'].values
                ax.plot(x, y, marker=marker, linewidth=2, markersize=8,
                       color=color, label=f'Ozaki-II {mode}')
        
        ax.set_xlabel('log₁₀(Condition Number)', fontsize=12)
        ax.set_ylabel('Performance (TFLOPS)', fontsize=12)
        ax.set_title(f'GEMM Performance: {dims_str}\n{num_moduli} Moduli', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Set x-ticks based on data
        x_min = int(df['log10_cond'].min())
        x_max = int(df['log10_cond'].max())
        ax.set_xticks(range(x_min, x_max + 1))
        
        # Save figure
        output_file = f"{output_prefix}_perf_{num_moduli}moduli.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_file}")

def plot_accuracy(df, N, K, M, moduli_list, output_prefix):
    """Generate accuracy plots for specified moduli counts."""
    
    if M == K and K == N:
        dims_str = f"{N}x{N} (A*A)"
    else:
        dims_str = f"A({N}x{K}) * B({K}x{M}) = C({N}x{M})"
    
    for num_moduli in moduli_list:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Filter data for this moduli count
        ozaki_data = df[(df['method'] == 'Ozaki-II_EMU') & (df['num_moduli'] == num_moduli)]
        
        if ozaki_data.empty:
            print(f"No data for {num_moduli} moduli, skipping...")
            plt.close()
            continue
        
        # Plot Ozaki-II for both modes
        for mode, marker, color in [('accurate', 'o', 'blue'), ('fast', 's', 'green')]:
            mode_data = ozaki_data[ozaki_data['fastmode'] == mode]
            if not mode_data.empty:
                x = mode_data['log10_cond'].values
                y = mode_data['frob_rel_err'].values
                # Convert to log10, handling zeros
                y_log = np.log10(np.maximum(y, 1e-20))
                ax.plot(x, y_log, marker=marker, linewidth=2, markersize=8,
                       color=color, label=f'Ozaki-II {mode}')
        
        ax.axhline(
            y=np.log10(MACHINE_ACCURACY),
            color='gray',
            linestyle=':',
            linewidth=1.5,
            label='Machine epsilon (FP64)',
        )

        ax.set_xlabel('log₁₀(Condition Number)', fontsize=12)
        ax.set_ylabel('log₁₀(Relative Frobenius Error)', fontsize=12)
        ax.set_title(f'GEMM Accuracy: {dims_str}\n{num_moduli} Moduli', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(LOG10_COND_K_MIN, LOG10_COND_K_MAX)
        ax.set_xticks(range(LOG10_COND_K_MIN, LOG10_COND_K_MAX + 1))
        ax.set_ylim(*LOG10_ERR_YLIM)
        
        # Save figure
        output_file = f"{output_prefix}_accuracy_{num_moduli}moduli.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_benchmark.py <csv_file_or_folder>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # Determine if input is a file or folder
    if os.path.isfile(input_path):
        csv_file = input_path
        folder = os.path.dirname(csv_file) or '.'
    elif os.path.isdir(input_path):
        folder = input_path
        csv_file = find_latest_csv(folder)
        if csv_file is None:
            print(f"No benchmark CSV files found in {folder}")
            sys.exit(1)
        print(f"Using latest CSV: {csv_file}")
    else:
        print(f"Error: {input_path} not found")
        sys.exit(1)
    
    # Get matrix dimensions from filenames
    N, K, M = get_matrix_dimensions(folder)
    if N is None or K is None or M is None:
        print(f"Warning: Could not determine matrix dimensions from filenames")
        N, K, M = "?", "?", "?"
    else:
        print(f"Matrix dimensions: N={N}, K={K}, M={M}")
    
    # Parse CSV
    df, metadata = parse_csv(csv_file)
    print(f"Loaded {len(df)} data rows")
    
    if df.empty:
        print("Error: No data found in CSV")
        sys.exit(1)
    
    # Output prefix (same directory as CSV, without extension)
    output_prefix = os.path.splitext(csv_file)[0]
    
    # Moduli to plot
    moduli_list = [8, 12, 16]
    
    # Check which moduli are available
    available_moduli = df[df['method'] == 'Ozaki-II_EMU']['num_moduli'].unique()
    moduli_list = [m for m in moduli_list if m in available_moduli]
    
    if not moduli_list:
        print("No data for moduli 8, 12, or 16. Available moduli:", sorted(available_moduli))
        # Use top 3 available moduli
        moduli_list = sorted(available_moduli)[-3:] if len(available_moduli) >= 3 else sorted(available_moduli)
        print(f"Using moduli: {moduli_list}")
    
    # Generate plots
    print("\nGenerating performance plots...")
    plot_performance(df, N, K, M, moduli_list, output_prefix)
    
    print("\nGenerating accuracy plots...")
    plot_accuracy(df, N, K, M, moduli_list, output_prefix)
    
    print(f"\nDone! Generated {len(moduli_list) * 2} figures.")

if __name__ == "__main__":
    main()
