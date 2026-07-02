#!/usr/bin/env python3
"""
Plot awkward matrix benchmark results as bar graphs.

Compares performance between aligned (N) and misaligned (N+1) matrix sizes.
Groups: Native FP64, Ozaki-II 12 splits, Ozaki-II 16 splits

Usage:
    python plot_awkward.py <results_file>
    python plot_awkward.py awkward_matrices_cleanup.txt
"""

import sys
import os
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_results_file(filepath):
    """Parse the benchmark results file."""
    data = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Parse data rows
            if line.startswith('|') and 'TF' in line and 'Method' not in line:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                
                if len(parts) >= 5:
                    method = parts[0].strip()
                    
                    # Skip header-like rows
                    if 'Method' in method or '---' in method:
                        continue
                    
                    try:
                        M = int(parts[1])
                        N = int(parts[2])
                        K = int(parts[3])
                        
                        # Parse performance
                        perf_match = re.search(r'([\d.]+)\s*TF', parts[4])
                        if perf_match:
                            tflops = float(perf_match.group(1))
                            
                            data.append({
                                'method': method,
                                'M': M,
                                'N': N,
                                'K': K,
                                'tflops': tflops
                            })
                    except (ValueError, IndexError):
                        continue
    
    return data

def group_by_size_pairs(data):
    """Group data by size pairs (N and N+1)."""
    # Get unique sizes
    sizes = sorted(set(d['M'] for d in data))
    
    # Find pairs (consecutive sizes differing by 1)
    pairs = []
    used = set()
    for s in sizes:
        if s in used:
            continue
        if s + 1 in sizes:
            pairs.append((s, s + 1))
            used.add(s)
            used.add(s + 1)
        else:
            # Single size without pair
            pairs.append((s,))
            used.add(s)
    
    return pairs

def get_performance(data, size, method_pattern):
    """Get performance for a specific size and method."""
    for d in data:
        if d['M'] == size and method_pattern in d['method']:
            return d['tflops']
    return 0

def plot_pairs(data, pairs, output_prefix, pairs_per_plot=4):
    """Create bar plots for size pairs."""
    
    num_plots = (len(pairs) + pairs_per_plot - 1) // pairs_per_plot
    
    for plot_idx in range(num_plots):
        start_idx = plot_idx * pairs_per_plot
        end_idx = min(start_idx + pairs_per_plot, len(pairs))
        plot_pairs = pairs[start_idx:end_idx]
        
        fig, axes = plt.subplots(1, len(plot_pairs), figsize=(5 * len(plot_pairs), 6))
        
        if len(plot_pairs) == 1:
            axes = [axes]
        
        for ax_idx, pair in enumerate(plot_pairs):
            ax = axes[ax_idx]
            
            # Prepare data for this pair
            methods = ['Native FP64', 'Ozaki-II 12', 'Ozaki-II 16']
            method_patterns = ['Native', '12 spl', '16 spl']
            colors = ['#2E86AB', '#A23B72', '#F18F01']
            
            x = np.arange(len(methods))
            width = 0.35
            
            if len(pair) == 2:
                # Two sizes to compare
                size1, size2 = pair
                
                vals1 = [get_performance(data, size1, p) for p in method_patterns]
                vals2 = [get_performance(data, size2, p) for p in method_patterns]
                
                bars1 = ax.bar(x - width/2, vals1, width, label=f'{size1}', color=colors, edgecolor='black', linewidth=1)
                bars2 = ax.bar(x + width/2, vals2, width, label=f'{size2}', color=colors, edgecolor='black', linewidth=1, alpha=0.6, hatch='//')
                
                ax.set_title(f'Size: {size1} vs {size2}', fontsize=12, fontweight='bold')
                
                # Add value labels
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.annotate(f'{height:.1f}',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=8)
                
                # Create legend for sizes
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='gray', edgecolor='black', label=f'{size1} (aligned)'),
                    Patch(facecolor='gray', edgecolor='black', alpha=0.6, hatch='//', label=f'{size2} (misaligned)')
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
                
            else:
                # Single size
                size1 = pair[0]
                vals1 = [get_performance(data, size1, p) for p in method_patterns]
                
                bars1 = ax.bar(x, vals1, width * 1.5, color=colors, edgecolor='black', linewidth=1)
                ax.set_title(f'Size: {size1}', fontsize=12, fontweight='bold')
                
                for bar in bars1:
                    height = bar.get_height()
                    ax.annotate(f'{height:.1f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)
            
            ax.set_ylabel('Performance (TFLOPS)', fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, fontsize=9)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('GEMM Performance: Aligned vs Misaligned Matrix Sizes', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if num_plots > 1:
            output_file = f"{output_prefix}_part{plot_idx + 1}.png"
        else:
            output_file = f"{output_prefix}.png"
        
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_awkward.py <results_file>")
        print("Example: python plot_awkward.py awkward_matrices_cleanup.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    
    # Parse data
    print(f"Parsing: {input_file}")
    data = parse_results_file(input_file)
    
    if not data:
        print("Error: No data found in file")
        sys.exit(1)
    
    print(f"Found {len(data)} data rows")
    
    # Group by size pairs
    pairs = group_by_size_pairs(data)
    print(f"Found {len(pairs)} size pairs/groups")
    
    # Generate output filename
    base_name = os.path.splitext(input_file)[0]
    
    # Create plots (4 pairs per plot to avoid overcrowding)
    plot_pairs(data, pairs, base_name, pairs_per_plot=4)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
