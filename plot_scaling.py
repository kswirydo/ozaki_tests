#!/usr/bin/env python3
"""
Plot GEMMul8 profiling results as stacked bar charts.
Reads percentage tables from scaling_results.txt and generates visualizations.
"""

import re
import sys
import matplotlib.pyplot as plt
import numpy as np

def parse_percentage_tables(filename):
    """Parse TABLE 4 and TABLE 5 (percentage tables) from the results file."""
    
    with open(filename, 'r') as f:
        content = f.read()
    
    data = {'12': [], '16': []}
    
    # Pattern to match data rows: | SIZE | Scale | GEMM | Recon |
    row_pattern = re.compile(r'\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|')
    
    # Find TABLE 4 (12 Moduli Percentage)
    table4_match = re.search(r'TABLE 4: 12 Moduli.*?Percentage.*?\n(.*?)(?=\n\nTABLE|\n\nLegend|\Z)', 
                             content, re.DOTALL)
    if table4_match:
        for match in row_pattern.finditer(table4_match.group(1)):
            size = int(match.group(1))
            scale = float(match.group(2))
            gemm = float(match.group(3))
            recon = float(match.group(4))
            data['12'].append({'size': size, 'scale': scale, 'gemm': gemm, 'recon': recon})
    
    # Find TABLE 5 (16 Moduli Percentage)
    table5_match = re.search(r'TABLE 5: 16 Moduli.*?Percentage.*?\n(.*?)(?=\n\nLegend|\n\n=|\Z)', 
                             content, re.DOTALL)
    if table5_match:
        for match in row_pattern.finditer(table5_match.group(1)):
            size = int(match.group(1))
            scale = float(match.group(2))
            gemm = float(match.group(3))
            recon = float(match.group(4))
            data['16'].append({'size': size, 'scale': scale, 'gemm': gemm, 'recon': recon})
    
    return data


def create_stacked_bar_plot(data, output_file='scaling_breakdown.png'):
    """Create a stacked bar chart showing execution time breakdown."""
    
    # Set up the figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('GEMMul8 Execution Time Breakdown by Phase', fontsize=14, fontweight='bold')
    
    # Color scheme
    colors = {
        'scale': '#E74C3C',   # Red for scaling
        'gemm': '#3498DB',    # Blue for GEMM
        'recon': '#2ECC71'    # Green for reconstruction
    }
    
    for idx, (moduli, ax) in enumerate([('12', axes[0]), ('16', axes[1])]):
        if not data[moduli]:
            continue
            
        sizes = [d['size'] for d in data[moduli]]
        scale_pct = [d['scale'] for d in data[moduli]]
        gemm_pct = [d['gemm'] for d in data[moduli]]
        recon_pct = [d['recon'] for d in data[moduli]]
        
        x = np.arange(len(sizes))
        width = 0.7
        
        # Create stacked bars
        bars1 = ax.bar(x, scale_pct, width, label='Scaling', color=colors['scale'], edgecolor='white', linewidth=0.5)
        bars2 = ax.bar(x, gemm_pct, width, bottom=scale_pct, label='INT8 GEMM', color=colors['gemm'], edgecolor='white', linewidth=0.5)
        bars3 = ax.bar(x, recon_pct, width, bottom=np.array(scale_pct) + np.array(gemm_pct), 
                       label='Reconstruction', color=colors['recon'], edgecolor='white', linewidth=0.5)
        
        # Formatting
        ax.set_xlabel('Matrix Size (N x N)', fontsize=11)
        ax.set_ylabel('Percentage of Execution Time (%)', fontsize=11)
        ax.set_title(f'{moduli} Moduli', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in sizes], rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 105)
        ax.set_yticks(range(0, 101, 10))
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=9)
        
        # Add percentage labels on bars (only for larger segments)
        for i, (s, g, r) in enumerate(zip(scale_pct, gemm_pct, recon_pct)):
            # GEMM label (always show, it's the largest)
            ax.text(i, s + g/2, f'{g:.0f}%', ha='center', va='center', 
                   fontsize=8, color='white', fontweight='bold')
            # Scale label (show if > 15%)
            if s > 15:
                ax.text(i, s/2, f'{s:.0f}%', ha='center', va='center', 
                       fontsize=8, color='white', fontweight='bold')
            # Recon label (show if > 10%)
            if r > 10:
                ax.text(i, s + g + r/2, f'{r:.0f}%', ha='center', va='center', 
                       fontsize=8, color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


def create_comparison_plot(data, output_file='scaling_comparison.png'):
    """Create a grouped bar chart comparing 12 vs 16 moduli."""
    
    if not data['12'] or not data['16']:
        print("Error: Missing data for comparison plot")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('GEMMul8 Phase Breakdown: 12 vs 16 Moduli Comparison', fontsize=14, fontweight='bold')
    
    sizes = [d['size'] for d in data['12']]
    x = np.arange(len(sizes))
    width = 0.35
    
    phases = [
        ('scale', 'Scaling Phase', '#E74C3C', '#C0392B'),
        ('gemm', 'INT8 GEMM Phase', '#3498DB', '#2980B9'),
        ('recon', 'Reconstruction Phase', '#2ECC71', '#27AE60')
    ]
    
    for ax, (phase, title, color12, color16) in zip(axes, phases):
        vals_12 = [d[phase] for d in data['12']]
        vals_16 = [d[phase] for d in data['16']]
        
        bars1 = ax.bar(x - width/2, vals_12, width, label='12 Moduli', color=color12, edgecolor='white')
        bars2 = ax.bar(x + width/2, vals_16, width, label='16 Moduli', color=color16, edgecolor='white')
        
        ax.set_xlabel('Matrix Size', fontsize=10)
        ax.set_ylabel('Percentage (%)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in sizes], rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Set appropriate y-axis limits
        max_val = max(max(vals_12), max(vals_16))
        ax.set_ylim(0, max_val * 1.15)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


def create_line_plot(data, output_file='scaling_trends.png'):
    """Create line plots showing trends across matrix sizes."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sizes = [d['size'] for d in data['12']]
    
    # Line styles and colors
    styles = {
        '12': {'scale': ('--', '#E74C3C', 'o'), 'gemm': ('--', '#3498DB', 's'), 'recon': ('--', '#2ECC71', '^')},
        '16': {'scale': ('-', '#C0392B', 'o'), 'gemm': ('-', '#2980B9', 's'), 'recon': ('-', '#27AE60', '^')}
    }
    
    labels = {'scale': 'Scaling', 'gemm': 'INT8 GEMM', 'recon': 'Reconstruction'}
    
    for moduli in ['12', '16']:
        for phase in ['scale', 'gemm', 'recon']:
            vals = [d[phase] for d in data[moduli]]
            linestyle, color, marker = styles[moduli][phase]
            ax.plot(sizes, vals, linestyle=linestyle, color=color, marker=marker, 
                   markersize=6, linewidth=2, alpha=0.8,
                   label=f'{labels[phase]} ({moduli} mod)')
    
    ax.set_xlabel('Matrix Size (N x N)', fontsize=12)
    ax.set_ylabel('Percentage of Execution Time (%)', fontsize=12)
    ax.set_title('GEMMul8 Phase Breakdown vs Matrix Size', fontsize=14, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes], rotation=45, ha='right')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'scaling_results.txt'
    
    print(f"Reading data from: {input_file}")
    data = parse_percentage_tables(input_file)
    
    if not data['12'] and not data['16']:
        print("Error: No data found in the file!")
        return 1
    
    print(f"Found {len(data['12'])} entries for 12 moduli")
    print(f"Found {len(data['16'])} entries for 16 moduli")
    
    # Generate all plots
    create_stacked_bar_plot(data, 'scaling_breakdown.png')
    create_comparison_plot(data, 'scaling_comparison.png')
    create_line_plot(data, 'scaling_trends.png')
    
    print("\nAll plots generated successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
