#!/usr/bin/env python3
"""
Analyze power profile data for GEMM phases only.

Parses GEMM_PHASE_START/END timestamps from benchmark log and extracts
corresponding power samples to compute average power and total energy
for each GEMM phase (native_gemm, ozaki12, ozaki16).

Usage:
    python analyze_gemm_power.py --power <power_profile.txt> --log <benchmark_log.txt> --output <results.json>
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_power_profile(filepath):
    """
    Parse power profiler output file.
    
    Returns:
        dict with keys:
            - timestamps_us: array of Unix timestamps in microseconds
            - power: array of power values (Watts) per device, shape (n_samples, n_devices)
            - power_caps: list of power caps per device
    """
    timestamps_us = []
    power_values = []
    power_caps = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('#'):
                if 'power_cap' in line:
                    match = re.search(r'power_cap\s+(\d+)', line)
                    if match:
                        power_caps.append(int(match.group(1)))
                continue
            
            parts = line.split()
            if len(parts) < 3:
                continue
            
            date_str = parts[0]
            time_str = parts[1]
            power_readings = [float(p) for p in parts[2:] if p]
            
            dt_str = f"{date_str} {time_str}"
            try:
                dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                try:
                    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue
            
            ts_us = int(dt.timestamp() * 1e6)
            timestamps_us.append(ts_us)
            power_values.append(power_readings)
    
    return {
        'timestamps_us': np.array(timestamps_us),
        'power': np.array(power_values),
        'power_caps': power_caps
    }


def parse_gemm_phases(filepath):
    """
    Parse GEMM phase markers from benchmark log.
    
    Markers format:
        GEMM_PHASE_START <phase_name> <size> <unix_timestamp_microseconds>
        GEMM_PHASE_END <phase_name> <size> <unix_timestamp_microseconds>
    
    Returns:
        list of dicts with keys: phase, size, start_us, end_us
    """
    phases = {}
    
    pattern_start = re.compile(r'GEMM_PHASE_START\s+(\w+)\s+(\d+)\s+(\d+)')
    pattern_end = re.compile(r'GEMM_PHASE_END\s+(\w+)\s+(\d+)\s+(\d+)')
    
    with open(filepath, 'r') as f:
        for line in f:
            match = pattern_start.search(line)
            if match:
                phase_name = match.group(1)
                size = int(match.group(2))
                ts_us = int(match.group(3))
                key = (phase_name, size)
                phases[key] = {'phase': phase_name, 'size': size, 'start_us': ts_us}
                continue
            
            match = pattern_end.search(line)
            if match:
                phase_name = match.group(1)
                size = int(match.group(2))
                ts_us = int(match.group(3))
                key = (phase_name, size)
                if key in phases:
                    phases[key]['end_us'] = ts_us
    
    result = []
    for key in sorted(phases.keys(), key=lambda x: phases[x].get('start_us', 0)):
        phase_data = phases[key]
        if 'start_us' in phase_data and 'end_us' in phase_data:
            result.append(phase_data)
    
    return result


def compute_power_energy(power_data, phase, device_idx=0):
    """
    Compute average power and total energy for a GEMM phase.
    
    Args:
        power_data: dict from parse_power_profile()
        phase: dict with start_us, end_us
        device_idx: which GPU device to analyze
    
    Returns:
        dict with avg_power_w, energy_j, duration_s, n_samples
    """
    timestamps = power_data['timestamps_us']
    power = power_data['power']
    
    if power.ndim == 1:
        power = power.reshape(-1, 1)
    
    # Use actual phase duration from benchmark timestamps
    actual_duration_s = (phase['end_us'] - phase['start_us']) / 1e6
    
    mask = (timestamps >= phase['start_us']) & (timestamps <= phase['end_us'])
    indices = np.where(mask)[0]
    
    # If no samples in range, try to interpolate from nearby samples
    if len(indices) == 0:
        # Find nearest samples before and after
        before = np.where(timestamps < phase['start_us'])[0]
        after = np.where(timestamps > phase['end_us'])[0]
        
        if len(before) > 0 and len(after) > 0:
            power_before = power[before[-1], device_idx] if device_idx < power.shape[1] else power[before[-1], 0]
            power_after = power[after[0], device_idx] if device_idx < power.shape[1] else power[after[0], 0]
            avg_power = (power_before + power_after) / 2
            energy = avg_power * actual_duration_s
            return {
                'avg_power_w': float(avg_power),
                'energy_j': float(energy),
                'duration_s': float(actual_duration_s),
                'n_samples': 0,
                'interpolated': True
            }
        return {
            'avg_power_w': 0.0,
            'energy_j': 0.0,
            'duration_s': float(actual_duration_s),
            'n_samples': 0,
            'warning': 'no_samples'
        }
    
    phase_power = power[indices, device_idx] if device_idx < power.shape[1] else power[indices, 0]
    avg_power = np.mean(phase_power)
    
    if len(indices) == 1:
        # Single sample: use avg power * actual duration
        energy = avg_power * actual_duration_s
        return {
            'avg_power_w': float(avg_power),
            'energy_j': float(energy),
            'duration_s': float(actual_duration_s),
            'n_samples': 1
        }
    
    # Multiple samples: compute energy from trapezoid integration
    phase_timestamps = timestamps[indices]
    dt = np.diff(phase_timestamps) / 1e6
    energy = np.sum((phase_power[:-1] + phase_power[1:]) / 2 * dt)
    
    # Add energy for time before first sample and after last sample
    time_before = (phase_timestamps[0] - phase['start_us']) / 1e6
    time_after = (phase['end_us'] - phase_timestamps[-1]) / 1e6
    energy += phase_power[0] * time_before + phase_power[-1] * time_after
    
    return {
        'avg_power_w': float(avg_power),
        'energy_j': float(energy),
        'duration_s': float(actual_duration_s),
        'n_samples': len(indices)
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze GEMM power consumption')
    parser.add_argument('--power', required=True, help='Power profile file')
    parser.add_argument('--log', required=True, help='Benchmark log file with phase markers')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--device', type=int, default=0, help='GPU device index (default: 0)')
    args = parser.parse_args()
    
    if not Path(args.power).exists():
        print(f"Error: Power profile not found: {args.power}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.log).exists():
        print(f"Error: Benchmark log not found: {args.log}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading power profile: {args.power}")
    power_data = parse_power_profile(args.power)
    print(f"  Samples: {len(power_data['timestamps_us'])}")
    print(f"  Devices: {power_data['power'].shape[1] if power_data['power'].ndim > 1 else 1}")
    print(f"  Power caps: {power_data['power_caps']} W")
    
    print(f"\nParsing GEMM phases: {args.log}")
    phases = parse_gemm_phases(args.log)
    print(f"  Found {len(phases)} GEMM phases")
    
    results = {
        'power_profile': args.power,
        'benchmark_log': args.log,
        'device_index': args.device,
        'power_caps_w': power_data['power_caps'],
        'phases': []
    }
    
    total_energy = 0.0
    total_duration = 0.0
    
    print("\n" + "=" * 110)
    print(f"{'Phase':<15} {'Size':>8} {'Duration (s)':>14} {'Avg Power (W)':>15} {'Energy (J)':>15} {'Samples':>10}")
    print("=" * 110)
    
    for phase in phases:
        stats = compute_power_energy(power_data, phase, args.device)
        
        phase_result = {
            'phase': phase['phase'],
            'size': phase['size'],
            'start_us': phase['start_us'],
            'end_us': phase['end_us'],
            **stats
        }
        results['phases'].append(phase_result)
        
        total_energy += stats['energy_j']
        total_duration += stats['duration_s']
        
        warning = ' (!)' if 'warning' in stats else ''
        print(f"{phase['phase']:<15} {phase['size']:>8} {stats['duration_s']:>14.6f} "
              f"{stats['avg_power_w']:>15.2f} {stats['energy_j']:>15.6f} {stats['n_samples']:>10}{warning}")
    
    print("=" * 110)
    
    avg_power_overall = total_energy / total_duration if total_duration > 0 else 0
    print(f"{'TOTAL':<15} {'':<8} {total_duration:>14.6f} {avg_power_overall:>15.2f} {total_energy:>15.6f}")
    print("=" * 110)
    
    by_phase_type = {}
    for phase_result in results['phases']:
        ptype = phase_result['phase']
        if ptype not in by_phase_type:
            by_phase_type[ptype] = {'energy_j': 0, 'duration_s': 0, 'sizes': []}
        by_phase_type[ptype]['energy_j'] += phase_result['energy_j']
        by_phase_type[ptype]['duration_s'] += phase_result['duration_s']
        by_phase_type[ptype]['sizes'].append(phase_result['size'])
    
    for ptype in by_phase_type:
        dur = by_phase_type[ptype]['duration_s']
        by_phase_type[ptype]['avg_power_w'] = by_phase_type[ptype]['energy_j'] / dur if dur > 0 else 0
    
    results['by_phase_type'] = by_phase_type
    
    # Group results by matrix size
    by_size = {}
    for phase_result in results['phases']:
        size = phase_result['size']
        ptype = phase_result['phase']
        if size not in by_size:
            by_size[size] = {}
        by_size[size][ptype] = phase_result
    
    results['by_size'] = {}
    
    # Note: Each method runs 50 iterations, so comparing native vs ozaki12 or ozaki16 is fair
    print("\n")
    print("=" * 160)
    print("                                              ENERGY USE PER MATRIX SIZE (50 iterations each)")
    print("=" * 160)
    print(f"{'Size':>8} | {'Native GEMM':<30} | {'Ozaki-II (12 mod)':<30} | {'Ozaki-II (16 mod)':<30} | {'Ratio 12/Nat':>12} | {'Ratio 16/Nat':>12}")
    print(f"{'':>8} | {'Power(W)':>10} {'Energy(J)':>12} {'Time(s)':>8} | {'Power(W)':>10} {'Energy(J)':>12} {'Time(s)':>8} | {'Power(W)':>10} {'Energy(J)':>12} {'Time(s)':>8} | {'':>12} | {'':>12}")
    print("-" * 160)
    
    total_native_energy = 0.0
    total_oz12_energy = 0.0
    total_oz16_energy = 0.0
    
    for size in sorted(by_size.keys()):
        size_data = by_size[size]
        
        # Native
        native = size_data.get('native_gemm', {})
        native_energy = native.get('energy_j', 0)
        native_power = native.get('avg_power_w', 0)
        native_duration = native.get('duration_s', 0)
        
        # Ozaki12
        oz12 = size_data.get('ozaki12', {})
        oz12_energy = oz12.get('energy_j', 0)
        oz12_power = oz12.get('avg_power_w', 0)
        oz12_duration = oz12.get('duration_s', 0)
        
        # Ozaki16
        oz16 = size_data.get('ozaki16', {})
        oz16_energy = oz16.get('energy_j', 0)
        oz16_power = oz16.get('avg_power_w', 0)
        oz16_duration = oz16.get('duration_s', 0)
        
        ratio_12 = oz12_energy / native_energy if native_energy > 0 else 0
        ratio_16 = oz16_energy / native_energy if native_energy > 0 else 0
        
        total_native_energy += native_energy
        total_oz12_energy += oz12_energy
        total_oz16_energy += oz16_energy
        
        results['by_size'][size] = {
            'native': {'energy_j': native_energy, 'avg_power_w': native_power, 'duration_s': native_duration},
            'ozaki12': {'energy_j': oz12_energy, 'avg_power_w': oz12_power, 'duration_s': oz12_duration},
            'ozaki16': {'energy_j': oz16_energy, 'avg_power_w': oz16_power, 'duration_s': oz16_duration},
            'ratio_oz12_native': ratio_12,
            'ratio_oz16_native': ratio_16
        }
        
        print(f"{size:>8} | {native_power:>10.2f} {native_energy:>12.6f} {native_duration:>8.4f} | "
              f"{oz12_power:>10.2f} {oz12_energy:>12.6f} {oz12_duration:>8.4f} | "
              f"{oz16_power:>10.2f} {oz16_energy:>12.6f} {oz16_duration:>8.4f} | "
              f"{ratio_12:>12.4f}x | {ratio_16:>12.4f}x")
    
    print("-" * 160)
    
    # Totals
    total_ratio_12 = total_oz12_energy / total_native_energy if total_native_energy > 0 else 0
    total_ratio_16 = total_oz16_energy / total_native_energy if total_native_energy > 0 else 0
    print(f"{'TOTAL':>8} | {'':>10} {total_native_energy:>12.6f} {'':>8} | "
          f"{'':>10} {total_oz12_energy:>12.6f} {'':>8} | "
          f"{'':>10} {total_oz16_energy:>12.6f} {'':>8} | "
          f"{total_ratio_12:>12.4f}x | {total_ratio_16:>12.4f}x")
    print("=" * 160)
    
    results['summary'] = {
        'total_native_energy_j': total_native_energy,
        'total_oz12_energy_j': total_oz12_energy,
        'total_oz16_energy_j': total_oz16_energy,
        'ratio_oz12_native': total_ratio_12,
        'ratio_oz16_native': total_ratio_16
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")
    
    # Generate bar chart
    chart_file = args.output.replace('.json', '_energy_chart.png')
    generate_energy_bar_chart(results['by_size'], chart_file)


def generate_energy_bar_chart(by_size_data, output_file):
    """Generate bar chart comparing energy use: Native vs Ozaki12 vs Ozaki16."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping chart generation")
        return
    
    sizes = sorted(by_size_data.keys())
    native_energy = [by_size_data[s]['native']['energy_j'] for s in sizes]
    oz12_energy = [by_size_data[s]['ozaki12']['energy_j'] for s in sizes]
    oz16_energy = [by_size_data[s]['ozaki16']['energy_j'] for s in sizes]
    
    x = np.arange(len(sizes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, native_energy, width, label='Native FP64', color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x, oz12_energy, width, label='Ozaki-II (12 mod)', color='#3498db', edgecolor='black')
    bars3 = ax.bar(x + width, oz16_energy, width, label='Ozaki-II (16 mod)', color='#e74c3c', edgecolor='black')
    
    ax.set_xlabel('Matrix Size', fontsize=12)
    ax.set_ylabel('Energy (Joules) for 50 iterations', fontsize=12)
    ax.set_title('GEMM Energy Consumption: Native vs Emulated', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend(loc='upper left')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3, which='both')
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Energy bar chart saved to: {output_file}")


if __name__ == '__main__':
    main()
