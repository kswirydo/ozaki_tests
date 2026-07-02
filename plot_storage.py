import matplotlib.pyplot as plt
import numpy as np

# Data
sizes = [384, 512, 1024, 2048, 4096, 8192, 16384, 24576, 32768, 40960]
a_mb = [1.1, 2.0, 8.0, 32.0, 128.0, 512.0, 2048.0, 4608.0, 8192.0, 12800.0]
abc_mb = [3*x for x in a_mb]
ws12_mb = [40.1, 41.8, 71.0, 188.0, 656.0, 2560.0, 10240.1, 23040.1, 40960.1, 64000.2]
ws16_mb = [42.6, 44.8, 83.0, 236.0, 848.0, 3328.0, 13312.1, 29952.1, 53248.1, 83200.2]

abc_plus_ws12 = [abc + ws for abc, ws in zip(abc_mb, ws12_mb)]
abc_plus_ws16 = [abc + ws for abc, ws in zip(abc_mb, ws16_mb)]

# Split data: up to 4096 (indices 0-4) and bigger than 4096 (indices 5-9)
split_idx = 5  # 4096 is at index 4, so 5+ is > 4096

sizes_small = sizes[:split_idx]
sizes_large = sizes[split_idx:]

abc_small = abc_mb[:split_idx]
abc_large = abc_mb[split_idx:]

ws12_plus_abc_small = abc_plus_ws12[:split_idx]
ws12_plus_abc_large = abc_plus_ws12[split_idx:]

ws16_plus_abc_small = abc_plus_ws16[:split_idx]
ws16_plus_abc_large = abc_plus_ws16[split_idx:]

width = 0.35

# Create 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Ozaki-12, small sizes (top-left)
ax = axes[0, 0]
x = np.arange(len(sizes_small))
ax.bar(x - width/2, abc_small, width, label='A+B+C', color='steelblue')
ax.bar(x + width/2, ws12_plus_abc_small, width, label='A+B+C + Ozaki-12 WS', color='darkorange')
ax.set_xlabel('Matrix Size')
ax.set_ylabel('Storage (MB)')
ax.set_title('Ozaki-12: Sizes ≤ 4096')
ax.set_xticks(x)
ax.set_xticklabels(sizes_small, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Ozaki-12, large sizes (top-right)
ax = axes[0, 1]
x = np.arange(len(sizes_large))
ax.bar(x - width/2, abc_large, width, label='A+B+C', color='steelblue')
ax.bar(x + width/2, ws12_plus_abc_large, width, label='A+B+C + Ozaki-12 WS', color='darkorange')
ax.set_xlabel('Matrix Size')
ax.set_ylabel('Storage (MB)')
ax.set_title('Ozaki-12: Sizes > 4096')
ax.set_xticks(x)
ax.set_xticklabels(sizes_large, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Ozaki-16, small sizes (bottom-left)
ax = axes[1, 0]
x = np.arange(len(sizes_small))
ax.bar(x - width/2, abc_small, width, label='A+B+C', color='steelblue')
ax.bar(x + width/2, ws16_plus_abc_small, width, label='A+B+C + Ozaki-16 WS', color='darkgreen')
ax.set_xlabel('Matrix Size')
ax.set_ylabel('Storage (MB)')
ax.set_title('Ozaki-16: Sizes ≤ 4096')
ax.set_xticks(x)
ax.set_xticklabels(sizes_small, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Ozaki-16, large sizes (bottom-right)
ax = axes[1, 1]
x = np.arange(len(sizes_large))
ax.bar(x - width/2, abc_large, width, label='A+B+C', color='steelblue')
ax.bar(x + width/2, ws16_plus_abc_large, width, label='A+B+C + Ozaki-16 WS', color='darkgreen')
ax.set_xlabel('Matrix Size')
ax.set_ylabel('Storage (MB)')
ax.set_title('Ozaki-16: Sizes > 4096')
ax.set_xticks(x)
ax.set_xticklabels(sizes_large, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('storage_comparison.png', dpi=150)
plt.show()
print("Saved to storage_comparison.png")
