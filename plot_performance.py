import matplotlib.pyplot as plt
import numpy as np

# Common matrix sizes (skip rows that have only B200 data)
matrix_sizes = [384, 512, 1024, 2048, 4096, 8192, 16384]

# B200 data
b200_native = [10.82, 18.79, 30.76, 35.06, 35.23, 35.82, 35.99]
b200_ozaki_12 = [0.27, 0.61, 4.38, 24.10, 74.38, 132.21, 166.44]
b200_ozaki_16 = [0.20, 0.47, 3.35, 18.37, 56.08, 98.67, 123.95]

# B300 data
b300_native = [0.51, 0.92, 1.06, 1.1, 1.1, 1.1, 1.1]
b300_ozaki_12 = [0.43, 0.97, 4.12, 7.37, 9.83, 11.01, 11.58]
b300_ozaki_16 = [0.37, 0.83, 3.27, 5.77, 7.59, 8.45, 8.87]

data = {
    'sizes': matrix_sizes,
    'b200_native': b200_native, 'b300_native': b300_native,
    'b200_ozaki_12': b200_ozaki_12, 'b300_ozaki_12': b300_ozaki_12,
    'b200_ozaki_16': b200_ozaki_16, 'b300_ozaki_16': b300_ozaki_16,
}

# Split: sizes up to 1024 and starting at 2048
split_idx = matrix_sizes.index(2048)

width = 0.12
G = 0.32  # distance between pair centers

colors = {
    'b200_native': '#1B4965', 'b300_native': '#62B6CB',
    'b200_ozaki_12': '#6A1B4D', 'b300_ozaki_12': '#C84B8C',
    'b200_ozaki_16': '#1B5E20', 'b300_ozaki_16': '#81C784',
}


def plot_group(ax, sl):
    sizes = data['sizes'][sl]
    x = np.arange(len(sizes))
    pairs = [
        ('b200_native', 'b300_native', 'Native GEMM'),
        ('b200_ozaki_12', 'b300_ozaki_12', 'Ozaki-II (12)'),
        ('b200_ozaki_16', 'b300_ozaki_16', 'Ozaki-II (16)'),
    ]
    centers = [-G, 0.0, G]
    for (b200_key, b300_key, _), c in zip(pairs, centers):
        ax.bar(x + c - width / 2, data[b200_key][sl], width,
               label=f'B200 {b200_key.split("_", 1)[1]}', color=colors[b200_key])
        ax.bar(x + c + width / 2, data[b300_key][sl], width,
               label=f'B300 {b300_key.split("_", 1)[1]}', color=colors[b300_key])
    ax.set_xlabel('Matrix Size', fontsize=12)
    ax.set_ylabel('Performance (TFlops)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.grid(axis='y', alpha=0.3)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

plot_group(ax1, slice(0, split_idx))
ax1.set_title('Matrix Size up to 1024', fontsize=13)

plot_group(ax2, slice(split_idx, None))
ax2.set_title('Matrix Size from 2048', fontsize=13)

handles = [
    plt.Rectangle((0, 0), 1, 1, color=colors['b200_native']),
    plt.Rectangle((0, 0), 1, 1, color=colors['b300_native']),
    plt.Rectangle((0, 0), 1, 1, color=colors['b200_ozaki_12']),
    plt.Rectangle((0, 0), 1, 1, color=colors['b300_ozaki_12']),
    plt.Rectangle((0, 0), 1, 1, color=colors['b200_ozaki_16']),
    plt.Rectangle((0, 0), 1, 1, color=colors['b300_ozaki_16']),
]
labels = ['B200 Native GEMM', 'B300 Native GEMM',
          'B200 Ozaki-II (12)', 'B300 Ozaki-II (12)',
          'B200 Ozaki-II (16)', 'B300 Ozaki-II (16)']
fig.legend(handles, labels, loc='upper center', ncol=6, fontsize=10)

fig.suptitle('B200 vs B300 Performance: Native GEMM vs Ozaki-II (INT8)', fontsize=15, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('performance_comparison_B200vsB300_int8.jpg', dpi=300, format='jpg', bbox_inches='tight')
print("Plot saved as performance_comparison_B200vsB300_int8.jpg")
