import matplotlib.pyplot as plt
import numpy as np

# Common matrix sizes (skip rows that have only B200 data)
matrix_sizes = [384, 512, 1024, 2048, 4096, 8192, 16384]

# B200 FP8 data
b200_native = [11.51, 21.76, 31.46, 35.03, 35.29, 35.86, 36.03]
b200_ozaki_10 = [0.47, 1.13, 6.95, 26.26, 50.84, 67.34, 75.77]
b200_ozaki_12 = [0.42, 1.01, 6.14, 22.66, 42.58, 56.37, 63.35]

# B300 FP8 data
b300_native = [0.51, 0.92, 1.06, 1.1, 1.1, 1.1, 1.1]
b300_ozaki_10 = [0.45, 1.04, 5.39, 17.01, 35.25, 58.4, 73.52]
b300_ozaki_12 = [0.42, 0.93, 4.81, 14.97, 30.7, 50.25, 63.07]

data = {
    'sizes': matrix_sizes,
    'b200_native': b200_native, 'b300_native': b300_native,
    'b200_ozaki_10': b200_ozaki_10, 'b300_ozaki_10': b300_ozaki_10,
    'b200_ozaki_12': b200_ozaki_12, 'b300_ozaki_12': b300_ozaki_12,
}

# Split: sizes up to 1024 and starting at 2048
split_idx = matrix_sizes.index(2048)

width = 0.12
G = 0.32  # distance between pair centers

colors = {
    'b200_native': '#1B4965', 'b300_native': '#62B6CB',
    'b200_ozaki_10': '#6A1B4D', 'b300_ozaki_10': '#C84B8C',
    'b200_ozaki_12': '#1B5E20', 'b300_ozaki_12': '#81C784',
}


def plot_group(ax, sl):
    sizes = data['sizes'][sl]
    x = np.arange(len(sizes))
    pairs = [
        ('b200_native', 'b300_native'),
        ('b200_ozaki_10', 'b300_ozaki_10'),
        ('b200_ozaki_12', 'b300_ozaki_12'),
    ]
    centers = [-G, 0.0, G]
    for (b200_key, b300_key), c in zip(pairs, centers):
        ax.bar(x + c - width / 2, data[b200_key][sl], width, color=colors[b200_key])
        ax.bar(x + c + width / 2, data[b300_key][sl], width, color=colors[b300_key])
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
    plt.Rectangle((0, 0), 1, 1, color=colors['b200_ozaki_10']),
    plt.Rectangle((0, 0), 1, 1, color=colors['b300_ozaki_10']),
    plt.Rectangle((0, 0), 1, 1, color=colors['b200_ozaki_12']),
    plt.Rectangle((0, 0), 1, 1, color=colors['b300_ozaki_12']),
]
labels = ['B200 Native GEMM', 'B300 Native GEMM',
          'B200 Ozaki-II (10 splits)', 'B300 Ozaki-II (10 splits)',
          'B200 Ozaki-II (12 splits)', 'B300 Ozaki-II (12 splits)']
fig.legend(handles, labels, loc='upper center', ncol=6, fontsize=10)

fig.suptitle('B200 vs B300 Performance: Native GEMM vs Ozaki-II (FP8)', fontsize=15, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('performance_comparison_B200vsB300_fp8.jpg', dpi=300, format='jpg', bbox_inches='tight')
print("Plot saved as performance_comparison_B200vsB300_fp8.jpg")
