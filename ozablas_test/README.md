# OzaBLAS Table Benchmark (Ozaki Scheme I & II)

Standalone benchmark used for the **Ozaki-I accuracy results** in the paper
(Fig. 3, `../figs/ozaki1_accuracy.jpg`). It generates ill-conditioned matrices
and measures performance and accuracy of Ozaki Scheme I and II against native
FP64 DGEMM. In the paper this was run on **AMD MI355X only** (HIP/ROCm).

> This benchmark is the AMD/OzaBLAS counterpart to the GEMMul8-based benchmarks
> in the repository root (Ozaki-II). It lives in a separate folder because it
> depends on a different external library (ozablas).

## External dependency: ozablas

ozablas is **not** included in this repo. Clone and build it first
([IlyaNyrkov/ozablas](https://github.com/IlyaNyrkov/ozablas); the paper used
commit `043719e`):

```bash
git clone https://github.com/IlyaNyrkov/ozablas
cd ozablas
make release-hip      # AMD GPUs (MI355X)  -> builds libozablas.so
# or: make release-cuda
```

Then point this project at that checkout via `OZABLAS_ROOT` (defaults to
`/home/kswirydo/ozablas`).

Required system libraries:
- **HIP/ROCm**: hipblas, rocblas, hiprand, rocsolver
- **CUDA**: cublas, curand, cusolver

## Building

### Option A — Makefile (HIP / MI355X)

```bash
make OZABLAS_ROOT=/path/to/ozablas
```

### Option B — CMake (HIP or CUDA)

```bash
mkdir build && cd build
cmake .. -DOZABLAS_ENABLE_HIP=ON -DOZABLAS_ROOT=/path/to/ozablas   # AMD
# or: cmake .. -DOZABLAS_ENABLE_CUDA=ON -DOZABLAS_ROOT=/path/to/ozablas
make
```

## Usage

```
./table_benchmark [options] [size1] [size2] ...

Options:
  --scheme1, -1    Run Ozaki Scheme I only
  --scheme2, -2    Run Ozaki Scheme II only
  --both,    -b    Run both schemes (default)
  --help,    -h    Show help
```

Make sure the ozablas library is on the loader path when running:

```bash
export LD_LIBRARY_PATH=/path/to/ozablas/build_hip_release/src:$LD_LIBRARY_PATH

# Both schemes, default sizes (1024 2048 4096 8192)
./table_benchmark

# Custom sizes
./table_benchmark 4096 8192 16384

# Only Scheme I (paper's Ozaki-I data; useful for debugging large matrices)
./table_benchmark -1 4096
```

## Output

- **stderr**: verbose progress (useful for debugging hangs)
- **stdout**: summary table (matrix size, native GEMM TFLOP/s, and per-scheme
  performance + relative Frobenius error for 12 and 16 slices)

## Configuration

Edit `table_benchmark.cpp` to change the benchmark parameters:

```cpp
static const int NUM_WARMUP = 2;       // Warmup iterations
static const int NUM_ITERATIONS = 50;  // Timed iterations
static const int LOG10_COND = 8;       // Condition number 10^8
```
