# ROCm Matrix Generator and GEMM Benchmark

This project contains two programs:
1. **Matrix Generator**: Creates dense matrices with prescribed condition numbers
2. **GEMM Benchmark**: Compares rocBLAS DGEMM vs Ozaki-II (GEMMul8) performance

## Programs

### 1. Matrix Generator (`matrix_generator`)

Generates dense matrices with a prescribed condition number using AMD GPUs (ROCm).

#### Algorithm

For each condition number κ = 10^k, generates matrix A = U × S × V' where:
- **U** (m×n): Orthogonal matrix from QR decomposition of random Gaussian matrix
- **V** (n×n): Orthogonal matrix from QR decomposition of random Gaussian matrix  
- **S** (n×n): Diagonal matrix with singular values σᵢ = 10^(i·k/(n-1)) for i = 0, 1, ..., n-1

This ensures cond(A) = σₘₐₓ/σₘᵢₙ = 10^k / 10^0 = 10^k.

### 2. GEMM Benchmark (`gemm_benchmark`)

Reads matrices from files and benchmarks:
- **rocBLAS DGEMM**: Standard double-precision GEMM using hipblasDgemm
- **Ozaki-II GEMM**: High-accuracy GEMM using INT8 tensor cores via GEMMul8 library

Computes C = A × A (matrix multiplied with itself) and reports:
- Execution time
- Performance (TFLOPS)
- Relative error compared to rocBLAS result

## Requirements

- AMD GPU with ROCm support
- ROCm 5.0+ installed (includes rocBLAS, rocSOLVER, hipRAND, hipBLAS)
- GEMMul8 library — **external dependency**. Clone and build it from
  [RIKEN-RCCS/GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8), then point the
  include/library paths in the build commands below at your local checkout
  (the paths in this README, e.g. `/home/kswirydo/GEMMul8/GEMMul8`, are examples
  and should be replaced with your own).
- Enough GPU memory for the chosen matrix size (the default 4096×4096 needs
  ~1 GB; very large sizes such as 54272×54272 need 100+ GB)

## Building

```bash
# Build both programs
make

# Build only matrix generator
make generator

# Build only benchmark
make benchmark
```

### Manual compilation

```bash
# Matrix generator
hipcc -O3 -std=c++17 matrix_generator.cpp -o matrix_generator \
      -lrocblas -lrocsolver -lhiprand \
      -I/opt/rocm/include -L/opt/rocm/lib

# GEMM benchmark
hipcc -O3 -std=c++17 gemm_benchmark.cu -o gemm_benchmark \
      -I/home/kswirydo/GEMMul8/GEMMul8/include \
      -L/home/kswirydo/GEMMul8/GEMMul8/lib \
      -lgemmul8 -lhipblas -lamdhip64 \
      -I/opt/rocm/include -L/opt/rocm/lib
```

## Usage

### Generate matrices

```bash
# Default: 4096 x 4096 matrices, auto-generated folder name
./matrix_generator

# Custom dimensions, auto-generated folder name
./matrix_generator 8192 8192

# Custom dimensions with specified output folder
./matrix_generator 4096 4096 my_matrices

# Show help
./matrix_generator --help
```

Output folder defaults to `matrices_MxN_YYYYMMDD_HHMMSS` format if not specified (e.g., `matrices_4096x4096_20260128_143022`).

### Run benchmark

```bash
# Run benchmark on matrices in current directory
make run_benchmark

# Run benchmark on matrices in specific folder
make run_benchmark MATRIX_FOLDER=matrices_4096x4096_20260128_143022

# Or manually with library path
LD_LIBRARY_PATH=/home/kswirydo/GEMMul8/GEMMul8/lib:$LD_LIBRARY_PATH ./gemm_benchmark path/to/matrices

# Show help
./gemm_benchmark --help
```

### Quick test

```bash
# Generate small test matrices (1024x1024) and benchmark them
make test
```

This creates a `test_matrices_1024x1024` folder with matrices and runs the benchmark on them.

## Output Files

### Matrix Generator
- `M_cond_1e2.txt` through `M_cond_1e17.txt` - matrices with condition numbers 10² to 10¹⁷

### GEMM Benchmark
- `gemm_benchmark_<device>_<timestamp>.csv` - benchmark results including:
  - Method (rocBLAS or Ozaki-II)
  - Number of moduli (for Ozaki-II)
  - Fast/accurate mode
  - Time (ms)
  - Performance (TFLOPS)
  - Relative errors (Frobenius, max, average)

## Ozaki-II Parameters

The benchmark tests multiple configurations:
- **Number of moduli**: 2-16 (more moduli = higher accuracy, lower performance)
- **Mode**: 
  - `fast`: Faster but slightly less accurate
  - `accurate`: More accurate but slower

## Memory Requirements

For an n×n matrix:
- Matrix generator: ~5n² doubles ≈ 40n² bytes
- GEMM benchmark: ~4n² doubles ≈ 32n² bytes + GEMMul8 workspace

For the default 4096×4096: approximately **0.7 GB** GPU memory for generator, **0.5 GB** for benchmark.
For very large sizes such as 54272×54272: approximately **93 GB** for generator, **70+ GB** for benchmark.

## Notes

- All matrices use column-major storage (BLAS/LAPACK convention)
- Double precision (64-bit floating point) throughout
- Random seed is fixed (12345) for reproducibility
- Benchmark uses rocBLAS result as reference for error computation
