# rocEMU — Benchmarking Ozaki-Scheme GEMM Emulation

`rocEMU` is a test and benchmarking harness for **Ozaki-scheme matrix-multiplication
emulation libraries**. The Ozaki scheme reconstructs a high-accuracy (FP64-equivalent)
matrix product `C = A × B` from a sum of low-precision integer/float tensor-core
products. This repository measures how well those emulated kernels trade **accuracy**,
**performance**, and **power** against a native FP64 GEMM, across a range of matrix
condition numbers and on both AMD and NVIDIA GPUs.

The primary library under test is [**GEMMul8**](https://github.com/RIKEN-RCCS/GEMMul8),
which provides two Ozaki-II backends:

| Backend | GEMMul8 entry point | Tensor-core dtype | Requires |
|---------|---------------------|-------------------|----------|
| **INT8 Ozaki-II** | `gemmul8::gemm`   | INT8 | rocBLAS / hipBLAS |
| **FP8 Ozaki-II**  | `gemmul8::gemmLt` | FP8  | hipBLASLt |

Each backend is parameterized by the **number of moduli** (typically 2–16): more moduli
means higher accuracy at the cost of throughput. Every emulated result is compared against
a **native FP64** reference (`hipblasDgemm` / `cublasDgemm`).

---

## What gets measured

For each method the benchmarks record:

- **Accuracy** — Frobenius-norm difference `‖C_native − C_emulated‖_F`, plus max/avg relative error
- **Performance** — execution time and effective TFLOPS
- **Power / energy** — via the power-instrumented benchmark variants and profiling scripts
- **Scaling** — behavior vs. matrix size, inner dimension (K), outer dimension (N), and aspect ratio

Matrices are generated with **prescribed condition numbers** (κ = 10² … 10¹⁷) so accuracy
can be studied as a function of conditioning.

---

## Repository layout

### Matrix generators
| Program | Source | Purpose |
|---------|--------|---------|
| `matrix_generator`        | `matrix_generator.cpp`        | Condition-number sweep for `C = A × A` benchmarks |
| `ab_matrix_generator`     | `ab_matrix_generator.cpp`     | Condition-number sweep of matrix **pairs** for `C = A × B` |
| `simple_matrix_generator` | `simple_matrix_generator.cpp` | Random `A`, `B` matrices (no prescribed conditioning) |
| `aspect_ratio_generator`  | `aspect_ratio_generator.cu`   | Non-square matrices for aspect-ratio studies |

Generation uses `A = U · S · Vᵀ` with singular values `σᵢ = 10^(i·k/(n−1))`, giving
`cond(A) = 10^k` exactly. Random seed is fixed (`12345`) for reproducibility, and all
matrices use column-major (BLAS/LAPACK) storage.

### Benchmarks
| Program | Source | Description |
|---------|--------|-------------|
| `gemm_benchmark`            | `gemm_benchmark.cu`            | `A × A`: native FP64 vs INT8 Ozaki-II across moduli |
| `ab_gemm_benchmark`         | `ab_gemm_benchmark.cu`         | `A × B` variant of the above |
| `ab_gemm_emu_compare`       | `ab_gemm_emu_compare.cu`       | 5-way compare: FP64, INT8 (12 & 16 moduli), FP8 (10 & 12 moduli); saves all result matrices |
| `table_benchmark`           | `table_benchmark.cu`           | Generates a matrix and emits a summary table row |
| `table_benchmark_fp8`       | `table_benchmark_fp8.cu`       | Table benchmark including the FP8 backend |
| `table_benchmark_power`     | `table_benchmark_power.cu`     | Table benchmark with power-measurement markers |
| `table_benchmark_power_fp8` | `table_benchmark_power_fp8.cu` | FP8 table benchmark with power markers |
| `inner_dim_benchmark`       | `inner_dim_benchmark.cu`       | Sweeps the contraction dimension K |
| `outer_dim_benchmark`       | `outer_dim_benchmark.cu`       | Sweeps the outer dimension N |
| `single_gemm_benchmark`     | `single_gemm_benchmark.cu`     | Single arbitrary `M × N × K` GEMM; generates a κ=10⁸ `A × B` and compares FP64 vs INT8 (12 & 16 moduli) |
| `gemm_vs_gemmlt_benchmark`  | `gemm_vs_gemmlt_benchmark.cu`  | INT8 `gemmul8::gemm` (hipBLAS) vs `gemmul8::gemmLt` (hipBLASLt) — perf, accuracy, and workspace side by side |
| `gemm_int8_vs_fp8_benchmark`| `gemm_int8_vs_fp8_benchmark.cu`| INT8 vs FP8 backends, both via `gemmLt` on one hipBLASLt handle — perf, accuracy, and workspace |
| `simple_gemm_benchmark`     | `simple_gemm_benchmark.cu`     | `A × B` on plain random matrices (from `simple_matrix_generator`): FP64 vs INT8 Ozaki-II |
| `standalone_benchmark`      | `standalone_benchmark.cu`      | Self-contained `A(M×K) × B(K×N)`: FP64 vs INT8 Ozaki-II (12 moduli only) |

### Analysis & plotting (Python)
- `compare_frobenius.py` — pairwise Frobenius comparison between two result sets
- `compare_frobenius_3way.py` — three-way comparison across GB300 / GB200 / MI355X result sets
- `analyze_gemm_power.py`, `analyze_gemm_power_fp8.py` — power/energy analysis
- `plot_benchmark.py`, `plot_scaling.py`, `plot_inner_dim.py`, `plot_outer_dim.py`, `plot_storage.py`, `plot_awkward.py` — result plots

### Power / profiling scripts
- `run_power_benchmark.sh`, `run_power_benchmark_fp8.sh` — drive the power-instrumented benchmarks
- `collect_profile.sh` — collect profiler traces

### Reference implementations (MATLAB)
- `emu_exp.m`, `emu_exp_fixed.m`, `test_ozaki2_comparison.m` — Ozaki-II reference/prototyping

### NVIDIA port (`nvidia/`)
CUDA/cuBLAS equivalents of the generators and benchmarks for NVIDIA GPUs (build with `nvidia/Makefile`).
Files mirror their ROCm counterparts unless noted.

| Source | Description |
|--------|-------------|
| `matrix_generator.cu`          | Condition-number matrix generator (cuBLAS + cuSOLVER + cuRAND); same output as `../matrix_generator.cpp` |
| `ab_matrix_generator.cu`       | `A × B` condition-number matrix-pair generator; same output as `../ab_matrix_generator.cpp` |
| `aspect_ratio_generator.cu`    | Aspect-ratio (dynamic-range) matrix generator |
| `gemm_benchmark.cu`            | `A × A`: native cuBLAS FP64 vs INT8 Ozaki-II |
| `ab_gemm_benchmark.cu`         | `A × B`: native cuBLAS FP64 vs INT8 Ozaki-II |
| `ab_gemm_emu_compare.cu`       | 5-way FP64 / INT8(12,16) / FP8(10,12) comparison, saves result matrices |
| `simple_gemm_benchmark.cu`     | `A × B` on plain random matrices: FP64 vs INT8 Ozaki-II |
| `single_gemm_benchmark.cu`     | Single arbitrary `M × N × K` GEMM comparison |
| `standalone_benchmark.cu`      | Self-contained `A × B`: FP64 vs INT8 Ozaki-II (12 moduli) |
| `inner_dim_benchmark.cu`       | Sweeps the contraction dimension K |
| `outer_dim_benchmark.cu`       | Sweeps the outer dimension N |
| `table_benchmark.cu`           | Table row: FP64 vs INT8 Ozaki-II (12 & 16 moduli) |
| `table_benchmark_fp8.cu`       | Table row including the FP8 backend |
| `table_benchmark_native.cu`    | Native cuBLAS-only baseline (no emulation), κ≈10⁸ `A × A` |
| `table_benchmark_cond_sweep.cu`| Native cuBLAS baseline swept across condition numbers (log₁₀κ = 1…16) |

### Data (tracked via Git LFS)
| Folder | Contents |
|--------|----------|
| `A_B_matrices/`   | 4096×4096 `A_*.txt` / `B_*.txt` input matrices |
| `c_results/`      | Result matrices from **MI355X** (AMD) |
| `c_results_b200/` | Result matrices from **GB200** (NVIDIA) |
| `c_results_b300/` | Result matrices from **GB300** (NVIDIA) |

> These directories are large and stored with **Git LFS**. Run `git lfs pull` after cloning
> to fetch the actual contents.

---

## Requirements

**AMD (default build):**
- AMD GPU with ROCm support, ROCm 5.0+
- rocBLAS, rocSOLVER, hipRAND, hipBLAS, and **hipBLASLt** (for the FP8 backend)
- [GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8) built and installed
- Enough GPU memory for your matrix size (≈ 40·n² bytes to generate, ≈ 32·n² bytes + workspace to benchmark; the 54272² default needs ~90 GB)

**NVIDIA port (`nvidia/`):**
- CUDA toolkit with cuBLAS / cuBLASLt, and a GEMMul8 build for CUDA

**Python tooling:** Python 3 with `numpy` and `matplotlib`.

**Git LFS:** required to fetch/track the matrix data folders.

---

## Building

The `GEMMUL8_PATH` and `ROCM_PATH` variables can be overridden on the command line.

```bash
# Build the default set (generators + core benchmarks)
make

# Point at a custom GEMMul8 / ROCm install
make GEMMUL8_PATH=/path/to/GEMMul8 ROCM_PATH=/opt/rocm

# Individual targets (examples)
make generator                 # matrix_generator
make ab_emu_compare            # 5-way FP64/INT8/FP8 comparison (needs hipBLASLt)
make table_benchmark_fp8       # FP8 table benchmark
make gemm_int8_vs_fp8_benchmark

# Debug build / cleanup
make debug
make clean
```

FP8 targets (`ab_emu_compare`, `table_benchmark_fp8`, `table_benchmark_power_fp8`,
`gemm_vs_gemmlt_benchmark`, `gemm_int8_vs_fp8_benchmark`) link against **hipBLASLt**.

---

## Usage

### 1. Generate matrices

```bash
# A×A condition-number sweep (default 54272², or specify dimensions + folder)
./matrix_generator 4096 4096 my_matrices

# A×B condition-number sweep (matrix pairs)
./ab_matrix_generator 4096 4096 A_B_matrices
```

Output for a κ-sweep is a set of files `..._cond1e2.txt … _cond1e17.txt`.

### 2. Run a benchmark

Benchmarks need GEMMul8 (and ROCm) on the library path. The `make run_*` targets set this
for you:

```bash
# A×A: native FP64 vs INT8 Ozaki-II
make run_benchmark MATRIX_FOLDER=my_matrices

# A×B
make run_ab_benchmark MATRIX_FOLDER=A_B_matrices

# 5-way FP64 / INT8(12,16) / FP8(10,12) comparison, saving result matrices
./ab_gemm_emu_compare A_B_matrices c_results

# Or run any binary manually with the library path exported
LD_LIBRARY_PATH=/path/to/GEMMul8/lib:/opt/rocm/lib:$LD_LIBRARY_PATH \
    ./gemm_benchmark my_matrices
```

### 3. Power benchmarking

```bash
./run_power_benchmark.sh          # INT8 backend
./run_power_benchmark_fp8.sh      # FP8 backend
python3 analyze_gemm_power.py     # analyze the resulting power_results_*.json
```

### 4. Compare / plot results

```bash
# Two-way Frobenius comparison (defaults: c_results vs c_results_b300)
python3 compare_frobenius.py

# Three-way comparison across GB300 / GB200 / MI355X
python3 compare_frobenius_3way.py

# Plots
python3 plot_benchmark.py my_matrices
```

### Quick smoke test

```bash
make test        # generates 1024² matrices and benchmarks them
```

---

## Output files

- **Generators:** `*_cond1eX.txt` matrices (κ = 10² … 10¹⁷), column-major, comma-separated.
- **`ab_gemm_emu_compare`:** per-method result matrices `C_<method>_cond1eX.txt` (e.g.
  `C_native_fp64_*`, `C_int8_ozaki12_*`, `C_int8_ozaki16_*`, `C_fp8_ozaki10_*`,
  `C_fp8_ozaki12_*`) plus a CSV of Frobenius differences vs. the FP64 reference.
- **Benchmarks:** `*_benchmark_<device>_<timestamp>.csv` with method, moduli, mode,
  time (ms), TFLOPS, and relative errors.
- **Power runs:** `power_results_*.json` and generated energy/scaling plots (`*.png`).

---

## Notes

- Native FP64 (`hipblasDgemm` / `cublasDgemm`) is the accuracy reference for all methods.
- Accuracy generally improves with more Ozaki moduli; performance decreases correspondingly.
- The `c_results*` folders hold device-specific outputs (MI355X, GB200, GB300) used by the
  Frobenius comparison scripts to cross-check emulation results across vendors.
- Paths such as `GEMMUL8_PATH` default to a local install location — override them to match
  your environment.
