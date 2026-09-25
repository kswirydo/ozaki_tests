# NVIDIA (CUDA / GEMMul8) benchmarks

CUDA ports of the rocEMU benchmarks used for the NVIDIA results in the paper
(B200 and GB300). The **table benchmarks** are the primary tools — they emit the
Native FP64 vs Ozaki-II (INT8/FP8) performance and workspace tables.

## Dependencies

- CUDA Toolkit (`nvcc`) — tested with CUDA 13.x.
- **GEMMul8** (external): clone and build from
  [RIKEN-RCCS/GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8), then point
  `GEMMUL8_PATH` at it.

## Compiling

`CUDA_ARCH` is auto-detected via `nvidia-smi`, but set it explicitly per device:

| GPU    | Host arch        | Build command                                             |
|--------|------------------|----------------------------------------------------------|
| B200   | x86_64           | `make tables CUDA_ARCH=sm_100 GEMMUL8_PATH=/path/GEMMul8` |
| GB300  | Grace (aarch64)  | `make tables CUDA_ARCH=sm_103 GEMMUL8_PATH=/path/GEMMul8` |

Notes:
- On Grace (aarch64) hosts an extra linker flag (`--stub-group-size`) is added
  automatically.
- `make tables` builds only the table benchmarks; `make` builds everything;
  `make clean` removes binaries.

## Table benchmarks

| Binary                       | Purpose                                                          |
|------------------------------|------------------------------------------------------------------|
| `table_benchmark`            | Native FP64 vs **INT8** Ozaki-II (12 & 16 moduli) + workspace     |
| `table_benchmark_fp8`        | Native FP64 vs **FP8** Ozaki-II (10 & 12 moduli) + workspace      |
| `table_benchmark_native`     | Native FP64 GEMM only                                            |
| `table_benchmark_cond_sweep` | Accuracy vs condition number sweep                              |
| `ab_table_latex_benchmark`   | A != B variant; emits LaTeX table rows (one per size)           |

Each benchmark generates conditioned matrices internally (condition number
`1e8` by default) and reports TFLOP/s, workspace size, and relative error,
averaged over 50 runs after 10 warmup iterations.

### Running

```bash
# Point the loader at your GEMMul8 build, then pass one or more matrix sizes.
export LD_LIBRARY_PATH=/path/to/GEMMul8/lib:$LD_LIBRARY_PATH

./table_benchmark 4096 8192 16384
./table_benchmark_fp8 4096 8192 16384
./ab_table_latex_benchmark 4096 8192 16384
```

## Other tools

Matrix generators (`matrix_generator`, `ab_matrix_generator`,
`aspect_ratio_generator`), standalone/single GEMM benchmarks, and the
inner/outer-dimension sweeps (Fig. 7 corner-case study) are also built by
`make`. See the top-level `README.md` for the algorithm and matrix-generation
details, which are shared with the ROCm versions.
