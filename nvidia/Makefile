# ============================================================================
# NVIDIA (CUDA / GEMMul8) build for rocEMU
#
# The PRIMARY targets are the *table benchmarks* used for the paper's
# performance/workspace tables (Native FP64 vs Ozaki-II emulated GEMM):
#     table_benchmark            INT8 Ozaki-II (12 & 16 moduli)   -> Table IV/V
#     table_benchmark_fp8        FP8  Ozaki-II (10 & 12 moduli)   -> Table IV/V
#     table_benchmark_native     native FP64 only
#     table_benchmark_cond_sweep accuracy vs condition number
#     ab_table_latex_benchmark   A != B table, emits LaTeX rows
#
# ----------------------------------------------------------------------------
# Compiling for a specific GPU
# ----------------------------------------------------------------------------
# CUDA_ARCH is auto-detected from nvidia-smi. Override it for the target device:
#
#     NVIDIA B200   (x86_64 host)   :  make CUDA_ARCH=sm_100
#     NVIDIA GB300  (Grace/aarch64) :  make CUDA_ARCH=sm_103
#
# On Grace (aarch64) hosts an extra linker flag is added automatically.
#
# Point GEMMUL8_PATH at your GEMMul8 checkout
# (external dependency: https://github.com/RIKEN-RCCS/GEMMul8):
#
#     make GEMMUL8_PATH=/path/to/GEMMul8 CUDA_ARCH=sm_103
#
# Common invocations:
#     make tables CUDA_ARCH=sm_100         # B200: build only the table benchmarks
#     make tables CUDA_ARCH=sm_103         # GB300: build only the table benchmarks
#     make        CUDA_ARCH=sm_100         # build everything
# ============================================================================

GEMMUL8_PATH ?= /mnt/local/home/kswirydo/GEMMul8

# Auto-detect compute capability (e.g. "sm_100"); fall back to sm_100 (B200).
CUDA_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.' | sed 's/^/sm_/')
ifeq ($(strip $(CUDA_ARCH)),)
CUDA_ARCH := sm_100
endif

NVCC       = nvcc
CXXFLAGS   = -arch=$(CUDA_ARCH) -O3 -std=c++17
CXXFLAGS20 = -arch=$(CUDA_ARCH) -O3 -std=c++20 -Xcompiler -Wall

INCLUDES = -I$(GEMMUL8_PATH)/include
LDFLAGS  = -L$(GEMMUL8_PATH)/lib -Xlinker -rpath -Xlinker $(GEMMUL8_PATH)/lib
LDFLAGS_GENERATOR =

# Grace (aarch64) hosts (e.g. GB300) need a larger stub group size at link time.
ifeq ($(shell uname -m),aarch64)
LDFLAGS           += -Xlinker --stub-group-size=33554432
LDFLAGS_GENERATOR += -Xlinker --stub-group-size=33554432
endif

LIBS_GEMMUL8 = -lgemmul8 -lcublas -lcudart
LIBS_FULL    = $(LIBS_GEMMUL8) -lcusolver -lcurand
LIBS_FP8     = $(LIBS_FULL) -lcublasLt
# ab_* tools also query the driver / NVML for device info.
LIBS_AB      = -lgemmul8 -lcublas -lcublasLt -lcudart -lcuda -lnvidia-ml -ldl -lcusolver -lcurand

TABLE_TARGETS = \
	table_benchmark \
	table_benchmark_native \
	table_benchmark_cond_sweep \
	table_benchmark_fp8 \
	ab_table_latex_benchmark

OTHER_TARGETS = \
	matrix_generator \
	ab_matrix_generator \
	aspect_ratio_generator \
	gemm_benchmark \
	ab_gemm_benchmark \
	ab_gemm_emu_compare \
	simple_gemm_benchmark \
	inner_dim_benchmark \
	outer_dim_benchmark \
	single_gemm_benchmark \
	standalone_benchmark

TARGETS = $(TABLE_TARGETS) $(OTHER_TARGETS)

.PHONY: all tables clean
all: $(TARGETS)
tables: $(TABLE_TARGETS)

# --- table benchmarks (paper tables) ---
table_benchmark: table_benchmark.cu cuda_conditioned_matrix.cuh gemmul8_helpers.cuh
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS) $(LIBS_FULL)

table_benchmark_fp8: table_benchmark_fp8.cu cuda_conditioned_matrix.cuh
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS) $(LIBS_FP8)

table_benchmark_native table_benchmark_cond_sweep: %: %.cu cuda_conditioned_matrix.cuh
	$(NVCC) $(CXXFLAGS) $< -o $@ -lcublas -lcusolver -lcurand -lcudart $(LDFLAGS_GENERATOR)

ab_table_latex_benchmark: ab_table_latex_benchmark.cu cuda_helpers.cuh cuda_conditioned_matrix.cuh
	$(NVCC) $(CXXFLAGS20) $(INCLUDES) $< -o $@ $(LDFLAGS) $(LIBS_AB)

# --- matrix generators ---
matrix_generator ab_matrix_generator: %: %.cu cuda_helpers.cuh
	$(NVCC) $(CXXFLAGS) $< -o $@ -lcublas -lcusolver -lcurand -lcudart $(LDFLAGS_GENERATOR)

aspect_ratio_generator: aspect_ratio_generator.cu cuda_helpers.cuh
	$(NVCC) $(CXXFLAGS) $< -o $@ -lcurand -lcudart $(LDFLAGS_GENERATOR)

# --- other GEMM benchmarks ---
gemm_benchmark ab_gemm_benchmark simple_gemm_benchmark: %: %.cu cuda_helpers.cuh
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS) $(LIBS_GEMMUL8)

ab_gemm_emu_compare: ab_gemm_emu_compare.cu cuda_helpers.cuh
	$(NVCC) $(CXXFLAGS20) $(INCLUDES) $< -o $@ $(LDFLAGS) $(LIBS_AB)

inner_dim_benchmark outer_dim_benchmark single_gemm_benchmark standalone_benchmark: %: %.cu cuda_helpers.cuh cuda_conditioned_matrix.cuh
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS) $(LIBS_FULL)

clean:
	rm -f $(TARGETS)
