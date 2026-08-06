# Makefile for ROCm Matrix Generator and GEMM Benchmark

# ROCm installation path (adjust if different)
ROCM_PATH ?= /opt/rocm

# GEMMul8 path
GEMMUL8_PATH ?= /home/kswirydo/GEMMul8

# Compiler
HIPCC = $(ROCM_PATH)/bin/hipcc

# Compiler flags
CXXFLAGS = -O3 -std=c++17 -Wall
CXXFLAGS += -Wno-unused-result -Wno-unused-command-line-argument

# Includes and library paths
INCLUDES = -I$(ROCM_PATH)/include
LDFLAGS = -L$(ROCM_PATH)/lib

# Libraries for matrix generator (with condition numbers)
LIBS_GENERATOR = -lrocblas -lrocsolver -lhiprand

# Libraries for simple matrix generator (random only)
LIBS_SIMPLE_GENERATOR = -lhiprand

# Libraries for benchmark (includes GEMMul8)
INCLUDES_BENCHMARK = $(INCLUDES) -I$(GEMMUL8_PATH)/include
LDFLAGS_BENCHMARK = $(LDFLAGS) -L$(GEMMUL8_PATH)/lib -Wl,-rpath,$(GEMMUL8_PATH)/lib
LIBS_BENCHMARK = -lgemmul8 -lhipblas -lamdhip64

# Targets
TARGET_GENERATOR = matrix_generator
TARGET_AB_GENERATOR = ab_matrix_generator
TARGET_SIMPLE_GENERATOR = simple_matrix_generator
TARGET_BENCHMARK = gemm_benchmark
TARGET_AB_BENCHMARK = ab_gemm_benchmark
TARGET_AB_EMU_COMPARE = ab_gemm_emu_compare
TARGET_SIMPLE_BENCHMARK = simple_gemm_benchmark
TARGET_TABLE_BENCHMARK = table_benchmark
TARGET_INNER_DIM_BENCHMARK = inner_dim_benchmark
TARGET_OUTER_DIM_BENCHMARK = outer_dim_benchmark
TARGET_SINGLE_GEMM_BENCHMARK = single_gemm_benchmark
TARGET_STANDALONE_BENCHMARK = standalone_benchmark
TARGET_ASPECT_RATIO_GENERATOR = aspect_ratio_generator
TARGET_TABLE_BENCHMARK_POWER = table_benchmark_power
TARGET_TABLE_BENCHMARK_FP8 = table_benchmark_fp8
TARGET_TABLE_BENCHMARK_POWER_FP8 = table_benchmark_power_fp8
TARGET_GEMM_VS_GEMMLT_BENCHMARK = gemm_vs_gemmlt_benchmark
TARGET_GEMM_INT8_VS_FP8_BENCHMARK = gemm_int8_vs_fp8_benchmark
TARGET_AB_TABLE_BENCHMARK = ab_table_benchmark
TARGET_AB_TABLE_BENCHMARK_FAST = ab_table_benchmark_fast
TARGET_AB_TABLE_SIZE_BENCHMARK = ab_table_size_benchmark
TARGET_SPECIAL_ZERO_GEMM = special_zero_gemm_gradeTest1
TARGET_SCALING_GEMM = scaling_gemm_gradeTest2
TARGET_SCALING_GEMM_FP8 = scaling_gemm_gradeTest3
TARGET_ASPECT_RATIO_TEST_ALT = aspect_ratio_test_alt

# Source files
SRCS_GENERATOR = matrix_generator.cpp
SRCS_AB_GENERATOR = ab_matrix_generator.cpp
SRCS_SIMPLE_GENERATOR = simple_matrix_generator.cpp
SRCS_BENCHMARK = gemm_benchmark.cu
SRCS_AB_BENCHMARK = ab_gemm_benchmark.cu
SRCS_AB_EMU_COMPARE = ab_gemm_emu_compare.cu
SRCS_SIMPLE_BENCHMARK = simple_gemm_benchmark.cu
SRCS_TABLE_BENCHMARK = table_benchmark.cu
SRCS_INNER_DIM_BENCHMARK = inner_dim_benchmark.cu
SRCS_OUTER_DIM_BENCHMARK = outer_dim_benchmark.cu
SRCS_SINGLE_GEMM_BENCHMARK = single_gemm_benchmark.cu
SRCS_STANDALONE_BENCHMARK = standalone_benchmark.cu
SRCS_ASPECT_RATIO_GENERATOR = aspect_ratio_generator.cu
SRCS_TABLE_BENCHMARK_POWER = table_benchmark_power.cu
SRCS_TABLE_BENCHMARK_FP8 = table_benchmark_fp8.cu
SRCS_TABLE_BENCHMARK_POWER_FP8 = table_benchmark_power_fp8.cu
SRCS_GEMM_VS_GEMMLT_BENCHMARK = gemm_vs_gemmlt_benchmark.cu
SRCS_GEMM_INT8_VS_FP8_BENCHMARK = gemm_int8_vs_fp8_benchmark.cu
SRCS_AB_TABLE_BENCHMARK = ab_table_benchmark.cu
SRCS_AB_TABLE_SIZE_BENCHMARK = ab_table_size_benchmark.cu
SRCS_SPECIAL_ZERO_GEMM = special_zero_gemm_gradeTest1.cu
SRCS_SCALING_GEMM = scaling_gemm_gradeTest2.cu
SRCS_SCALING_GEMM_FP8 = scaling_gemm_gradeTest3.cu
SRCS_ASPECT_RATIO_TEST_ALT = aspect_ratio_test_alt.cu

# Default target: build all
all: $(TARGET_GENERATOR) $(TARGET_AB_GENERATOR) $(TARGET_SIMPLE_GENERATOR) $(TARGET_BENCHMARK) $(TARGET_AB_BENCHMARK) $(TARGET_SIMPLE_BENCHMARK)

# Matrix generator (condition number matrices for A*A benchmarks)
$(TARGET_GENERATOR): $(SRCS_GENERATOR)
	$(HIPCC) $(CXXFLAGS) $(INCLUDES) $(SRCS_GENERATOR) -o $(TARGET_GENERATOR) $(LDFLAGS) $(LIBS_GENERATOR)

# A/B matrix generator (condition number matrices for A*B benchmarks)
$(TARGET_AB_GENERATOR): $(SRCS_AB_GENERATOR)
	$(HIPCC) $(CXXFLAGS) $(INCLUDES) $(SRCS_AB_GENERATOR) -o $(TARGET_AB_GENERATOR) $(LDFLAGS) $(LIBS_GENERATOR)

# Simple matrix generator (random A and B matrices, no condition numbers)
$(TARGET_SIMPLE_GENERATOR): $(SRCS_SIMPLE_GENERATOR)
	$(HIPCC) $(CXXFLAGS) $(INCLUDES) $(SRCS_SIMPLE_GENERATOR) -o $(TARGET_SIMPLE_GENERATOR) $(LDFLAGS) $(LIBS_SIMPLE_GENERATOR)

# GEMM benchmark (for A*A condition number matrices)
$(TARGET_BENCHMARK): $(SRCS_BENCHMARK)
	$(HIPCC) $(CXXFLAGS) $(INCLUDES_BENCHMARK) $(SRCS_BENCHMARK) -o $(TARGET_BENCHMARK) $(LDFLAGS_BENCHMARK) $(LIBS_BENCHMARK)

# A/B GEMM benchmark (for A*B matrices with condition numbers)
$(TARGET_AB_BENCHMARK): $(SRCS_AB_BENCHMARK)
	$(HIPCC) $(CXXFLAGS) $(INCLUDES_BENCHMARK) $(SRCS_AB_BENCHMARK) -o $(TARGET_AB_BENCHMARK) $(LDFLAGS_BENCHMARK) $(LIBS_BENCHMARK)

# A/B GEMM emulation comparison (native FP64 vs INT8/FP8 Ozaki-II, multiple moduli)
# Requires hipBLASLt for the FP8 backend.
$(TARGET_AB_EMU_COMPARE): $(SRCS_AB_EMU_COMPARE)
	$(HIPCC) $(CXXFLAGS) $(INCLUDES_BENCHMARK) $(SRCS_AB_EMU_COMPARE) -o $(TARGET_AB_EMU_COMPARE) $(LDFLAGS_BENCHMARK) $(LIBS_BENCHMARK) -lhipblaslt

# Simple GEMM benchmark (for simple A*B matrices without condition numbers)
$(TARGET_SIMPLE_BENCHMARK): $(SRCS_SIMPLE_BENCHMARK)
	$(HIPCC) $(CXXFLAGS) $(INCLUDES_BENCHMARK) $(SRCS_SIMPLE_BENCHMARK) -o $(TARGET_SIMPLE_BENCHMARK) $(LDFLAGS_BENCHMARK) $(LIBS_BENCHMARK)

# Table benchmark (generates matrix and outputs table row)
$(TARGET_TABLE_BENCHMARK): $(SRCS_TABLE_BENCHMARK)
	$(HIPCC) $(CXXFLAGS) $(INCLUDES_BENCHMARK) $(SRCS_TABLE_BENCHMARK) -o $(TARGET_TABLE_BENCHMARK) $(LDFLAGS_BENCHMARK) $(LIBS_BENCHMARK) $(LIBS_GENERATOR)

# Inner dimension benchmark (varies K from N down to 256)
$(TARGET_INNER_DIM_BENCHMARK): $(SRCS_INNER_DIM_BENCHMARK)
	$(HIPCC) $(CXXFLAGS) $(INCLUDES_BENCHMARK) $(SRCS_INNER_DIM_BENCHMARK) -o $(TARGET_INNER_DIM_BENCHMARK) $(LDFLAGS_BENCHMARK) $(LIBS_BENCHMARK) $(LIBS_GENERATOR)

# Outer dimension benchmark (varies N from K to 40*K)
$(TARGET_OUTER_DIM_BENCHMARK): $(SRCS_OUTER_DIM_BENCHMARK)
	$(HIPCC) $(CXXFLAGS) $(INCLUDES_BENCHMARK) $(SRCS_OUTER_DIM_BENCHMARK) -o $(TARGET_OUTER_DIM_BENCHMARK) $(LDFLAGS_BENCHMARK) $(LIBS_BENCHMARK) $(LIBS_GENERATOR)

# Single GEMM benchmark (arbitrary M, N, K)
$(TARGET_SINGLE_GEMM_BENCHMARK): $(SRCS_SINGLE_GEMM_BENCHMARK)
	$(HIPCC) $(CXXFLAGS) $(INCLUDES_BENCHMARK) $(SRCS_SINGLE_GEMM_BENCHMARK) -o $(TARGET_SINGLE_GEMM_BENCHMARK) $(LDFLAGS_BENCHMARK) $(LIBS_BENCHMARK) $(LIBS_GENERATOR)

# Standalone benchmark (12 moduli only)
$(TARGET_STANDALONE_BENCHMARK): $(SRCS_STANDALONE_BENCHMARK)
	$(HIPCC) $(CXXFLAGS) $(INCLUDES_BENCHMARK) $(SRCS_STANDALONE_BENCHMARK) -o $(TARGET_STANDALONE_BENCHMARK) $(LDFLAGS_BENCHMARK) $(LIBS_BENCHMARK) $(LIBS_GENERATOR)

# Aspect ratio matrix generator
$(TARGET_ASPECT_RATIO_GENERATOR): $(SRCS_ASPECT_RATIO_GENERATOR)
	$(HIPCC) $(CXXFLAGS) $(INCLUDES) $(SRCS_ASPECT_RATIO_GENERATOR) -o $(TARGET_ASPECT_RATIO_GENERATOR) $(LDFLAGS) -lhiprand

# Table benchmark with power measurement markers
$(TARGET_TABLE_BENCHMARK_POWER): $(SRCS_TABLE_BENCHMARK_POWER)
	$(HIPCC) $(CXXFLAGS) $(INCLUDES_BENCHMARK) $(SRCS_TABLE_BENCHMARK_POWER) -o $(TARGET_TABLE_BENCHMARK_POWER) $(LDFLAGS_BENCHMARK) $(LIBS_BENCHMARK) $(LIBS_GENERATOR)

# Table benchmark FP8 (without power measurement) - requires hipBLASLt
$(TARGET_TABLE_BENCHMARK_FP8): $(SRCS_TABLE_BENCHMARK_FP8)
	$(HIPCC) $(CXXFLAGS) $(INCLUDES_BENCHMARK) $(SRCS_TABLE_BENCHMARK_FP8) -o $(TARGET_TABLE_BENCHMARK_FP8) $(LDFLAGS_BENCHMARK) $(LIBS_BENCHMARK) $(LIBS_GENERATOR) -lhipblaslt

# Table benchmark FP8 with power measurement markers - requires hipBLASLt
$(TARGET_TABLE_BENCHMARK_POWER_FP8): $(SRCS_TABLE_BENCHMARK_POWER_FP8)
	$(HIPCC) $(CXXFLAGS) $(INCLUDES_BENCHMARK) $(SRCS_TABLE_BENCHMARK_POWER_FP8) -o $(TARGET_TABLE_BENCHMARK_POWER_FP8) $(LDFLAGS_BENCHMARK) $(LIBS_BENCHMARK) $(LIBS_GENERATOR) -lhipblaslt

# GEMM vs GEMMLt comparison (INT8) - requires hipBLASLt
$(TARGET_GEMM_VS_GEMMLT_BENCHMARK): $(SRCS_GEMM_VS_GEMMLT_BENCHMARK)
	$(HIPCC) $(CXXFLAGS) $(INCLUDES_BENCHMARK) $(SRCS_GEMM_VS_GEMMLT_BENCHMARK) -o $(TARGET_GEMM_VS_GEMMLT_BENCHMARK) $(LDFLAGS_BENCHMARK) $(LIBS_BENCHMARK) $(LIBS_GENERATOR) -lhipblaslt

# INT8 vs FP8 comparison (gemmLt) - requires hipBLASLt
$(TARGET_GEMM_INT8_VS_FP8_BENCHMARK): $(SRCS_GEMM_INT8_VS_FP8_BENCHMARK)
	$(HIPCC) $(CXXFLAGS) $(INCLUDES_BENCHMARK) $(SRCS_GEMM_INT8_VS_FP8_BENCHMARK) -o $(TARGET_GEMM_INT8_VS_FP8_BENCHMARK) $(LDFLAGS_BENCHMARK) $(LIBS_BENCHMARK) $(LIBS_GENERATOR) -lhipblaslt

# A/B condition-number table sweep (4096^3 GEMM, CSV + heatmap companion script)
$(TARGET_AB_TABLE_BENCHMARK): $(SRCS_AB_TABLE_BENCHMARK)
	$(HIPCC) $(CXXFLAGS) $(INCLUDES_BENCHMARK) $(SRCS_AB_TABLE_BENCHMARK) -o $(TARGET_AB_TABLE_BENCHMARK) $(LDFLAGS_BENCHMARK) $(LIBS_BENCHMARK) $(LIBS_GENERATOR) -lhipblaslt

# Size sweep of C = A*B (A != B, cond 10^3): native FP64 vs INT8 Ozaki-II 12/16
$(TARGET_AB_TABLE_SIZE_BENCHMARK): $(SRCS_AB_TABLE_SIZE_BENCHMARK)
	$(HIPCC) $(CXXFLAGS) $(INCLUDES_BENCHMARK) $(SRCS_AB_TABLE_SIZE_BENCHMARK) -o $(TARGET_AB_TABLE_SIZE_BENCHMARK) $(LDFLAGS_BENCHMARK) $(LIBS_BENCHMARK) $(LIBS_GENERATOR)

# Same sweep with Ozaki-II fastmode=true for all INT8/FP8 runs
$(TARGET_AB_TABLE_BENCHMARK_FAST): $(SRCS_AB_TABLE_BENCHMARK)
	$(HIPCC) $(CXXFLAGS) -DAB_TABLE_OZAKI_FAST_MODE=1 $(INCLUDES_BENCHMARK) $(SRCS_AB_TABLE_BENCHMARK) -o $(TARGET_AB_TABLE_BENCHMARK_FAST) $(LDFLAGS_BENCHMARK) $(LIBS_BENCHMARK) $(LIBS_GENERATOR) -lhipblaslt

# Build only generator
generator: $(TARGET_GENERATOR)

# Build only A/B generator
ab_generator: $(TARGET_AB_GENERATOR)

# Build only simple generator
simple_generator: $(TARGET_SIMPLE_GENERATOR)

# Build only benchmark
benchmark: $(TARGET_BENCHMARK)

# Build only A/B benchmark
ab_benchmark: $(TARGET_AB_BENCHMARK)

# Special-zero experiment: FP64 vs INT8 Ozaki-II with planted zero dot products
$(TARGET_SPECIAL_ZERO_GEMM): $(SRCS_SPECIAL_ZERO_GEMM)
	$(HIPCC) $(CXXFLAGS) $(INCLUDES_BENCHMARK) $(SRCS_SPECIAL_ZERO_GEMM) -o $(TARGET_SPECIAL_ZERO_GEMM) $(LDFLAGS_BENCHMARK) $(LIBS_BENCHMARK) -lhipblaslt

special_zero_gemm_gradeTest1: $(TARGET_SPECIAL_ZERO_GEMM)

# Build and run the special-zero experiment (usage: make run_special_zero_gemm_gradeTest1 [ARGS="1024 16 12345"])
ARGS ?=
run_special_zero_gemm_gradeTest1: $(TARGET_SPECIAL_ZERO_GEMM)
	LD_LIBRARY_PATH=$(GEMMUL8_PATH)/lib:$(ROCM_PATH)/lib:$$LD_LIBRARY_PATH ./$(TARGET_SPECIAL_ZERO_GEMM) $(ARGS)

# Diagonal-scaling robustness test: FP64 vs FP64-scaled vs INT8-scaled 3-way compare
$(TARGET_SCALING_GEMM): $(SRCS_SCALING_GEMM)
	$(HIPCC) $(CXXFLAGS) $(INCLUDES_BENCHMARK) $(SRCS_SCALING_GEMM) -o $(TARGET_SCALING_GEMM) $(LDFLAGS_BENCHMARK) $(LIBS_BENCHMARK) -lhipblaslt

scaling_gemm_gradeTest2: $(TARGET_SCALING_GEMM)

run_scaling_gemm_gradeTest2: $(TARGET_SCALING_GEMM)
	LD_LIBRARY_PATH=$(GEMMUL8_PATH)/lib:$(ROCM_PATH)/lib:$$LD_LIBRARY_PATH ./$(TARGET_SCALING_GEMM) $(ARGS)

# Same scaling test as gradeTest2 plus the FP8 Ozaki-II backend
$(TARGET_SCALING_GEMM_FP8): $(SRCS_SCALING_GEMM_FP8)
	$(HIPCC) $(CXXFLAGS) $(INCLUDES_BENCHMARK) $(SRCS_SCALING_GEMM_FP8) -o $(TARGET_SCALING_GEMM_FP8) $(LDFLAGS_BENCHMARK) $(LIBS_BENCHMARK) -lhipblaslt

scaling_gemm_gradeTest3: $(TARGET_SCALING_GEMM_FP8)

# Usage: make run_scaling_gemm_gradeTest3 [ARGS="1024 16 12345 53 12"]
run_scaling_gemm_gradeTest3: $(TARGET_SCALING_GEMM_FP8)
	LD_LIBRARY_PATH=$(GEMMUL8_PATH)/lib:$(ROCM_PATH)/lib:$$LD_LIBRARY_PATH ./$(TARGET_SCALING_GEMM_FP8) $(ARGS)

# Aspect-ratio test: dynamic range in every row of A and every column of B
$(TARGET_ASPECT_RATIO_TEST_ALT): $(SRCS_ASPECT_RATIO_TEST_ALT)
	$(HIPCC) $(CXXFLAGS) $(INCLUDES_BENCHMARK) $(SRCS_ASPECT_RATIO_TEST_ALT) -o $(TARGET_ASPECT_RATIO_TEST_ALT) $(LDFLAGS_BENCHMARK) $(LIBS_BENCHMARK) $(LIBS_GENERATOR)

aspect_ratio_test_alt: $(TARGET_ASPECT_RATIO_TEST_ALT)

# Usage: make run_aspect_ratio_test_alt [ARGS="4096 31 12345 out.csv"]
run_aspect_ratio_test_alt: $(TARGET_ASPECT_RATIO_TEST_ALT)
	LD_LIBRARY_PATH=$(GEMMUL8_PATH)/lib:$(ROCM_PATH)/lib:$$LD_LIBRARY_PATH ./$(TARGET_ASPECT_RATIO_TEST_ALT) $(ARGS)

# Build only A/B emulation comparison
ab_emu_compare: $(TARGET_AB_EMU_COMPARE)

# Build only simple benchmark
simple_benchmark: $(TARGET_SIMPLE_BENCHMARK)

# Build only table benchmark
table_benchmark: $(TARGET_TABLE_BENCHMARK)

# Build only inner dimension benchmark
inner_dim_benchmark: $(TARGET_INNER_DIM_BENCHMARK)

# Build only outer dimension benchmark
outer_dim_benchmark: $(TARGET_OUTER_DIM_BENCHMARK)

# Build only single GEMM benchmark
single_gemm_benchmark: $(TARGET_SINGLE_GEMM_BENCHMARK)

# Build table benchmark with power measurement markers
table_benchmark_power: $(TARGET_TABLE_BENCHMARK_POWER)

# Build table benchmark FP8
table_benchmark_fp8: $(TARGET_TABLE_BENCHMARK_FP8)

# Build table benchmark FP8 with power measurement markers
table_benchmark_power_fp8: $(TARGET_TABLE_BENCHMARK_POWER_FP8)

# Build gemm vs gemmLt comparison benchmark
gemm_vs_gemmlt_benchmark: $(TARGET_GEMM_VS_GEMMLT_BENCHMARK)

# Build INT8 vs FP8 comparison benchmark
gemm_int8_vs_fp8_benchmark: $(TARGET_GEMM_INT8_VS_FP8_BENCHMARK)

# Build A/B table condition sweep benchmark
ab_table_benchmark: $(TARGET_AB_TABLE_BENCHMARK)

ab_table_benchmark_fast: $(TARGET_AB_TABLE_BENCHMARK_FAST)

ab_table_size_benchmark: $(TARGET_AB_TABLE_SIZE_BENCHMARK)

# Usage: make run_ab_table_size_benchmark [ARGS="1024 2048 4096 8192"]
run_ab_table_size_benchmark: $(TARGET_AB_TABLE_SIZE_BENCHMARK)
	LD_LIBRARY_PATH=$(GEMMUL8_PATH)/lib:$(ROCM_PATH)/lib:$$LD_LIBRARY_PATH ./$(TARGET_AB_TABLE_SIZE_BENCHMARK) $(if $(ARGS),$(ARGS),1024 2048 4096 8192)

run_ab_table_benchmark: $(TARGET_AB_TABLE_BENCHMARK)
	LD_LIBRARY_PATH=$(GEMMUL8_PATH)/lib:$(ROCM_PATH)/lib:$$LD_LIBRARY_PATH ./$(TARGET_AB_TABLE_BENCHMARK)

run_ab_table_benchmark_fast: $(TARGET_AB_TABLE_BENCHMARK_FAST)
	LD_LIBRARY_PATH=$(GEMMUL8_PATH)/lib:$(ROCM_PATH)/lib:$$LD_LIBRARY_PATH ./$(TARGET_AB_TABLE_BENCHMARK_FAST)

run_ab_table_benchmark_plot: $(TARGET_AB_TABLE_BENCHMARK)
	LD_LIBRARY_PATH=$(GEMMUL8_PATH)/lib:$(ROCM_PATH)/lib:$$LD_LIBRARY_PATH ./$(TARGET_AB_TABLE_BENCHMARK)
	python3 plot_ab_table_heatmap.py ab_table_benchmark_accurate_*.csv --scale auto

run_ab_table_benchmark_fast_plot: $(TARGET_AB_TABLE_BENCHMARK_FAST)
	LD_LIBRARY_PATH=$(GEMMUL8_PATH)/lib:$(ROCM_PATH)/lib:$$LD_LIBRARY_PATH ./$(TARGET_AB_TABLE_BENCHMARK_FAST)
	python3 plot_ab_table_heatmap.py ab_table_benchmark_fast_*.csv --scale wide

run_ab_table_compare_plot:
	python3 plot_ab_table_compare_modes.py

# Debug build
debug: CXXFLAGS = -g -O0 -std=c++17 -Wall -DDEBUG
debug: all

# Clean
clean:
	rm -f $(TARGET_GENERATOR) $(TARGET_AB_GENERATOR) $(TARGET_SIMPLE_GENERATOR) $(TARGET_BENCHMARK) $(TARGET_AB_BENCHMARK) $(TARGET_AB_EMU_COMPARE) $(TARGET_SIMPLE_BENCHMARK) $(TARGET_TABLE_BENCHMARK) $(TARGET_INNER_DIM_BENCHMARK) $(TARGET_OUTER_DIM_BENCHMARK) $(TARGET_SINGLE_GEMM_BENCHMARK) $(TARGET_STANDALONE_BENCHMARK) $(TARGET_ASPECT_RATIO_GENERATOR) $(TARGET_TABLE_BENCHMARK_POWER) $(TARGET_TABLE_BENCHMARK_FP8) $(TARGET_TABLE_BENCHMARK_POWER_FP8) $(TARGET_GEMM_VS_GEMMLT_BENCHMARK) $(TARGET_GEMM_INT8_VS_FP8_BENCHMARK) $(TARGET_AB_TABLE_BENCHMARK) $(TARGET_AB_TABLE_BENCHMARK_FAST)
	rm -f M_cond_*.txt A_*.txt B_*.txt
	rm -f gemm_benchmark_*.csv ab_gemm_benchmark_*.csv simple_gemm_benchmark_*.csv

# Test generator with small matrix
test_generator: $(TARGET_GENERATOR)
	./$(TARGET_GENERATOR) 1024 1024

# Run benchmark with specified folder (usage: make run_benchmark MATRIX_FOLDER=path/to/matrices)
MATRIX_FOLDER ?= .
run_benchmark: $(TARGET_BENCHMARK)
	LD_LIBRARY_PATH=$(GEMMUL8_PATH)/lib:$(ROCM_PATH)/lib:$$LD_LIBRARY_PATH ./$(TARGET_BENCHMARK) $(MATRIX_FOLDER)

# Run A/B benchmark with specified folder (usage: make run_ab_benchmark MATRIX_FOLDER=path/to/matrices)
run_ab_benchmark: $(TARGET_AB_BENCHMARK)
	LD_LIBRARY_PATH=$(GEMMUL8_PATH)/lib:$(ROCM_PATH)/lib:$$LD_LIBRARY_PATH ./$(TARGET_AB_BENCHMARK) $(MATRIX_FOLDER)

# Run A/B benchmark and generate plots (usage: make run_ab_benchmark_plot MATRIX_FOLDER=path/to/matrices)
run_ab_benchmark_plot: $(TARGET_AB_BENCHMARK)
	LD_LIBRARY_PATH=$(GEMMUL8_PATH)/lib:$(ROCM_PATH)/lib:$$LD_LIBRARY_PATH ./$(TARGET_AB_BENCHMARK) $(MATRIX_FOLDER)
	python3 plot_benchmark.py $(MATRIX_FOLDER)

# Run simple benchmark with specified folder (usage: make run_simple_benchmark MATRIX_FOLDER=path/to/matrices)
run_simple_benchmark: $(TARGET_SIMPLE_BENCHMARK)
	LD_LIBRARY_PATH=$(GEMMUL8_PATH)/lib:$(ROCM_PATH)/lib:$$LD_LIBRARY_PATH ./$(TARGET_SIMPLE_BENCHMARK) $(MATRIX_FOLDER)

# Generate small test matrices and run benchmark
test: $(TARGET_GENERATOR) $(TARGET_BENCHMARK)
	$(eval TEST_FOLDER := test_matrices_1024x1024)
	./$(TARGET_GENERATOR) 1024 1024 $(TEST_FOLDER)
	LD_LIBRARY_PATH=$(GEMMUL8_PATH)/lib:$(ROCM_PATH)/lib:$$LD_LIBRARY_PATH ./$(TARGET_BENCHMARK) $(TEST_FOLDER)

.PHONY: all generator ab_generator simple_generator benchmark ab_benchmark ab_emu_compare simple_benchmark table_benchmark table_benchmark_power table_benchmark_fp8 table_benchmark_power_fp8 gemm_vs_gemmlt_benchmark gemm_int8_vs_fp8_benchmark ab_table_benchmark ab_table_benchmark_fast ab_table_size_benchmark run_ab_table_size_benchmark run_ab_table_benchmark run_ab_table_benchmark_fast run_ab_table_benchmark_plot run_ab_table_benchmark_fast_plot run_ab_table_compare_plot inner_dim_benchmark outer_dim_benchmark single_gemm_benchmark debug clean test_generator run_benchmark run_ab_benchmark run_ab_benchmark_plot run_simple_benchmark test
