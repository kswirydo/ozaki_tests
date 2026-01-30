# Makefile for ROCm Matrix Generator and GEMM Benchmark

# ROCm installation path (adjust if different)
ROCM_PATH ?= /opt/rocm

# GEMMul8 path
GEMMUL8_PATH ?= /home/kswirydo/GEMMul8/GEMMul8

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
LDFLAGS_BENCHMARK = $(LDFLAGS) -L$(GEMMUL8_PATH)/lib
LIBS_BENCHMARK = -lgemmul8 -lhipblas -lamdhip64

# Targets
TARGET_GENERATOR = matrix_generator
TARGET_AB_GENERATOR = ab_matrix_generator
TARGET_SIMPLE_GENERATOR = simple_matrix_generator
TARGET_BENCHMARK = gemm_benchmark
TARGET_AB_BENCHMARK = ab_gemm_benchmark
TARGET_SIMPLE_BENCHMARK = simple_gemm_benchmark

# Source files
SRCS_GENERATOR = matrix_generator.cpp
SRCS_AB_GENERATOR = ab_matrix_generator.cpp
SRCS_SIMPLE_GENERATOR = simple_matrix_generator.cpp
SRCS_BENCHMARK = gemm_benchmark.cu
SRCS_AB_BENCHMARK = ab_gemm_benchmark.cu
SRCS_SIMPLE_BENCHMARK = simple_gemm_benchmark.cu

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

# Simple GEMM benchmark (for simple A*B matrices without condition numbers)
$(TARGET_SIMPLE_BENCHMARK): $(SRCS_SIMPLE_BENCHMARK)
	$(HIPCC) $(CXXFLAGS) $(INCLUDES_BENCHMARK) $(SRCS_SIMPLE_BENCHMARK) -o $(TARGET_SIMPLE_BENCHMARK) $(LDFLAGS_BENCHMARK) $(LIBS_BENCHMARK)

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

# Build only simple benchmark
simple_benchmark: $(TARGET_SIMPLE_BENCHMARK)

# Debug build
debug: CXXFLAGS = -g -O0 -std=c++17 -Wall -DDEBUG
debug: all

# Clean
clean:
	rm -f $(TARGET_GENERATOR) $(TARGET_AB_GENERATOR) $(TARGET_SIMPLE_GENERATOR) $(TARGET_BENCHMARK) $(TARGET_AB_BENCHMARK) $(TARGET_SIMPLE_BENCHMARK)
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

# Run simple benchmark with specified folder (usage: make run_simple_benchmark MATRIX_FOLDER=path/to/matrices)
run_simple_benchmark: $(TARGET_SIMPLE_BENCHMARK)
	LD_LIBRARY_PATH=$(GEMMUL8_PATH)/lib:$(ROCM_PATH)/lib:$$LD_LIBRARY_PATH ./$(TARGET_SIMPLE_BENCHMARK) $(MATRIX_FOLDER)

# Generate small test matrices and run benchmark
test: $(TARGET_GENERATOR) $(TARGET_BENCHMARK)
	$(eval TEST_FOLDER := test_matrices_1024x1024)
	./$(TARGET_GENERATOR) 1024 1024 $(TEST_FOLDER)
	LD_LIBRARY_PATH=$(GEMMUL8_PATH)/lib:$(ROCM_PATH)/lib:$$LD_LIBRARY_PATH ./$(TARGET_BENCHMARK) $(TEST_FOLDER)

.PHONY: all generator ab_generator simple_generator benchmark ab_benchmark simple_benchmark debug clean test_generator run_benchmark run_ab_benchmark run_simple_benchmark test
