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

# Libraries for matrix generator
LIBS_GENERATOR = -lrocblas -lrocsolver -lhiprand

# Libraries for benchmark (includes GEMMul8)
INCLUDES_BENCHMARK = $(INCLUDES) -I$(GEMMUL8_PATH)/include
LDFLAGS_BENCHMARK = $(LDFLAGS) -L$(GEMMUL8_PATH)/lib
LIBS_BENCHMARK = -lgemmul8 -lhipblas -lamdhip64

# Targets
TARGET_GENERATOR = matrix_generator
TARGET_BENCHMARK = gemm_benchmark

# Source files
SRCS_GENERATOR = matrix_generator.cpp
SRCS_BENCHMARK = gemm_benchmark.cu

# Default target: build both
all: $(TARGET_GENERATOR) $(TARGET_BENCHMARK)

# Matrix generator
$(TARGET_GENERATOR): $(SRCS_GENERATOR)
	$(HIPCC) $(CXXFLAGS) $(INCLUDES) $(SRCS_GENERATOR) -o $(TARGET_GENERATOR) $(LDFLAGS) $(LIBS_GENERATOR)

# GEMM benchmark
$(TARGET_BENCHMARK): $(SRCS_BENCHMARK)
	$(HIPCC) $(CXXFLAGS) $(INCLUDES_BENCHMARK) $(SRCS_BENCHMARK) -o $(TARGET_BENCHMARK) $(LDFLAGS_BENCHMARK) $(LIBS_BENCHMARK)

# Build only generator
generator: $(TARGET_GENERATOR)

# Build only benchmark
benchmark: $(TARGET_BENCHMARK)

# Debug build
debug: CXXFLAGS = -g -O0 -std=c++17 -Wall -DDEBUG
debug: all

# Clean
clean:
	rm -f $(TARGET_GENERATOR) $(TARGET_BENCHMARK)
	rm -f M_cond_*.txt
	rm -f gemm_benchmark_*.csv

# Test generator with small matrix
test_generator: $(TARGET_GENERATOR)
	./$(TARGET_GENERATOR) 1024 1024

# Run benchmark with specified folder (usage: make run_benchmark MATRIX_FOLDER=path/to/matrices)
MATRIX_FOLDER ?= .
run_benchmark: $(TARGET_BENCHMARK)
	LD_LIBRARY_PATH=$(GEMMUL8_PATH)/lib:$(ROCM_PATH)/lib:$$LD_LIBRARY_PATH ./$(TARGET_BENCHMARK) $(MATRIX_FOLDER)

# Generate small test matrices and run benchmark
# Creates folder like: test_matrices_1024x1024_YYYYMMDD_HHMMSS
test: $(TARGET_GENERATOR) $(TARGET_BENCHMARK)
	$(eval TEST_FOLDER := test_matrices_1024x1024)
	./$(TARGET_GENERATOR) 1024 1024 $(TEST_FOLDER)
	LD_LIBRARY_PATH=$(GEMMUL8_PATH)/lib:$(ROCM_PATH)/lib:$$LD_LIBRARY_PATH ./$(TARGET_BENCHMARK) $(TEST_FOLDER)

.PHONY: all generator benchmark debug clean test_generator run_benchmark test
