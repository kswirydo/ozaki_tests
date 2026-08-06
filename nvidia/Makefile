# CUDA builds for rocEMU (NVIDIA / GEMMul8)

CUDA_PATH    ?= /usr/local/cuda-13.1
GEMMUL8_PATH ?= /mnt/local/home/kswirydo/GEMMul8
GPU_ARCH     ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')

NVCC      = $(CUDA_PATH)/bin/nvcc
NVCC_ARCH = -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)

CXXFLAGS = -O3 -std=c++20 -Xcompiler -Wall
INCLUDES = -I$(GEMMUL8_PATH)/include
LDFLAGS  = -L$(GEMMUL8_PATH)/lib -Xlinker -rpath -Xlinker $(GEMMUL8_PATH)/lib
LIBS_GEMMUL8 = -lgemmul8 -lcublas -lcublasLt -lcudart -lcuda -lnvidia-ml -ldl

TARGET_AB_EMU_COMPARE = ab_gemm_emu_compare
TARGET_AB_TABLE_LATEX = ab_table_latex_benchmark

# Matrix generation (cuSOLVER QR + cuRAND) is needed by the LaTeX table benchmark.
LIBS_GEN = -lcusolver -lcurand

.PHONY: all ab_gemm_emu_compare ab_table_latex_benchmark clean

all: ab_gemm_emu_compare ab_table_latex_benchmark

ab_gemm_emu_compare: ab_gemm_emu_compare.cu cuda_helpers.cuh
	$(NVCC) $(NVCC_ARCH) $(CXXFLAGS) $(INCLUDES) ab_gemm_emu_compare.cu -o $(TARGET_AB_EMU_COMPARE) $(LDFLAGS) $(LIBS_GEMMUL8)

ab_table_latex_benchmark: ab_table_latex_benchmark.cu cuda_helpers.cuh cuda_conditioned_matrix.cuh
	$(NVCC) $(NVCC_ARCH) $(CXXFLAGS) $(INCLUDES) ab_table_latex_benchmark.cu -o $(TARGET_AB_TABLE_LATEX) $(LDFLAGS) $(LIBS_GEMMUL8) $(LIBS_GEN)

clean:
	rm -f $(TARGET_AB_EMU_COMPARE) $(TARGET_AB_TABLE_LATEX)
