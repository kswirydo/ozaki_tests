# ============================================================================
# OzaBLAS table benchmark (Ozaki Scheme I & II) — AMD MI355X / HIP
#
# In the paper this benchmark was run on MI355X only, so this Makefile targets
# the HIP/ROCm backend. (A CUDA build is also available via CMakeLists.txt.)
#
# ozablas is an EXTERNAL dependency: https://github.com/IlyaNyrkov/ozablas
# Build it first, then point OZABLAS_ROOT at your checkout:
#
#     cd $OZABLAS_ROOT && make release-hip      # builds libozablas.so
#     make OZABLAS_ROOT=/path/to/ozablas        # (here) builds table_benchmark
#
# Run:
#     LD_LIBRARY_PATH=$OZABLAS_ROOT/build_hip_release/src:$LD_LIBRARY_PATH \
#         ./table_benchmark 4096 8192 16384
# ============================================================================

OZABLAS_ROOT ?= /home/kswirydo/ozablas
ROCM_PATH    ?= /opt/rocm

# Locate the prebuilt ozablas library (release-hip build, else generic build).
OZABLAS_LIBDIR := $(firstword $(wildcard $(OZABLAS_ROOT)/build_hip_release/src $(OZABLAS_ROOT)/build/src))

HIPCC    = $(ROCM_PATH)/bin/hipcc
CXXFLAGS = -O3 -std=c++20 -march=native -funroll-loops -DOZA_BUILD_HIP
INCLUDES = -I$(OZABLAS_ROOT)/include -I$(OZABLAS_ROOT)/utils -I$(ROCM_PATH)/include
LDFLAGS  = -L$(OZABLAS_LIBDIR) -Wl,-rpath,$(OZABLAS_LIBDIR) -L$(ROCM_PATH)/lib
LIBS     = -lozablas -lhipblas -lrocblas -lhiprand -lrocsolver

TARGET = table_benchmark

.PHONY: all clean
all: $(TARGET)

$(TARGET): table_benchmark.cpp
	@test -n "$(OZABLAS_LIBDIR)" || { \
		echo "ERROR: libozablas not found under $(OZABLAS_ROOT)."; \
		echo "       Build ozablas first ('make release-hip') or set OZABLAS_ROOT=/path/to/ozablas."; \
		exit 1; }
	$(HIPCC) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS) $(LIBS)

clean:
	rm -f $(TARGET)
