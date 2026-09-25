# ============================================================================
# Standalone OzaBLAS Table Benchmark (Ozaki Scheme I & II)
# ============================================================================
# ozablas is an EXTERNAL dependency (https://github.com/IlyaNyrkov/ozablas).
# Build it first, then point this project at your checkout via OZABLAS_ROOT:
#
#   # in your ozablas checkout:
#   make release-hip     # AMD GPUs (MI355X)   -- used for the paper
#   # or: make release-cuda
#
#   # here:
#   mkdir build && cd build
#   cmake .. -DOZABLAS_ENABLE_HIP=ON -DOZABLAS_ROOT=/path/to/ozablas
#   make
# ============================================================================

cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(OzaBLAS_TableBenchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops")

# Options
option(OZABLAS_ENABLE_CUDA "Enable NVIDIA CUDA backend" OFF)
option(OZABLAS_ENABLE_HIP  "Enable AMD HIP backend" OFF)

# Path to the external ozablas checkout (override with -DOZABLAS_ROOT=... or the
# OZABLAS_ROOT environment variable).
if(NOT OZABLAS_ROOT)
    if(DEFINED ENV{OZABLAS_ROOT})
        set(OZABLAS_ROOT "$ENV{OZABLAS_ROOT}")
    else()
        set(OZABLAS_ROOT "/home/kswirydo/ozablas")
    endif()
endif()
message(STATUS "Using OZABLAS_ROOT = ${OZABLAS_ROOT}")

# ============================================================================
# HIP Backend
# ============================================================================
if(OZABLAS_ENABLE_HIP)
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)

    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES "native")
    endif()

    list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
    find_package(hip REQUIRED CONFIG)
    find_package(hipblas REQUIRED CONFIG)
    find_package(rocblas REQUIRED CONFIG)
    find_package(hiprand REQUIRED CONFIG)
    find_package(rocsolver REQUIRED CONFIG)

    add_compile_definitions(OZA_BUILD_HIP)

    # Find pre-built ozablas library
    find_library(OZABLAS_LIB ozablas
        HINTS ${OZABLAS_ROOT}/build_hip_release/src
              ${OZABLAS_ROOT}/build/src
    )

    if(NOT OZABLAS_LIB)
        message(FATAL_ERROR "Could not find libozablas.so under ${OZABLAS_ROOT}. Build ozablas first with 'make release-hip'")
    endif()

    add_executable(table_benchmark table_benchmark.cpp)
    set_source_files_properties(table_benchmark.cpp PROPERTIES LANGUAGE HIP)

    target_include_directories(table_benchmark PRIVATE
        ${OZABLAS_ROOT}/include
        ${OZABLAS_ROOT}/utils
    )

    target_link_libraries(table_benchmark PRIVATE
        ${OZABLAS_LIB}
        roc::hipblas
        roc::rocblas
        hip::hiprand
        roc::rocsolver
    )
endif()

# ============================================================================
# CUDA Backend
# ============================================================================
if(OZABLAS_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "native")
    endif()

    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(OZA_BUILD_CUDA)

    # Find pre-built ozablas library
    find_library(OZABLAS_LIB ozablas
        HINTS ${OZABLAS_ROOT}/build_cuda_release/src
              ${OZABLAS_ROOT}/build/src
    )

    if(NOT OZABLAS_LIB)
        message(FATAL_ERROR "Could not find libozablas.so under ${OZABLAS_ROOT}. Build ozablas first with 'make release-cuda'")
    endif()

    add_executable(table_benchmark table_benchmark.cpp)
    set_source_files_properties(table_benchmark.cpp PROPERTIES LANGUAGE CUDA)

    target_include_directories(table_benchmark PRIVATE
        ${OZABLAS_ROOT}/include
        ${OZABLAS_ROOT}/utils
    )

    target_link_libraries(table_benchmark PRIVATE
        ${OZABLAS_LIB}
        CUDA::cublas
        CUDA::cudart
        CUDA::curand
        CUDA::cusolver
    )
endif()

if(NOT OZABLAS_ENABLE_CUDA AND NOT OZABLAS_ENABLE_HIP)
    message(FATAL_ERROR "Please set -DOZABLAS_ENABLE_HIP=ON or -DOZABLAS_ENABLE_CUDA=ON")
endif()
