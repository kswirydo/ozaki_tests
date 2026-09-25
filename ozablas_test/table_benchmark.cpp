/**
 * Table Benchmark: Generate square matrices and benchmark OzaBLAS performance
 * 
 * Generates matrices with condition number 10^8 (like rocEMU benchmark)
 * 
 * Outputs a table with:
 * - Matrix size
 * - Native DGEMM performance (TFLOPS)
 * - Ozaki Scheme I & II: Performance (TFLOPS) and Accuracy (relative error)
 * 
 * Usage:
 *   ./table_benchmark <size>
 *   ./table_benchmark <size1> <size2> <size3> ...
 *   ./table_benchmark            (runs default sizes: 1024 2048 4096 8192)
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <memory>
#include <chrono>
#include <cmath>

#include "ozablas/ozablas.hpp"
#include "ozablas/core/workspace.hpp"
#include "ozablas/core/executor.hpp"
#include "matrix_compare.hpp"

#ifdef OZA_BUILD_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cusolverDn.h>

#define GPU_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

#define CURAND_CHECK(call) do { \
    curandStatus_t status = call; \
    if (status != CURAND_STATUS_SUCCESS) { \
        std::cerr << "cuRAND error: " << status << std::endl; \
        exit(1); \
    } \
} while(0)

#define CUSOLVER_CHECK(call) do { \
    cusolverStatus_t status = call; \
    if (status != CUSOLVER_STATUS_SUCCESS) { \
        std::cerr << "cuSOLVER error: " << status << std::endl; \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error: " << status << std::endl; \
        exit(1); \
    } \
} while(0)
#endif

#ifdef OZA_BUILD_HIP
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>
#include <hiprand/hiprand.h>
#include <rocsolver/rocsolver.h>

#define GPU_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

#define HIPRAND_CHECK(call) do { \
    hiprandStatus_t status = call; \
    if (status != HIPRAND_STATUS_SUCCESS) { \
        std::cerr << "hipRAND error: " << status << std::endl; \
        exit(1); \
    } \
} while(0)

#define ROCBLAS_CHECK(call) do { \
    rocblas_status status = call; \
    if (status != rocblas_status_success) { \
        std::cerr << "rocBLAS error: " << status << std::endl; \
        exit(1); \
    } \
} while(0)

#define HIPBLAS_CHECK(call) do { \
    hipblasStatus_t status = call; \
    if (status != HIPBLAS_STATUS_SUCCESS) { \
        std::cerr << "hipBLAS error: " << status << std::endl; \
        exit(1); \
    } \
} while(0)
#endif

static const int NUM_WARMUP = 2;
static const int NUM_ITERATIONS = 50;  // Match rocEMU for stable measurements
static const int LOG10_COND = 8;  // Condition number 10^8

// =========================================================================
// GPU Kernels for conditioned matrix generation
// =========================================================================
#ifdef OZA_BUILD_HIP
__global__ void zero_matrix_kernel(double* A, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) A[idx] = 0.0;
}

__global__ void set_diagonal_kernel(double* S, size_t n, size_t ld, double* sv) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) S[idx * ld + idx] = sv[idx];
}

__global__ void compute_singular_values_kernel(double* sv, size_t n, double log10_cond) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double exponent = (n > 1) ? ((double)idx / (double)(n - 1) * log10_cond) : 0.0;
        sv[idx] = pow(10.0, exponent);
    }
}

void generate_conditioned_matrix(rocblas_handle rb_handle, hiprandGenerator_t gen,
                                  size_t n, int log10_cond, double* d_A) {
    size_t size = n * n;
    
    double *d_U, *d_V, *d_S, *d_temp, *d_tau, *d_sv;
    GPU_CHECK(hipMalloc(&d_U, size * sizeof(double)));
    GPU_CHECK(hipMalloc(&d_V, size * sizeof(double)));
    GPU_CHECK(hipMalloc(&d_S, size * sizeof(double)));
    GPU_CHECK(hipMalloc(&d_temp, size * sizeof(double)));
    GPU_CHECK(hipMalloc(&d_tau, n * sizeof(double)));
    GPU_CHECK(hipMalloc(&d_sv, n * sizeof(double)));
    
    // Generate random U and V
    HIPRAND_CHECK(hiprandGenerateNormalDouble(gen, d_U, size, 0.0, 1.0));
    HIPRAND_CHECK(hiprandGenerateNormalDouble(gen, d_V, size, 0.0, 1.0));
    
    // QR decomposition to get orthogonal matrices
    rocsolver_dgeqrf(rb_handle, n, n, d_U, n, d_tau);
    rocsolver_dorgqr(rb_handle, n, n, n, d_U, n, d_tau);
    
    rocsolver_dgeqrf(rb_handle, n, n, d_V, n, d_tau);
    rocsolver_dorgqr(rb_handle, n, n, n, d_V, n, d_tau);
    
    // Zero S and set diagonal with singular values
    int block_size = 256;
    size_t num_blocks = (size + block_size - 1) / block_size;
    hipLaunchKernelGGL(zero_matrix_kernel, dim3(num_blocks), dim3(block_size), 0, 0, d_S, size);
    
    num_blocks = (n + block_size - 1) / block_size;
    hipLaunchKernelGGL(compute_singular_values_kernel, dim3(num_blocks), dim3(block_size), 
                       0, 0, d_sv, n, (double)log10_cond);
    hipLaunchKernelGGL(set_diagonal_kernel, dim3(num_blocks), dim3(block_size), 
                       0, 0, d_S, n, n, d_sv);
    GPU_CHECK(hipDeviceSynchronize());
    
    // A = U * S * V'
    double alpha = 1.0, beta = 0.0;
    ROCBLAS_CHECK(rocblas_dgemm(rb_handle, rocblas_operation_none, rocblas_operation_none,
                                n, n, n, &alpha, d_U, n, d_S, n, &beta, d_temp, n));
    ROCBLAS_CHECK(rocblas_dgemm(rb_handle, rocblas_operation_none, rocblas_operation_transpose,
                                n, n, n, &alpha, d_temp, n, d_V, n, &beta, d_A, n));
    GPU_CHECK(hipDeviceSynchronize());
    
    GPU_CHECK(hipFree(d_U));
    GPU_CHECK(hipFree(d_V));
    GPU_CHECK(hipFree(d_S));
    GPU_CHECK(hipFree(d_temp));
    GPU_CHECK(hipFree(d_tau));
    GPU_CHECK(hipFree(d_sv));
}
#endif

#ifdef OZA_BUILD_CUDA
__global__ void zero_matrix_kernel_cuda(double* A, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) A[idx] = 0.0;
}

__global__ void set_diagonal_kernel_cuda(double* S, size_t n, size_t ld, double* sv) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) S[idx * ld + idx] = sv[idx];
}

__global__ void compute_singular_values_kernel_cuda(double* sv, size_t n, double log10_cond) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double exponent = (n > 1) ? ((double)idx / (double)(n - 1) * log10_cond) : 0.0;
        sv[idx] = pow(10.0, exponent);
    }
}

void generate_conditioned_matrix(cublasHandle_t cb_handle, cusolverDnHandle_t solver_handle,
                                  curandGenerator_t gen, size_t n, int log10_cond, double* d_A) {
    size_t size = n * n;
    
    double *d_U, *d_V, *d_S, *d_temp, *d_tau, *d_sv;
    int *d_info;
    GPU_CHECK(cudaMalloc(&d_U, size * sizeof(double)));
    GPU_CHECK(cudaMalloc(&d_V, size * sizeof(double)));
    GPU_CHECK(cudaMalloc(&d_S, size * sizeof(double)));
    GPU_CHECK(cudaMalloc(&d_temp, size * sizeof(double)));
    GPU_CHECK(cudaMalloc(&d_tau, n * sizeof(double)));
    GPU_CHECK(cudaMalloc(&d_sv, n * sizeof(double)));
    GPU_CHECK(cudaMalloc(&d_info, sizeof(int)));
    
    // Generate random U and V
    CURAND_CHECK(curandGenerateNormalDouble(gen, d_U, size, 0.0, 1.0));
    CURAND_CHECK(curandGenerateNormalDouble(gen, d_V, size, 0.0, 1.0));
    
    // QR decomposition workspace query
    int lwork_geqrf, lwork_orgqr;
    CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(solver_handle, n, n, d_U, n, &lwork_geqrf));
    CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(solver_handle, n, n, n, d_U, n, d_tau, &lwork_orgqr));
    int lwork = std::max(lwork_geqrf, lwork_orgqr);
    double* d_work;
    GPU_CHECK(cudaMalloc(&d_work, lwork * sizeof(double)));
    
    // QR decomposition for U
    CUSOLVER_CHECK(cusolverDnDgeqrf(solver_handle, n, n, d_U, n, d_tau, d_work, lwork, d_info));
    CUSOLVER_CHECK(cusolverDnDorgqr(solver_handle, n, n, n, d_U, n, d_tau, d_work, lwork, d_info));
    
    // QR decomposition for V
    CUSOLVER_CHECK(cusolverDnDgeqrf(solver_handle, n, n, d_V, n, d_tau, d_work, lwork, d_info));
    CUSOLVER_CHECK(cusolverDnDorgqr(solver_handle, n, n, n, d_V, n, d_tau, d_work, lwork, d_info));
    
    // Zero S and set diagonal with singular values
    int block_size = 256;
    size_t num_blocks = (size + block_size - 1) / block_size;
    zero_matrix_kernel_cuda<<<num_blocks, block_size>>>(d_S, size);
    
    num_blocks = (n + block_size - 1) / block_size;
    compute_singular_values_kernel_cuda<<<num_blocks, block_size>>>(d_sv, n, (double)log10_cond);
    set_diagonal_kernel_cuda<<<num_blocks, block_size>>>(d_S, n, n, d_sv);
    GPU_CHECK(cudaDeviceSynchronize());
    
    // A = U * S * V'
    double alpha = 1.0, beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(cb_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             n, n, n, &alpha, d_U, n, d_S, n, &beta, d_temp, n));
    CUBLAS_CHECK(cublasDgemm(cb_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             n, n, n, &alpha, d_temp, n, d_V, n, &beta, d_A, n));
    GPU_CHECK(cudaDeviceSynchronize());
    
    GPU_CHECK(cudaFree(d_U));
    GPU_CHECK(cudaFree(d_V));
    GPU_CHECK(cudaFree(d_S));
    GPU_CHECK(cudaFree(d_temp));
    GPU_CHECK(cudaFree(d_tau));
    GPU_CHECK(cudaFree(d_sv));
    GPU_CHECK(cudaFree(d_work));
    GPU_CHECK(cudaFree(d_info));
}
#endif

struct BenchmarkResult {
    double native_tflops;
    double scheme1_12_tflops;
    double scheme1_12_error;
    double scheme1_16_tflops;
    double scheme1_16_error;
    double scheme2_12_tflops;
    double scheme2_12_error;
    double scheme2_16_tflops;
    double scheme2_16_error;
};

double compute_frobenius_rel_error(const std::vector<double>& ref, const std::vector<double>& test) {
    double diff_sum = 0.0, norm_sum = 0.0;
    for (size_t i = 0; i < ref.size(); i++) {
        double diff = ref[i] - test[i];
        diff_sum += diff * diff;
        norm_sum += ref[i] * ref[i];
    }
    return (norm_sum > 0.0) ? std::sqrt(diff_sum) / std::sqrt(norm_sum) : std::sqrt(diff_sum);
}

BenchmarkResult run_benchmark(std::shared_ptr<ozablas::Executor> exec, size_t n, bool run_scheme1, bool run_scheme2) {
    BenchmarkResult result = {};
    
    size_t M = n, N = n, K = n;
    size_t size = M * N;
    double ops = 2.0 * static_cast<double>(M) * N * K;
    
    std::vector<double> h_C_native(size, 0.0);
    std::vector<double> h_C_ozaki(size, 0.0);
    
    // Allocate device memory (only A and C needed - we compute A * A like rocEMU)
    double *d_A, *d_C_native, *d_C_ozaki;
    exec->allocate((void**)&d_A, M * K * sizeof(double));
    exec->allocate((void**)&d_C_native, M * N * sizeof(double));
    exec->allocate((void**)&d_C_ozaki, M * N * sizeof(double));
    
    // Generate matrix with condition number 10^LOG10_COND on GPU
    std::cerr << "  Generating " << n << "x" << n << " matrix (cond=10^" << LOG10_COND << ")..." << std::flush;
    
#ifdef OZA_BUILD_HIP
    rocblas_handle rb_handle;
    hipblasHandle_t hb_handle;
    hiprandGenerator_t rand_gen;
    ROCBLAS_CHECK(rocblas_create_handle(&rb_handle));
    HIPBLAS_CHECK(hipblasCreate(&hb_handle));
    HIPRAND_CHECK(hiprandCreateGenerator(&rand_gen, HIPRAND_RNG_PSEUDO_DEFAULT));
    HIPRAND_CHECK(hiprandSetPseudoRandomGeneratorSeed(rand_gen, 12345ULL));
    
    generate_conditioned_matrix(rb_handle, rand_gen, n, LOG10_COND, d_A);
    
    HIPRAND_CHECK(hiprandDestroyGenerator(rand_gen));
    ROCBLAS_CHECK(rocblas_destroy_handle(rb_handle));
#elif defined(OZA_BUILD_CUDA)
    cublasHandle_t cb_handle;
    cusolverDnHandle_t solver_handle;
    curandGenerator_t rand_gen;
    CUBLAS_CHECK(cublasCreate(&cb_handle));
    CUSOLVER_CHECK(cusolverDnCreate(&solver_handle));
    CURAND_CHECK(curandCreateGenerator(&rand_gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(rand_gen, 12345ULL));
    
    generate_conditioned_matrix(cb_handle, solver_handle, rand_gen, n, LOG10_COND, d_A);
    
    CURAND_CHECK(curandDestroyGenerator(rand_gen));
    CUSOLVER_CHECK(cusolverDnDestroy(solver_handle));
    CUBLAS_CHECK(cublasDestroy(cb_handle));
#endif
    std::cerr << " done" << std::endl;
    
    const double alpha = 1.0, beta = 0.0;
    
    // =========================================================================
    // Native DGEMM Benchmark (C = A * A, like rocEMU)
    // =========================================================================
    std::cerr << "  Native DGEMM..." << std::flush;
    
#ifdef OZA_BUILD_HIP
    // Warmup
    for (int i = 0; i < NUM_WARMUP; i++) {
        HIPBLAS_CHECK(hipblasDgemm(hb_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                   n, n, n, &alpha, d_A, n, d_A, n, &beta, d_C_native, n));
    }
    GPU_CHECK(hipDeviceSynchronize());
#elif defined(OZA_BUILD_CUDA)
    for (int i = 0; i < NUM_WARMUP; i++) {
        CUBLAS_CHECK(cublasDgemm(cb_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 n, n, n, &alpha, d_A, n, d_A, n, &beta, d_C_native, n));
    }
    GPU_CHECK(cudaDeviceSynchronize());
#endif
    
    // Timed iterations
    auto start = std::chrono::high_resolution_clock::now();
#ifdef OZA_BUILD_HIP
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        HIPBLAS_CHECK(hipblasDgemm(hb_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                   n, n, n, &alpha, d_A, n, d_A, n, &beta, d_C_native, n));
    }
    GPU_CHECK(hipDeviceSynchronize());
#elif defined(OZA_BUILD_CUDA)
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        CUBLAS_CHECK(cublasDgemm(cb_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 n, n, n, &alpha, d_A, n, d_A, n, &beta, d_C_native, n));
    }
    GPU_CHECK(cudaDeviceSynchronize());
#endif
    auto end = std::chrono::high_resolution_clock::now();
    
    double native_ms = std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERATIONS;
    result.native_tflops = (ops * 1e-12) / (native_ms * 1e-3);
    
    // Copy native result for accuracy comparison
    exec->copy_to_host(h_C_native.data(), d_C_native, M * N * sizeof(double));
    std::cerr << " " << std::fixed << std::setprecision(2) << result.native_tflops << " TFLOPS" << std::endl;

#ifdef OZA_BUILD_HIP
    HIPBLAS_CHECK(hipblasDestroy(hb_handle));
#endif
    
    // =========================================================================
    // Ozaki Scheme I Benchmarks (12 and 16 slices) - C = A * A
    // WARNING: Scheme I scales O(S^2), so 12 slices = 144 GEMMs, 16 slices = 256 GEMMs
    // =========================================================================
    if (run_scheme1) for (int slices : {12, 16}) {
        int num_gemms = slices * slices;
        std::cerr << "  Scheme I (s=" << slices << ", " << num_gemms << " GEMMs per call)..." << std::endl;
        
        std::cerr << "    Allocating workspace..." << std::flush;
        ozablas::WorkspaceScheme1 ws(exec, M, N, K, slices);
        std::cerr << " done" << std::endl;
        
        std::cerr << "    Warmup (" << NUM_WARMUP << " iters, " << NUM_WARMUP * num_gemms << " total GEMMs)..." << std::flush;
        for (int i = 0; i < NUM_WARMUP; i++) {
            std::cerr << "[" << (i+1) << ": calling gemm..." << std::flush;
            ozablas::ozaki_scheme1_gemm(ws, d_A, d_A, d_C_ozaki);
            std::cerr << "sync..." << std::flush;
            exec->synchronize();
            std::cerr << "done]" << std::flush;
        }
        std::cerr << " done" << std::endl;
        
        std::cerr << "    Timing (" << NUM_ITERATIONS << " iters, " << NUM_ITERATIONS * num_gemms << " total GEMMs)..." << std::flush;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            ozablas::ozaki_scheme1_gemm(ws, d_A, d_A, d_C_ozaki);
            exec->synchronize();
            if ((i + 1) % 10 == 0) std::cerr << (i + 1) << std::flush;
            else std::cerr << "." << std::flush;
        }
        end = std::chrono::high_resolution_clock::now();
        std::cerr << " done" << std::endl;
        
        double ms = std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERATIONS;
        double tflops = (ops * 1e-12) / (ms * 1e-3);
        
        std::cerr << "    Copying result..." << std::flush;
        exec->copy_to_host(h_C_ozaki.data(), d_C_ozaki, M * N * sizeof(double));
        std::cerr << " done" << std::endl;
        
        double error = compute_frobenius_rel_error(h_C_native, h_C_ozaki);
        auto metrics = matrix_utils::compare::compute_errors(h_C_native, h_C_ozaki);
        
        if (slices == 12) {
            result.scheme1_12_tflops = tflops;
            result.scheme1_12_error = error;
        } else {
            result.scheme1_16_tflops = tflops;
            result.scheme1_16_error = error;
        }
        
        std::cerr << "    Result: " << std::fixed << std::setprecision(2) << tflops << " TFLOPS" << std::endl;
        std::cerr << "    Frobenius rel err: " << std::scientific << std::setprecision(2) << error << std::endl;
        std::cerr << "    Max rel err:       " << std::scientific << std::setprecision(2) << metrics.max_relative_error << std::endl;
    }
    
    // =========================================================================
    // Ozaki Scheme II Benchmarks (12 and 16 slices) - C = A * A
    // Scheme II scales O(S), so 12 slices = 12 GEMMs, 16 slices = 16 GEMMs
    // =========================================================================
    if (run_scheme2) for (int slices : {12, 16}) {
        std::cerr << "  Scheme II (s=" << slices << ", " << slices << " GEMMs)..." << std::endl;
        
        std::cerr << "    Allocating workspace..." << std::flush;
        ozablas::WorkspaceScheme2 ws(exec, M, N, K, slices);
        std::cerr << " done" << std::endl;
        
        std::cerr << "    Warmup (" << NUM_WARMUP << " iters)..." << std::flush;
        for (int i = 0; i < NUM_WARMUP; i++) {
            ozablas::ozaki_scheme2_gemm(ws, d_A, d_A, d_C_ozaki);
            exec->synchronize();
            std::cerr << "." << std::flush;
        }
        std::cerr << " done" << std::endl;
        
        std::cerr << "    Timing (" << NUM_ITERATIONS << " iters)..." << std::flush;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            ozablas::ozaki_scheme2_gemm(ws, d_A, d_A, d_C_ozaki);
            exec->synchronize();
            if ((i + 1) % 10 == 0) std::cerr << (i + 1) << std::flush;
            else std::cerr << "." << std::flush;
        }
        end = std::chrono::high_resolution_clock::now();
        std::cerr << " done" << std::endl;
        
        double ms = std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERATIONS;
        double tflops = (ops * 1e-12) / (ms * 1e-3);
        
        std::cerr << "    Copying result..." << std::flush;
        exec->copy_to_host(h_C_ozaki.data(), d_C_ozaki, M * N * sizeof(double));
        std::cerr << " done" << std::endl;
        
        double error = compute_frobenius_rel_error(h_C_native, h_C_ozaki);
        auto metrics = matrix_utils::compare::compute_errors(h_C_native, h_C_ozaki);
        
        if (slices == 12) {
            result.scheme2_12_tflops = tflops;
            result.scheme2_12_error = error;
        } else {
            result.scheme2_16_tflops = tflops;
            result.scheme2_16_error = error;
        }
        
        std::cerr << "    Result: " << std::fixed << std::setprecision(2) << tflops << " TFLOPS" << std::endl;
        std::cerr << "    Frobenius rel err: " << std::scientific << std::setprecision(2) << error << std::endl;
        std::cerr << "    Max rel err:       " << std::scientific << std::setprecision(2) << metrics.max_relative_error << std::endl;
    }
    
    // Cleanup
    exec->free(d_A);
    exec->free(d_C_native);
    exec->free(d_C_ozaki);
    
    return result;
}

void print_table_header(bool run_scheme1, bool run_scheme2) {
    std::cout << "+-------------+-------------+";
    if (run_scheme1) std::cout << "-----------------------------+-----------------------------+";
    if (run_scheme2) std::cout << "-----------------------------+-----------------------------+";
    std::cout << std::endl;
    
    std::cout << "| Matrix Size | Native GEMM |";
    if (run_scheme1) std::cout << "   Ozaki Scheme I (s=12)     |   Ozaki Scheme I (s=16)     |";
    if (run_scheme2) std::cout << "   Ozaki Scheme II (s=12)    |   Ozaki Scheme II (s=16)    |";
    std::cout << std::endl;
    
    std::cout << "+-------------+-------------+";
    if (run_scheme1) std::cout << "-----------------------------+-----------------------------+";
    if (run_scheme2) std::cout << "-----------------------------+-----------------------------+";
    std::cout << std::endl;
    
    std::cout << "|             | Performance |";
    if (run_scheme1) std::cout << " Performance | Accuracy      | Performance | Accuracy      |";
    if (run_scheme2) std::cout << " Performance | Accuracy      | Performance | Accuracy      |";
    std::cout << std::endl;
    
    std::cout << "+-------------+-------------+";
    if (run_scheme1) std::cout << "-------------+---------------+-------------+---------------+";
    if (run_scheme2) std::cout << "-------------+---------------+-------------+---------------+";
    std::cout << std::endl;
}

void print_table_row(size_t n, const BenchmarkResult& r, bool run_scheme1, bool run_scheme2) {
    std::cout << "| " << std::setw(11) << n 
              << " | " << std::setw(8) << std::fixed << std::setprecision(2) << r.native_tflops << " TF";
    if (run_scheme1) {
        std::cout << " | " << std::setw(8) << std::setprecision(2) << r.scheme1_12_tflops << " TF"
                  << " | " << std::scientific << std::setprecision(2) << r.scheme1_12_error
                  << " | " << std::fixed << std::setw(8) << std::setprecision(2) << r.scheme1_16_tflops << " TF"
                  << " | " << std::scientific << std::setprecision(2) << r.scheme1_16_error;
    }
    if (run_scheme2) {
        std::cout << " | " << std::fixed << std::setw(8) << std::setprecision(2) << r.scheme2_12_tflops << " TF"
                  << " | " << std::scientific << std::setprecision(2) << r.scheme2_12_error
                  << " | " << std::fixed << std::setw(8) << std::setprecision(2) << r.scheme2_16_tflops << " TF"
                  << " | " << std::scientific << std::setprecision(2) << r.scheme2_16_error;
    }
    std::cout << " |" << std::endl;
}

void print_table_footer(bool run_scheme1, bool run_scheme2) {
    std::cout << "+-------------+-------------+";
    if (run_scheme1) std::cout << "-------------+---------------+-------------+---------------+";
    if (run_scheme2) std::cout << "-------------+---------------+-------------+---------------+";
    std::cout << std::endl;
}

void print_summary_table(const std::vector<size_t>& sizes, const std::vector<BenchmarkResult>& results, bool run_scheme1, bool run_scheme2) {
    std::cout << std::endl;
    std::cout << "======================================================================================================================" << std::endl;
    std::cout << "                                              OZABLAS BENCHMARK SUMMARY" << std::endl;
    std::cout << "                                          Condition Number: 10^" << LOG10_COND << std::endl;
    std::cout << "======================================================================================================================" << std::endl;
    std::cout << std::endl;
    
    print_table_header(run_scheme1, run_scheme2);
    for (size_t i = 0; i < sizes.size(); i++) {
        print_table_row(sizes[i], results[i], run_scheme1, run_scheme2);
    }
    print_table_footer(run_scheme1, run_scheme2);
    
    std::cout << std::endl;
    std::cout << "Legend:" << std::endl;
    std::cout << "  - Performance in TFLOPS (TF)" << std::endl;
    std::cout << "  - Accuracy: Relative Frobenius error vs native FP64 DGEMM" << std::endl;
    std::cout << "  - s=12, s=16: Number of slices (more slices = better accuracy, lower performance)" << std::endl;
    std::cout << std::endl;
    std::cout << "Accuracy Metrics (detailed in stderr):" << std::endl;
    std::cout << "  - Frobenius rel err: ||C_ozaki - C_native||_F / ||C_native||_F" << std::endl;
    std::cout << "  - Max rel err: max_i |C_ozaki[i] - C_native[i]| / |C_native[i]|" << std::endl;
    std::cout << "======================================================================================================================" << std::endl;
}

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options] [size1] [size2] ..." << std::endl;
    std::cerr << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --scheme1, -1    Run Ozaki Scheme I only" << std::endl;
    std::cerr << "  --scheme2, -2    Run Ozaki Scheme II only" << std::endl;
    std::cerr << "  --both, -b       Run both schemes (default)" << std::endl;
    std::cerr << "  --help, -h       Show this help" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Examples:" << std::endl;
    std::cerr << "  " << prog << " 4096 8192           # Both schemes, sizes 4096 and 8192" << std::endl;
    std::cerr << "  " << prog << " -1 16384            # Scheme I only, size 16384" << std::endl;
    std::cerr << "  " << prog << " --scheme2 4096 8192 # Scheme II only" << std::endl;
}

int main(int argc, char* argv[]) {
    std::vector<size_t> sizes;
    bool run_scheme1 = true;
    bool run_scheme2 = true;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--scheme1" || arg == "-1") {
            run_scheme1 = true;
            run_scheme2 = false;
        } else if (arg == "--scheme2" || arg == "-2") {
            run_scheme1 = false;
            run_scheme2 = true;
        } else if (arg == "--both" || arg == "-b") {
            run_scheme1 = true;
            run_scheme2 = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg[0] == '-') {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        } else {
            sizes.push_back(std::stoull(arg));
        }
    }
    
    if (sizes.empty()) {
        sizes = {1024, 2048, 4096, 8192};
        std::cerr << "No sizes provided, using defaults: ";
        for (size_t s : sizes) std::cerr << s << " ";
        std::cerr << std::endl;
    }
    
    std::cerr << "Condition number: 10^" << LOG10_COND << std::endl;
    std::cerr << "Running: ";
    if (run_scheme1 && run_scheme2) std::cerr << "Scheme I + II";
    else if (run_scheme1) std::cerr << "Scheme I only";
    else std::cerr << "Scheme II only";
    std::cerr << std::endl;
    std::cerr << "Matrix sizes: ";
    for (size_t s : sizes) std::cerr << s << " ";
    std::cerr << std::endl << std::endl;
    
    // Initialize executor
    std::shared_ptr<ozablas::Executor> exec;
#ifdef OZA_BUILD_CUDA
    exec = std::make_shared<ozablas::CudaExecutor>(0);
    std::cerr << "Using CUDA backend" << std::endl;
#elif defined(OZA_BUILD_HIP)
    exec = std::make_shared<ozablas::HipExecutor>(0);
    std::cerr << "Using HIP backend" << std::endl;
#else
    std::cerr << "Error: No GPU backend available (neither CUDA nor HIP)" << std::endl;
    return 1;
#endif
    
    std::cerr << std::endl;
    
    // Collect all results
    std::vector<BenchmarkResult> results;
    
    for (size_t n : sizes) {
        std::cerr << "Processing " << n << "x" << n << "..." << std::endl;
        BenchmarkResult result = run_benchmark(exec, n, run_scheme1, run_scheme2);
        results.push_back(result);
        std::cerr << std::endl;
    }
    
    // Print summary table at the end
    print_summary_table(sizes, results, run_scheme1, run_scheme2);
    
    return 0;
}
