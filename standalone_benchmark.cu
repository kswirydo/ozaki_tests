/**
 * Standalone GEMM Benchmark
 * 
 * Computes C(MxN) = A(MxK) * B(KxN)
 * Compares Native FP64 GEMM vs Ozaki-II (12 splits only)
 * 
 * Usage:
 *   ./standalone_benchmark <M> <N> <K>
 */

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <hiprand/hiprand.h>
#include "gemmul8.hpp"
#include <cstdlib>

// Enable GEMMUL8_PROFILE support
namespace oz2 { extern bool g_profiling_enabled; }

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(err) << std::endl; \
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

#define ROCBLAS_CHECK(call) do { \
    rocblas_status status = call; \
    if (status != rocblas_status_success) { \
        std::cerr << "rocBLAS error: " << status << std::endl; \
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

static const int NUM_WARMUP = 5;
static const int NUM_ITERATIONS = 50;
static const int LOG10_COND = 8;
static const int NUM_MODULI = 12;

__global__ void zero_matrix_kernel(double* A, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) A[idx] = 0.0;
}

__global__ void set_diagonal_kernel(double* S, size_t rows, size_t cols, size_t ld, double* sv) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t min_dim = (rows < cols) ? rows : cols;
    if (idx < min_dim) S[idx * ld + idx] = sv[idx];
}

__global__ void compute_singular_values_kernel(double* sv, size_t n, double log10_cond) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double exponent = (n > 1) ? ((double)idx / (double)(n - 1) * log10_cond) : 0.0;
        sv[idx] = pow(10.0, exponent);
    }
}

void generate_conditioned_matrix(rocblas_handle rb_handle, hiprandGenerator_t gen,
                                  size_t rows, size_t cols, int log10_cond, double* d_A) {
    size_t min_dim = std::min(rows, cols);
    size_t size_U = rows * min_dim;
    size_t size_V = cols * min_dim;
    size_t size_S = min_dim * min_dim;
    
    double *d_U, *d_V, *d_S, *d_temp, *d_tau, *d_sv;
    HIP_CHECK(hipMalloc(&d_U, size_U * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_V, size_V * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_S, size_S * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_temp, rows * min_dim * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_tau, min_dim * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_sv, min_dim * sizeof(double)));
    
    HIPRAND_CHECK(hiprandGenerateNormalDouble(gen, d_U, size_U, 0.0, 1.0));
    HIPRAND_CHECK(hiprandGenerateNormalDouble(gen, d_V, size_V, 0.0, 1.0));
    
    rocsolver_dgeqrf(rb_handle, rows, min_dim, d_U, rows, d_tau);
    rocsolver_dorgqr(rb_handle, rows, min_dim, min_dim, d_U, rows, d_tau);
    
    rocsolver_dgeqrf(rb_handle, cols, min_dim, d_V, cols, d_tau);
    rocsolver_dorgqr(rb_handle, cols, min_dim, min_dim, d_V, cols, d_tau);
    
    int block_size = 256;
    size_t num_blocks = (size_S + block_size - 1) / block_size;
    hipLaunchKernelGGL(zero_matrix_kernel, dim3(num_blocks), dim3(block_size), 0, 0, d_S, size_S);
    
    num_blocks = (min_dim + block_size - 1) / block_size;
    hipLaunchKernelGGL(compute_singular_values_kernel, dim3(num_blocks), dim3(block_size), 
                       0, 0, d_sv, min_dim, (double)log10_cond);
    hipLaunchKernelGGL(set_diagonal_kernel, dim3(num_blocks), dim3(block_size), 
                       0, 0, d_S, min_dim, min_dim, min_dim, d_sv);
    HIP_CHECK(hipDeviceSynchronize());
    
    double alpha = 1.0, beta = 0.0;
    ROCBLAS_CHECK(rocblas_dgemm(rb_handle, rocblas_operation_none, rocblas_operation_none,
                                rows, min_dim, min_dim, &alpha, d_U, rows, d_S, min_dim, &beta, d_temp, rows));
    ROCBLAS_CHECK(rocblas_dgemm(rb_handle, rocblas_operation_none, rocblas_operation_transpose,
                                rows, cols, min_dim, &alpha, d_temp, rows, d_V, cols, &beta, d_A, rows));
    HIP_CHECK(hipDeviceSynchronize());
    
    HIP_CHECK(hipFree(d_U));
    HIP_CHECK(hipFree(d_V));
    HIP_CHECK(hipFree(d_S));
    HIP_CHECK(hipFree(d_temp));
    HIP_CHECK(hipFree(d_tau));
    HIP_CHECK(hipFree(d_sv));
}

double compute_frobenius_rel_error(const double* C_ref, const double* C_test, size_t n) {
    double diff_sum = 0.0, norm_sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = C_ref[i] - C_test[i];
        diff_sum += diff * diff;
        norm_sum += C_ref[i] * C_ref[i];
    }
    return std::sqrt(diff_sum) / std::sqrt(norm_sum);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <M> <N> <K>" << std::endl;
        std::cerr << "  Computes C(MxN) = A(MxK) * B(KxN)" << std::endl;
        return 1;
    }
    
    // Check for GEMMUL8_PROFILE environment variable
    const char* prof = getenv("GEMMUL8_PROFILE");
    if (prof && std::string(prof) == "1") {
        oz2::g_profiling_enabled = true;
        std::cerr << "[GEMMUL8] Profiling enabled" << std::endl;
    }
    
    size_t M = std::stoull(argv[1]);
    size_t N = std::stoull(argv[2]);
    size_t K = std::stoull(argv[3]);
    
    size_t size_A = M * K;
    size_t size_B = K * N;
    size_t size_C = M * N;
    
    // Memory calculations
    size_t mem_A = size_A * sizeof(double);
    size_t mem_B = size_B * sizeof(double);
    size_t mem_C = size_C * sizeof(double);
    size_t mem_matrices = mem_A + mem_B + mem_C;
    size_t mem_workspace = gemmul8::workSize(M, N, K, NUM_MODULI);
    size_t mem_total = mem_matrices + mem_workspace;
    
    std::cout << "Standalone GEMM Benchmark" << std::endl;
    std::cout << "=========================" << std::endl;
    std::cout << "C(" << M << "x" << N << ") = A(" << M << "x" << K << ") * B(" << K << "x" << N << ")" << std::endl;
    std::cout << "Condition number: 10^" << LOG10_COND << std::endl;
    std::cout << "Warmup: " << NUM_WARMUP << ", Iterations: " << NUM_ITERATIONS << std::endl;
    std::cout << std::endl;
    std::cout << "Memory:" << std::endl;
    std::cout << "  Matrices (A+B+C): " << std::fixed << std::setprecision(2) << mem_matrices / (1024.0*1024.0) << " MB" << std::endl;
    std::cout << "  Ozaki workspace:  " << mem_workspace / (1024.0*1024.0) << " MB" << std::endl;
    std::cout << "  Total:            " << mem_total / (1024.0*1024.0*1024.0) << " GB" << std::endl;
    std::cout << std::endl;
    
    // Initialize handles
    rocblas_handle rb_handle;
    hipblasHandle_t hb_handle;
    hiprandGenerator_t gen;
    
    ROCBLAS_CHECK(rocblas_create_handle(&rb_handle));
    HIPBLAS_CHECK(hipblasCreate(&hb_handle));
    HIPRAND_CHECK(hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT));
    HIPRAND_CHECK(hiprandSetPseudoRandomGeneratorSeed(gen, 12345ULL));
    
    // Allocate device memory for matrices first
    double *d_A, *d_B, *d_C;
    void* d_work = nullptr;
    size_t worksize = gemmul8::workSize(M, N, K, NUM_MODULI);
    
    HIP_CHECK(hipMalloc(&d_A, size_A * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_B, size_B * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_C, size_C * sizeof(double)));
    
    // Generate conditioned matrices BEFORE allocating workspace
    // (matrix generation needs temporary memory for QR decomposition)
    std::cout << "Generating matrices..." << std::flush;
    generate_conditioned_matrix(rb_handle, gen, M, K, LOG10_COND, d_A);
    generate_conditioned_matrix(rb_handle, gen, K, N, LOG10_COND, d_B);
    std::cout << " done" << std::endl;
    
    // Now allocate Ozaki workspace
    std::cout << "Allocating Ozaki workspace..." << std::flush;
    HIP_CHECK(hipMalloc(&d_work, worksize));
    std::cout << " done" << std::endl;
    
    double alpha = 1.0, beta = 0.0;
    
    // Zero C
    int block_size = 256;
    size_t num_blocks = (size_C + block_size - 1) / block_size;
    hipLaunchKernelGGL(zero_matrix_kernel, dim3(num_blocks), dim3(block_size), 0, 0, d_C, size_C);
    HIP_CHECK(hipDeviceSynchronize());
    
    // ========== Native FP64 GEMM ==========
    std::cout << "Running Native FP64 GEMM..." << std::flush;
    
    // Warmup
    for (int i = 0; i < NUM_WARMUP; i++) {
        HIPBLAS_CHECK(hipblasDgemm(hb_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                   M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M));
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        HIPBLAS_CHECK(hipblasDgemm(hb_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                   M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M));
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double native_time = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
    double native_tflops = (2.0 * M * N * K) / (native_time * 1e12);
    std::cout << " " << std::fixed << std::setprecision(2) << native_tflops << " TFLOPS" << std::endl;
    
    // Save native result for comparison
    std::vector<double> h_C_native(size_C);
    HIP_CHECK(hipMemcpy(h_C_native.data(), d_C, size_C * sizeof(double), hipMemcpyDeviceToHost));
    
    // Zero C again
    hipLaunchKernelGGL(zero_matrix_kernel, dim3(num_blocks), dim3(block_size), 0, 0, d_C, size_C);
    HIP_CHECK(hipDeviceSynchronize());
    
    // ========== Ozaki-II GEMM (12 moduli) ==========
    std::cout << "Running Ozaki-II GEMM (" << NUM_MODULI << " moduli)..." << std::flush;
    
    // Warmup
    for (int i = 0; i < NUM_WARMUP; i++) {
        gemmul8::gemm<double>(hb_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                              M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M,
                              NUM_MODULI, false, d_work);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Timed runs
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        gemmul8::gemm<double>(hb_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                              M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M,
                              NUM_MODULI, false, d_work);
    }
    HIP_CHECK(hipDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    
    double ozaki_time = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
    double ozaki_tflops = (2.0 * M * N * K) / (ozaki_time * 1e12);
    std::cout << " " << ozaki_tflops << " TFLOPS" << std::endl;
    
    // Compute error
    std::vector<double> h_C_ozaki(size_C);
    HIP_CHECK(hipMemcpy(h_C_ozaki.data(), d_C, size_C * sizeof(double), hipMemcpyDeviceToHost));
    double rel_error = compute_frobenius_rel_error(h_C_native.data(), h_C_ozaki.data(), size_C);
    
    // Results
    std::cout << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << "  Native FP64:  " << native_tflops << " TFLOPS" << std::endl;
    std::cout << "  Ozaki-II:     " << ozaki_tflops << " TFLOPS" << std::endl;
    std::cout << "  Speedup:      " << (ozaki_tflops / native_tflops) << "x" << std::endl;
    std::cout << "  Rel. Error:   " << std::scientific << rel_error << std::endl;
    
    // Cleanup
    HIP_CHECK(hipFree(d_work));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    HIPRAND_CHECK(hiprandDestroyGenerator(gen));
    HIPBLAS_CHECK(hipblasDestroy(hb_handle));
    ROCBLAS_CHECK(rocblas_destroy_handle(rb_handle));
    
    return 0;
}
