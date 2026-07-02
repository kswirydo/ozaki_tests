/**
 * Single GEMM Benchmark: Test specific M, N, K dimensions
 * 
 * Generates matrices A(M x K) and B(K x N) with condition number 10^8
 * Tests: C(M x N) = A * B
 * Compares Native FP64 GEMM vs Ozaki-II (12 and 16 splits)
 * 
 * Usage:
 *   ./single_gemm_benchmark <M> <N> <K>
 *   ./single_gemm_benchmark 4096 4096 1024
 */

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <hiprand/hiprand.h>
#include "gemmul8.hpp"

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

static const int NUM_WARMUP = 40;
static const int NUM_ITERATIONS = 50;
static const int LOG10_COND = 8;  // Condition number 10^8

// Kernels for matrix generation
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

/**
 * Generate a conditioned matrix of size rows x cols
 * Uses SVD-like construction: A = U * S * V^T
 */
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
    
    // Generate random U (rows x min_dim) and V (cols x min_dim)
    HIPRAND_CHECK(hiprandGenerateNormalDouble(gen, d_U, size_U, 0.0, 1.0));
    HIPRAND_CHECK(hiprandGenerateNormalDouble(gen, d_V, size_V, 0.0, 1.0));
    
    // QR decomposition to get orthogonal matrices
    rocsolver_dgeqrf(rb_handle, rows, min_dim, d_U, rows, d_tau);
    rocsolver_dorgqr(rb_handle, rows, min_dim, min_dim, d_U, rows, d_tau);
    
    rocsolver_dgeqrf(rb_handle, cols, min_dim, d_V, cols, d_tau);
    rocsolver_dorgqr(rb_handle, cols, min_dim, min_dim, d_V, cols, d_tau);
    
    // Zero S and set diagonal with singular values
    int block_size = 256;
    size_t num_blocks = (size_S + block_size - 1) / block_size;
    hipLaunchKernelGGL(zero_matrix_kernel, dim3(num_blocks), dim3(block_size), 0, 0, d_S, size_S);
    
    num_blocks = (min_dim + block_size - 1) / block_size;
    hipLaunchKernelGGL(compute_singular_values_kernel, dim3(num_blocks), dim3(block_size), 
                       0, 0, d_sv, min_dim, (double)log10_cond);
    hipLaunchKernelGGL(set_diagonal_kernel, dim3(num_blocks), dim3(block_size), 
                       0, 0, d_S, min_dim, min_dim, min_dim, d_sv);
    HIP_CHECK(hipDeviceSynchronize());
    
    // A = U * S * V^T
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

double compute_max_abs_error(const double* C_ref, const double* C_test, size_t n) {
    double max_err = 0.0;
    for (size_t i = 0; i < n; i++) {
        double err = std::abs(C_ref[i] - C_test[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <M> <N> <K>" << std::endl;
        std::cerr << "  Computes C(MxN) = A(MxK) * B(KxN)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Example: " << argv[0] << " 4096 4096 1024" << std::endl;
        return 1;
    }
    
    size_t M = std::stoull(argv[1]);
    size_t N = std::stoull(argv[2]);
    size_t K = std::stoull(argv[3]);
    
    if (M < 128 || N < 128 || K < 128) {
        std::cerr << "Error: All dimensions must be at least 128" << std::endl;
        return 1;
    }
    
    size_t size_A = M * K;
    size_t size_B = K * N;
    size_t size_C = M * N;
    
    // Memory requirements
    size_t mem_matrices = (size_A + size_B + 2 * size_C) * sizeof(double);
    size_t mem_workspace_12 = gemmul8::workSize<false>(M, N, K, 12);
    size_t mem_workspace_16 = gemmul8::workSize<false>(M, N, K, 16);
    size_t mem_total = mem_matrices + mem_workspace_16;
    
    std::cout << "=========================================" << std::endl;
    std::cout << "       Single GEMM Benchmark" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  C(" << M << " x " << N << ") = A(" << M << " x " << K << ") * B(" << K << " x " << N << ")" << std::endl;
    std::cout << "  Condition number: 10^" << LOG10_COND << std::endl;
    std::cout << "  Iterations: " << NUM_ITERATIONS << std::endl;
    std::cout << "  FLOPs per GEMM: " << std::scientific << std::setprecision(2) << (2.0 * M * N * K) << std::endl;
    std::cout << std::endl;
    std::cout << "Memory Requirements:" << std::endl;
    size_t mem_ABC = (size_A + size_B + size_C) * sizeof(double);
    double ratio_12 = (double)mem_workspace_12 / (double)mem_ABC;
    double ratio_16 = (double)mem_workspace_16 / (double)mem_ABC;
    
    std::cout << "  Matrix A: " << std::fixed << std::setprecision(2) << (size_A * sizeof(double)) / (1024.0*1024.0) << " MB" << std::endl;
    std::cout << "  Matrix B: " << (size_B * sizeof(double)) / (1024.0*1024.0) << " MB" << std::endl;
    std::cout << "  Matrix C: " << (size_C * sizeof(double)) / (1024.0*1024.0) << " MB" << std::endl;
    std::cout << "  Total A+B+C: " << mem_ABC / (1024.0*1024.0) << " MB" << std::endl;
    std::cout << "  Ozaki workspace (12 splits): " << mem_workspace_12 / (1024.0*1024.0) << " MB (ratio: " << std::setprecision(2) << ratio_12 << "x)" << std::endl;
    std::cout << "  Ozaki workspace (16 splits): " << mem_workspace_16 / (1024.0*1024.0) << " MB (ratio: " << std::setprecision(2) << ratio_16 << "x)" << std::endl;
    std::cout << "  Total (with 16 splits): " << mem_total / (1024.0*1024.0*1024.0) << " GB" << std::endl;
    std::cout << std::endl;
    
    // Check available memory
    size_t free_mem, total_mem;
    hipMemGetInfo(&free_mem, &total_mem);
    std::cout << "GPU Memory: " << free_mem / (1024.0*1024.0*1024.0) << " GB free / " 
              << total_mem / (1024.0*1024.0*1024.0) << " GB total" << std::endl;
    
    if (mem_total > free_mem * 0.9) {
        std::cerr << "Error: Not enough GPU memory!" << std::endl;
        return 1;
    }
    std::cout << std::endl;
    
    // Initialize handles
    rocblas_handle rb_handle;
    hipblasHandle_t hb_handle;
    hiprandGenerator_t gen;
    
    ROCBLAS_CHECK(rocblas_create_handle(&rb_handle));
    HIPBLAS_CHECK(hipblasCreate(&hb_handle));
    HIPRAND_CHECK(hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT));
    HIPRAND_CHECK(hiprandSetPseudoRandomGeneratorSeed(gen, 12345ULL));
    
    // Allocate device memory
    double *d_A, *d_B, *d_C_native, *d_C_ozaki;
    void* d_work;
    HIP_CHECK(hipMalloc(&d_A, size_A * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_B, size_B * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_C_native, size_C * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_C_ozaki, size_C * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_work, mem_workspace_16));
    
    // Generate conditioned matrices
    std::cout << "Generating matrices..." << std::flush;
    generate_conditioned_matrix(rb_handle, gen, M, K, LOG10_COND, d_A);
    generate_conditioned_matrix(rb_handle, gen, K, N, LOG10_COND, d_B);
    std::cout << " done" << std::endl;
    std::cout << std::endl;
    
    double alpha = 1.0, beta = 0.0;
    
    // Host buffers for error computation
    std::vector<double> h_C_native(size_C), h_C_ozaki(size_C);
    
    // Results storage
    double native_tflops = 0;
    double ozaki12_tflops = 0, ozaki12_error = 0, ozaki12_max_err = 0;
    double ozaki16_tflops = 0, ozaki16_error = 0, ozaki16_max_err = 0;
    
    // Native GEMM benchmark
    std::cout << "Running Native FP64 GEMM..." << std::flush;
    for (int i = 0; i < NUM_WARMUP; i++) {
        HIPBLAS_CHECK(hipblasDgemm(hb_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                   M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C_native, M));
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        HIPBLAS_CHECK(hipblasDgemm(hb_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                   M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C_native, M));
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double native_time = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
    native_tflops = (2.0 * M * N * K) / (native_time * 1e12);
    std::cout << " " << std::fixed << std::setprecision(2) << native_tflops << " TFLOPS" << std::endl;
    
    // Copy native result to host
    HIP_CHECK(hipMemcpy(h_C_native.data(), d_C_native, size_C * sizeof(double), hipMemcpyDeviceToHost));
    
    // Ozaki-II (12 splits) benchmark
    std::cout << "Running Ozaki-II (12 splits)..." << std::flush;
    for (int i = 0; i < NUM_WARMUP; i++) {
        gemmul8::gemm<double>(hb_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                              M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C_ozaki, M,
                              12, false, d_work);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        gemmul8::gemm<double>(hb_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                              M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C_ozaki, M,
                              12, false, d_work);
    }
    HIP_CHECK(hipDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    
    double ozaki12_time = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
    ozaki12_tflops = (2.0 * M * N * K) / (ozaki12_time * 1e12);
    
    HIP_CHECK(hipMemcpy(h_C_ozaki.data(), d_C_ozaki, size_C * sizeof(double), hipMemcpyDeviceToHost));
    ozaki12_error = compute_frobenius_rel_error(h_C_native.data(), h_C_ozaki.data(), size_C);
    ozaki12_max_err = compute_max_abs_error(h_C_native.data(), h_C_ozaki.data(), size_C);
    std::cout << " " << std::fixed << std::setprecision(2) << ozaki12_tflops << " TFLOPS" << std::endl;
    
    // Ozaki-II (16 splits) benchmark
    std::cout << "Running Ozaki-II (16 splits)..." << std::flush;
    for (int i = 0; i < NUM_WARMUP; i++) {
        gemmul8::gemm<double>(hb_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                              M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C_ozaki, M,
                              16, false, d_work);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        gemmul8::gemm<double>(hb_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                              M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C_ozaki, M,
                              16, false, d_work);
    }
    HIP_CHECK(hipDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    
    double ozaki16_time = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
    ozaki16_tflops = (2.0 * M * N * K) / (ozaki16_time * 1e12);
    
    HIP_CHECK(hipMemcpy(h_C_ozaki.data(), d_C_ozaki, size_C * sizeof(double), hipMemcpyDeviceToHost));
    ozaki16_error = compute_frobenius_rel_error(h_C_native.data(), h_C_ozaki.data(), size_C);
    ozaki16_max_err = compute_max_abs_error(h_C_native.data(), h_C_ozaki.data(), size_C);
    std::cout << " " << std::fixed << std::setprecision(2) << ozaki16_tflops << " TFLOPS" << std::endl;
    
    std::cout << std::endl;
    
    // Print results table
    std::cout << "=============================================================================" << std::endl;
    std::cout << "                                  RESULTS" << std::endl;
    std::cout << "        C(" << M << "x" << N << ") = A(" << M << "x" << K << ") * B(" << K << "x" << N << ")" << std::endl;
    std::cout << "=============================================================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "+------------------+-------+-------+-------+------------+----------------+----------------+" << std::endl;
    std::cout << "| Method           |   M   |   N   |   K   | Performance|  Rel. Frob Err |  Max Abs Err   |" << std::endl;
    std::cout << "+------------------+-------+-------+-------+------------+----------------+----------------+" << std::endl;
    std::cout << "| Native FP64      | " << std::setw(5) << M << " | " << std::setw(5) << N << " | " << std::setw(5) << K 
              << " | " << std::setw(7) << std::fixed << std::setprecision(2) << native_tflops << " TF |"
              << "    (reference) |    (reference) |" << std::endl;
    std::cout << "| Ozaki-II 12 spl  | " << std::setw(5) << M << " | " << std::setw(5) << N << " | " << std::setw(5) << K 
              << " | " << std::setw(7) << ozaki12_tflops << " TF | "
              << std::scientific << std::setprecision(2) << std::setw(12) << ozaki12_error << " | "
              << std::setw(12) << ozaki12_max_err << " |" << std::endl;
    std::cout << "| Ozaki-II 16 spl  | " << std::setw(5) << M << " | " << std::setw(5) << N << " | " << std::setw(5) << K 
              << " | " << std::setw(7) << std::fixed << std::setprecision(2) << ozaki16_tflops << " TF | "
              << std::scientific << std::setprecision(2) << std::setw(12) << ozaki16_error << " | "
              << std::setw(12) << ozaki16_max_err << " |" << std::endl;
    std::cout << "+------------------+-------+-------+-------+------------+----------------+----------------+" << std::endl;
    
    std::cout << std::endl;
    std::cout << "Speedup vs Native:" << std::endl;
    std::cout << "  Ozaki-II 12 splits: " << std::fixed << std::setprecision(2) << (ozaki12_tflops / native_tflops) << "x" << std::endl;
    std::cout << "  Ozaki-II 16 splits: " << (ozaki16_tflops / native_tflops) << "x" << std::endl;
    std::cout << std::endl;
    
    // Cleanup
    HIP_CHECK(hipFree(d_work));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C_native));
    HIP_CHECK(hipFree(d_C_ozaki));
    HIPRAND_CHECK(hiprandDestroyGenerator(gen));
    HIPBLAS_CHECK(hipblasDestroy(hb_handle));
    ROCBLAS_CHECK(rocblas_destroy_handle(rb_handle));
    
    return 0;
}
