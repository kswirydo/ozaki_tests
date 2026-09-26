/**
 * Inner Dimension Benchmark: Vary inner dimension K while keeping outer dimension N fixed
 * 
 * Tests: A(N x K) * B(K x N) = C(N x N)
 * K starts at N and decreases by 128 until K = 128
 * 
 * Outputs a table with:
 * - Dimensions (N x K x N)
 * - Native GEMM performance (TFLOPS)
 * - Ozaki II (12 moduli): Performance (TFLOPS) and Accuracy (relative error)
 * - Ozaki II (16 moduli): Performance (TFLOPS) and Accuracy (relative error)
 * 
 * Usage:
 *   ./inner_dim_benchmark <N>
 *   ./inner_dim_benchmark 4096
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

static const int NUM_WARMUP = 2;
static const int NUM_ITERATIONS = 50;
static const int LOG10_COND = 8;  // Condition number 10^8
static const size_t K_STEP = 128;
static const size_t K_MIN = 128;

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
 * where U is rows x min(rows,cols), S is diagonal, V is cols x min(rows,cols)
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
    // Step 1: temp = U * S  (rows x min_dim) * (min_dim x min_dim) = (rows x min_dim)
    double alpha = 1.0, beta = 0.0;
    ROCBLAS_CHECK(rocblas_dgemm(rb_handle, rocblas_operation_none, rocblas_operation_none,
                                rows, min_dim, min_dim, &alpha, d_U, rows, d_S, min_dim, &beta, d_temp, rows));
    // Step 2: A = temp * V^T  (rows x min_dim) * (min_dim x cols) = (rows x cols)
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

struct BenchmarkResult {
    size_t N;
    size_t K;
    double native_tflops;
    double ozaki12_tflops;
    double ozaki12_error;
    double ozaki16_tflops;
    double ozaki16_error;
};

// Returns true if successful, false if out of memory
bool run_benchmark(size_t N, size_t K, 
                   rocblas_handle rb_handle, 
                   hipblasHandle_t hb_handle,
                   hiprandGenerator_t gen,
                   BenchmarkResult& result) {
    result = {N, K, 0, 0, 0, 0, 0};
    
    size_t size_A = N * K;  // A is N x K
    size_t size_B = K * N;  // B is K x N
    size_t size_C = N * N;  // C is N x N
    
    // Calculate total memory needed
    size_t mem_matrices = (size_A + size_B + 2 * size_C) * sizeof(double);
    size_t mem_workspace = gemmul8::workSize<false, gemmul8::Backend::INT8>(N, N, K, 16);
    size_t mem_total = mem_matrices + mem_workspace;
    
    // Check available memory first
    size_t free_mem, total_mem;
    hipMemGetInfo(&free_mem, &total_mem);
    
    if (mem_total > free_mem * 0.9) {  // Leave 10% margin
        std::cerr << "  WARNING: Not enough GPU memory!" << std::endl;
        std::cerr << "    Required: " << mem_total / (1024.0*1024.0*1024.0) << " GB" << std::endl;
        std::cerr << "    Available: " << free_mem / (1024.0*1024.0*1024.0) << " GB" << std::endl;
        return false;
    }
    
    // Allocate device memory with error checking
    double *d_A = nullptr, *d_B = nullptr, *d_C_native = nullptr, *d_C_ozaki = nullptr;
    void* d_work = nullptr;
    
    hipError_t err;
    err = hipMalloc(&d_A, size_A * sizeof(double));
    if (err != hipSuccess) {
        std::cerr << "  WARNING: Failed to allocate d_A: " << hipGetErrorString(err) << std::endl;
        return false;
    }
    
    err = hipMalloc(&d_B, size_B * sizeof(double));
    if (err != hipSuccess) {
        std::cerr << "  WARNING: Failed to allocate d_B: " << hipGetErrorString(err) << std::endl;
        hipFree(d_A);
        return false;
    }
    
    err = hipMalloc(&d_C_native, size_C * sizeof(double));
    if (err != hipSuccess) {
        std::cerr << "  WARNING: Failed to allocate d_C_native: " << hipGetErrorString(err) << std::endl;
        hipFree(d_A); hipFree(d_B);
        return false;
    }
    
    err = hipMalloc(&d_C_ozaki, size_C * sizeof(double));
    if (err != hipSuccess) {
        std::cerr << "  WARNING: Failed to allocate d_C_ozaki: " << hipGetErrorString(err) << std::endl;
        hipFree(d_A); hipFree(d_B); hipFree(d_C_native);
        return false;
    }
    
    // Generate conditioned matrices A (NxK) and B (KxN)
    std::cerr << "  Generating matrices A(" << N << "x" << K << ") and B(" << K << "x" << N << ")..." << std::flush;
    generate_conditioned_matrix(rb_handle, gen, N, K, LOG10_COND, d_A);
    generate_conditioned_matrix(rb_handle, gen, K, N, LOG10_COND, d_B);
    std::cerr << " done" << std::endl;
    
    double alpha = 1.0, beta = 0.0;
    
    // Native GEMM benchmark: C = A * B
    std::cerr << "  Benchmarking native GEMM..." << std::flush;
    for (int i = 0; i < NUM_WARMUP; i++) {
        HIPBLAS_CHECK(hipblasDgemm(hb_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                   N, N, K, &alpha, d_A, N, d_B, K, &beta, d_C_native, N));
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        HIPBLAS_CHECK(hipblasDgemm(hb_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                   N, N, K, &alpha, d_A, N, d_B, K, &beta, d_C_native, N));
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double native_time = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
    result.native_tflops = (2.0 * N * N * K) / (native_time * 1e12);
    std::cerr << " " << result.native_tflops << " TFLOPS" << std::endl;
    
    // Copy native result to host for comparison
    std::vector<double> h_C_native(size_C), h_C_ozaki(size_C);
    HIP_CHECK(hipMemcpy(h_C_native.data(), d_C_native, size_C * sizeof(double), hipMemcpyDeviceToHost));
    
    // Ozaki-II benchmarks
    size_t worksize = gemmul8::workSize<false, gemmul8::Backend::INT8>(N, N, K, 16);
    err = hipMalloc(&d_work, worksize);
    if (err != hipSuccess) {
        std::cerr << "  WARNING: Failed to allocate workspace: " << hipGetErrorString(err) << std::endl;
        hipFree(d_A); hipFree(d_B); hipFree(d_C_native); hipFree(d_C_ozaki);
        return false;
    }
    
    for (int moduli : {12, 16}) {
        std::cerr << "  Benchmarking Ozaki-II (" << moduli << " moduli)..." << std::flush;
        
        for (int i = 0; i < NUM_WARMUP; i++) {
            gemmul8::gemm<double>(hb_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                  N, N, K, &alpha, d_A, N, d_B, K, &beta, d_C_ozaki, N,
                                  moduli, false, d_work);
        }
        HIP_CHECK(hipDeviceSynchronize());
        
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            gemmul8::gemm<double>(hb_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                  N, N, K, &alpha, d_A, N, d_B, K, &beta, d_C_ozaki, N,
                                  moduli, false, d_work);
        }
        HIP_CHECK(hipDeviceSynchronize());
        end = std::chrono::high_resolution_clock::now();
        
        double ozaki_time = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
        double tflops = (2.0 * N * N * K) / (ozaki_time * 1e12);
        
        HIP_CHECK(hipMemcpy(h_C_ozaki.data(), d_C_ozaki, size_C * sizeof(double), hipMemcpyDeviceToHost));
        double error = compute_frobenius_rel_error(h_C_native.data(), h_C_ozaki.data(), size_C);
        
        if (moduli == 12) {
            result.ozaki12_tflops = tflops;
            result.ozaki12_error = error;
        } else {
            result.ozaki16_tflops = tflops;
            result.ozaki16_error = error;
        }
        std::cerr << " " << tflops << " TFLOPS, error=" << error << std::endl;
    }
    
    // Cleanup
    HIP_CHECK(hipFree(d_work));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C_native));
    HIP_CHECK(hipFree(d_C_ozaki));
    
    return true;
}

void print_table_header() {
    std::cout << "+-----------------------+-------------+-----------------------------+-----------------------------+" << std::endl;
    std::cout << "| Dimensions (NxKxN)    | Native GEMM | Ozaki II (12 splits)        | Ozaki II (16 splits)        |" << std::endl;
    std::cout << "+-----------------------+-------------+-----------------------------+-----------------------------+" << std::endl;
    std::cout << "|                       | Performance | Performance | Accuracy      | Performance | Accuracy      |" << std::endl;
    std::cout << "+-----------------------+-------------+-------------+---------------+-------------+---------------+" << std::endl;
}

void print_table_row(const BenchmarkResult& r) {
    std::ostringstream dims;
    dims << r.N << "x" << r.K << "x" << r.N;
    
    std::cout << "| " << std::setw(21) << dims.str()
              << " | " << std::setw(8) << std::fixed << std::setprecision(2) << r.native_tflops << " TF"
              << " | " << std::setw(8) << std::setprecision(2) << r.ozaki12_tflops << " TF"
              << " | " << std::scientific << std::setprecision(2) << r.ozaki12_error
              << " | " << std::fixed << std::setw(8) << std::setprecision(2) << r.ozaki16_tflops << " TF"
              << " | " << std::scientific << std::setprecision(2) << r.ozaki16_error
              << " |" << std::endl;
}

void print_table_footer() {
    std::cout << "+-----------------------+-------------+-------------+---------------+-------------+---------------+" << std::endl;
}

void print_summary_table(const std::vector<BenchmarkResult>& results) {
    std::cout << std::endl;
    std::cout << "================================================================================================" << std::endl;
    std::cout << "                                    SUMMARY TABLE" << std::endl;
    std::cout << "                    Inner Dimension Sweep: K from " << results.front().K 
              << " to " << results.back().K << " (step " << K_STEP << ")" << std::endl;
    std::cout << "                               Condition Number: 10^" << LOG10_COND << std::endl;
    std::cout << "================================================================================================" << std::endl;
    std::cout << std::endl;
    
    print_table_header();
    for (const auto& r : results) {
        print_table_row(r);
    }
    print_table_footer();
    
    std::cout << std::endl;
    std::cout << "Legend:" << std::endl;
    std::cout << "  - Dimensions: A(NxK) * B(KxN) = C(NxN)" << std::endl;
    std::cout << "  - Performance in TFLOPS (TF)" << std::endl;
    std::cout << "  - Accuracy: Relative Frobenius error vs native FP64" << std::endl;
    std::cout << "================================================================================================" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <N>" << std::endl;
        std::cerr << "  N = outer dimension (result is NxN)" << std::endl;
        std::cerr << "  K starts at N and decreases by " << K_STEP << " until " << K_MIN << std::endl;
        std::cerr << std::endl;
        std::cerr << "Example: " << argv[0] << " 4096" << std::endl;
        std::cerr << "  Will test: 4096x4096x4096, 4096x3968x4096, ..., 4096x128x4096" << std::endl;
        return 1;
    }
    
    size_t N = std::stoull(argv[1]);
    
    if (N < K_MIN) {
        std::cerr << "Error: N must be at least " << K_MIN << std::endl;
        return 1;
    }
    
    // Generate list of K values
    std::vector<size_t> K_values;
    for (size_t k = N; k >= K_MIN; k -= K_STEP) {
        K_values.push_back(k);
    }
    
    std::cerr << "Inner Dimension Benchmark" << std::endl;
    std::cerr << "=========================" << std::endl;
    std::cerr << "Outer dimension N: " << N << std::endl;
    std::cerr << "Condition number: 10^" << LOG10_COND << std::endl;
    std::cerr << "K values: ";
    for (size_t k : K_values) std::cerr << k << " ";
    std::cerr << std::endl << std::endl;
    
    // Initialize handles (reuse across all benchmarks)
    rocblas_handle rb_handle;
    hipblasHandle_t hb_handle;
    hiprandGenerator_t gen;
    
    ROCBLAS_CHECK(rocblas_create_handle(&rb_handle));
    HIPBLAS_CHECK(hipblasCreate(&hb_handle));
    HIPRAND_CHECK(hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT));
    HIPRAND_CHECK(hiprandSetPseudoRandomGeneratorSeed(gen, 12345ULL));
    
    // Collect all results
    std::vector<BenchmarkResult> results;
    
    bool out_of_memory = false;
    for (size_t K : K_values) {
        std::cerr << "Processing A(" << N << "x" << K << ") * B(" << K << "x" << N << ")..." << std::endl;
        BenchmarkResult result;
        if (!run_benchmark(N, K, rb_handle, hb_handle, gen, result)) {
            std::cerr << "Stopping benchmark due to memory constraints." << std::endl;
            out_of_memory = true;
            break;
        }
        results.push_back(result);
        std::cerr << std::endl;
    }
    
    if (out_of_memory && results.empty()) {
        std::cerr << "No results collected. Try a smaller N value." << std::endl;
        HIPRAND_CHECK(hiprandDestroyGenerator(gen));
        HIPBLAS_CHECK(hipblasDestroy(hb_handle));
        ROCBLAS_CHECK(rocblas_destroy_handle(rb_handle));
        return 1;
    }
    
    // Cleanup handles
    HIPRAND_CHECK(hiprandDestroyGenerator(gen));
    HIPBLAS_CHECK(hipblasDestroy(hb_handle));
    ROCBLAS_CHECK(rocblas_destroy_handle(rb_handle));
    
    // Print summary table at the end
    print_summary_table(results);
    
    return 0;
}
