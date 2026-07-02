/**
 * Table Benchmark with Power Measurement Support
 * 
 * Same as table_benchmark but outputs precise timestamps for GEMM phases
 * to enable power/energy measurement of GEMM operations only (not matrix prep).
 * 
 * Timestamp markers output to stderr:
 *   GEMM_PHASE_START <phase_name> <size> <unix_timestamp_microseconds>
 *   GEMM_PHASE_END <phase_name> <size> <unix_timestamp_microseconds>
 * 
 * Usage:
 *   ./table_benchmark_power <size>
 *   ./table_benchmark_power <size1> <size2> <size3> ...
 */

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <hiprand/hiprand.h>
#include "gemmul8.hpp"
#include <cstdlib>

namespace oz2 { bool g_profiling_enabled = false; }

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include <sys/time.h>

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

static const int NUM_WARMUP = 10;
static const int NUM_ITERATIONS = 50;
static const int LOG10_COND = 8;

static int64_t get_timestamp_us() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (int64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

static void emit_phase_start(const char* phase, size_t size) {
    HIP_CHECK(hipDeviceSynchronize());
    int64_t ts = get_timestamp_us();
    std::cerr << "GEMM_PHASE_START " << phase << " " << size << " " << ts << std::endl;
}

static void emit_phase_end(const char* phase, size_t size) {
    HIP_CHECK(hipDeviceSynchronize());
    int64_t ts = get_timestamp_us();
    std::cerr << "GEMM_PHASE_END " << phase << " " << size << " " << ts << std::endl;
}

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
    HIP_CHECK(hipMalloc(&d_U, size * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_V, size * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_S, size * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_temp, size * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_tau, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_sv, n * sizeof(double)));
    
    HIPRAND_CHECK(hiprandGenerateNormalDouble(gen, d_U, size, 0.0, 1.0));
    HIPRAND_CHECK(hiprandGenerateNormalDouble(gen, d_V, size, 0.0, 1.0));
    
    rocsolver_dgeqrf(rb_handle, n, n, d_U, n, d_tau);
    rocsolver_dorgqr(rb_handle, n, n, n, d_U, n, d_tau);
    
    rocsolver_dgeqrf(rb_handle, n, n, d_V, n, d_tau);
    rocsolver_dorgqr(rb_handle, n, n, n, d_V, n, d_tau);
    
    int block_size = 256;
    size_t num_blocks = (size + block_size - 1) / block_size;
    hipLaunchKernelGGL(zero_matrix_kernel, dim3(num_blocks), dim3(block_size), 0, 0, d_S, size);
    
    num_blocks = (n + block_size - 1) / block_size;
    hipLaunchKernelGGL(compute_singular_values_kernel, dim3(num_blocks), dim3(block_size), 
                       0, 0, d_sv, n, (double)log10_cond);
    hipLaunchKernelGGL(set_diagonal_kernel, dim3(num_blocks), dim3(block_size), 
                       0, 0, d_S, n, n, d_sv);
    HIP_CHECK(hipDeviceSynchronize());
    
    double alpha = 1.0, beta = 0.0;
    ROCBLAS_CHECK(rocblas_dgemm(rb_handle, rocblas_operation_none, rocblas_operation_none,
                                n, n, n, &alpha, d_U, n, d_S, n, &beta, d_temp, n));
    ROCBLAS_CHECK(rocblas_dgemm(rb_handle, rocblas_operation_none, rocblas_operation_transpose,
                                n, n, n, &alpha, d_temp, n, d_V, n, &beta, d_A, n));
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
    double native_tflops;
    double ozaki12_tflops;
    double ozaki12_error;
    double ozaki16_tflops;
    double ozaki16_error;
};

BenchmarkResult run_benchmark(size_t n) {
    BenchmarkResult result = {0, 0, 0, 0, 0};
    
    size_t size = n * n;
    size_t bytes = size * sizeof(double);
    
    rocblas_handle rb_handle;
    hipblasHandle_t hb_handle;
    hiprandGenerator_t gen;
    
    ROCBLAS_CHECK(rocblas_create_handle(&rb_handle));
    HIPBLAS_CHECK(hipblasCreate(&hb_handle));
    HIPRAND_CHECK(hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT));
    HIPRAND_CHECK(hiprandSetPseudoRandomGeneratorSeed(gen, 12345ULL));
    
    double *d_A, *d_C_native, *d_C_ozaki;
    HIP_CHECK(hipMalloc(&d_A, bytes));
    HIP_CHECK(hipMalloc(&d_C_native, bytes));
    HIP_CHECK(hipMalloc(&d_C_ozaki, bytes));
    
    std::cerr << "  Generating " << n << "x" << n << " matrix (cond=10^" << LOG10_COND << ")..." << std::flush;
    generate_conditioned_matrix(rb_handle, gen, n, LOG10_COND, d_A);
    std::cerr << " done" << std::endl;
    
    double alpha = 1.0, beta = 0.0;
    
    // === Native GEMM benchmark ===
    std::cerr << "  Benchmarking native GEMM..." << std::flush;
    
    // Warmup (not measured for power)
    for (int i = 0; i < NUM_WARMUP; i++) {
        HIPBLAS_CHECK(hipblasDgemm(hb_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                   n, n, n, &alpha, d_A, n, d_A, n, &beta, d_C_native, n));
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Timed iterations with power markers
    emit_phase_start("native_gemm", n);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        HIPBLAS_CHECK(hipblasDgemm(hb_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                   n, n, n, &alpha, d_A, n, d_A, n, &beta, d_C_native, n));
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    emit_phase_end("native_gemm", n);
    
    double native_time = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
    result.native_tflops = (2.0 * n * n * n) / (native_time * 1e12);
    std::cerr << " " << result.native_tflops << " TFLOPS" << std::endl;
    
    std::vector<double> h_C_native(size), h_C_ozaki(size);
    HIP_CHECK(hipMemcpy(h_C_native.data(), d_C_native, bytes, hipMemcpyDeviceToHost));
    
    // Ozaki-II workspace
    size_t worksize = gemmul8::workSize<false>(n, n, n, 16);
    void* d_work;
    HIP_CHECK(hipMalloc(&d_work, worksize));
    
    // === Ozaki-II 12 moduli benchmark ===
    std::cerr << "  Benchmarking Ozaki-II (12 moduli)..." << std::flush;
    
    for (int i = 0; i < NUM_WARMUP; i++) {
        gemmul8::gemm<double>(hb_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                              n, n, n, &alpha, d_A, n, d_A, n, &beta, d_C_ozaki, n,
                              12, false, d_work);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    emit_phase_start("ozaki12", n);
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        gemmul8::gemm<double>(hb_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                              n, n, n, &alpha, d_A, n, d_A, n, &beta, d_C_ozaki, n,
                              12, false, d_work);
    }
    HIP_CHECK(hipDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    emit_phase_end("ozaki12", n);
    
    double ozaki12_time = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
    result.ozaki12_tflops = (2.0 * n * n * n) / (ozaki12_time * 1e12);
    
    HIP_CHECK(hipMemcpy(h_C_ozaki.data(), d_C_ozaki, bytes, hipMemcpyDeviceToHost));
    result.ozaki12_error = compute_frobenius_rel_error(h_C_native.data(), h_C_ozaki.data(), size);
    std::cerr << " " << result.ozaki12_tflops << " TFLOPS, error=" << result.ozaki12_error << std::endl;
    
    // === Ozaki-II 16 moduli benchmark ===
    std::cerr << "  Benchmarking Ozaki-II (16 moduli)..." << std::flush;
    
    for (int i = 0; i < NUM_WARMUP; i++) {
        gemmul8::gemm<double>(hb_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                              n, n, n, &alpha, d_A, n, d_A, n, &beta, d_C_ozaki, n,
                              16, false, d_work);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    emit_phase_start("ozaki16", n);
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        gemmul8::gemm<double>(hb_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                              n, n, n, &alpha, d_A, n, d_A, n, &beta, d_C_ozaki, n,
                              16, false, d_work);
    }
    HIP_CHECK(hipDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    emit_phase_end("ozaki16", n);
    
    double ozaki16_time = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
    result.ozaki16_tflops = (2.0 * n * n * n) / (ozaki16_time * 1e12);
    
    HIP_CHECK(hipMemcpy(h_C_ozaki.data(), d_C_ozaki, bytes, hipMemcpyDeviceToHost));
    result.ozaki16_error = compute_frobenius_rel_error(h_C_native.data(), h_C_ozaki.data(), size);
    std::cerr << " " << result.ozaki16_tflops << " TFLOPS, error=" << result.ozaki16_error << std::endl;
    
    // Cleanup
    HIP_CHECK(hipFree(d_work));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_C_native));
    HIP_CHECK(hipFree(d_C_ozaki));
    HIPRAND_CHECK(hiprandDestroyGenerator(gen));
    HIPBLAS_CHECK(hipblasDestroy(hb_handle));
    ROCBLAS_CHECK(rocblas_destroy_handle(rb_handle));
    
    return result;
}

void print_table_header() {
    std::cout << "+-------------+----------+----------+----------+-------------+-----------------------------+-----------------------------+" << std::endl;
    std::cout << "| Matrix Size | Matrices | WS 12spl | WS 16spl | Native GEMM | Ozaki II (12 splits)        | Ozaki II (16 splits)        |" << std::endl;
    std::cout << "+-------------+----------+----------+----------+-------------+-----------------------------+-----------------------------+" << std::endl;
    std::cout << "|             |   (MB)   |   (MB)   |   (MB)   | Performance | Performance | Accuracy      | Performance | Accuracy      |" << std::endl;
    std::cout << "+-------------+----------+----------+----------+-------------+-------------+---------------+-------------+---------------+" << std::endl;
}

void print_table_row(size_t n, const BenchmarkResult& r) {
    size_t matrix_mem = 2 * n * n * sizeof(double);
    size_t ozaki_ws_12 = gemmul8::workSize<false>(n, n, n, 12);
    size_t ozaki_ws_16 = gemmul8::workSize<false>(n, n, n, 16);
    double matrix_mb = matrix_mem / (1024.0 * 1024.0);
    double ozaki_mb_12 = ozaki_ws_12 / (1024.0 * 1024.0);
    double ozaki_mb_16 = ozaki_ws_16 / (1024.0 * 1024.0);
    
    std::cout << "| " << std::setw(11) << n 
              << " | " << std::setw(8) << std::fixed << std::setprecision(1) << matrix_mb
              << " | " << std::setw(8) << std::setprecision(1) << ozaki_mb_12
              << " | " << std::setw(8) << std::setprecision(1) << ozaki_mb_16
              << " | " << std::setw(8) << std::setprecision(2) << r.native_tflops << " TF"
              << " | " << std::setw(8) << std::setprecision(2) << r.ozaki12_tflops << " TF"
              << " | " << std::scientific << std::setprecision(2) << r.ozaki12_error
              << " | " << std::fixed << std::setw(8) << std::setprecision(2) << r.ozaki16_tflops << " TF"
              << " | " << std::scientific << std::setprecision(2) << r.ozaki16_error
              << " |" << std::endl;
}

void print_table_footer() {
    std::cout << "+-------------+----------+----------+----------+-------------+-------------+---------------+-------------+---------------+" << std::endl;
}

void print_summary_table(const std::vector<size_t>& sizes, const std::vector<BenchmarkResult>& results) {
    std::cout << std::endl;
    std::cout << "======================================================================================================================" << std::endl;
    std::cout << "                                                  SUMMARY TABLE" << std::endl;
    std::cout << "                                             Condition Number: 10^" << LOG10_COND << std::endl;
    std::cout << "======================================================================================================================" << std::endl;
    std::cout << std::endl;
    
    print_table_header();
    for (size_t i = 0; i < sizes.size(); i++) {
        print_table_row(sizes[i], results[i]);
    }
    print_table_footer();
    
    std::cout << std::endl;
    std::cout << "Legend:" << std::endl;
    std::cout << "  - Matrices: Memory for A and C matrices (A*A computation)" << std::endl;
    std::cout << "  - WS 12spl: Ozaki-II workspace memory for 12 splits" << std::endl;
    std::cout << "  - WS 16spl: Ozaki-II workspace memory for 16 splits" << std::endl;
    std::cout << "  - Performance in TFLOPS (TF)" << std::endl;
    std::cout << "  - Accuracy: Relative Frobenius error vs native FP64" << std::endl;
    std::cout << "======================================================================================================================" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <size1> [size2] [size3] ..." << std::endl;
        std::cerr << "Example: " << argv[0] << " 1024 2048 4096 8192" << std::endl;
        return 1;
    }
    
    const char* prof = getenv("GEMMUL8_PROFILE");
    if (prof && std::string(prof) == "1") {
        oz2::g_profiling_enabled = true;
        std::cerr << "[GEMMUL8] Profiling enabled" << std::endl;
    }
    
    std::vector<size_t> sizes;
    for (int i = 1; i < argc; i++) {
        sizes.push_back(std::stoull(argv[i]));
    }
    
    std::cerr << "Condition number: 10^" << LOG10_COND << std::endl;
    std::cerr << "Matrix sizes: ";
    for (size_t s : sizes) std::cerr << s << " ";
    std::cerr << std::endl << std::endl;
    
    std::vector<BenchmarkResult> results;
    
    for (size_t n : sizes) {
        std::cerr << "Processing " << n << "x" << n << "..." << std::endl;
        BenchmarkResult result = run_benchmark(n);
        results.push_back(result);
        std::cerr << std::endl;
    }
    
    print_summary_table(sizes, results);
    
    return 0;
}
