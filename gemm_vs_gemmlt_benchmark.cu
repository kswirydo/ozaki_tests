/**
 * GEMM vs GEMMLt Benchmark (INT8 Backend)
 * ---------------------------------------
 * Compares the two INT8 Ozaki-II entry points provided by GEMMul8:
 *   - gemmul8::gemm   (legacy hipBLAS handle)
 *   - gemmul8::gemmLt (hipBLASLt handle + stream)
 *
 * For each requested matrix size and each modulus count, it runs 50 timed
 * iterations (after 10 warmup runs) of each variant and reports:
 *   - Native FP64 GEMM performance (reference)
 *   - gemm   INT8 performance (TFLOPS) and accuracy vs native
 *   - gemmLt INT8 performance (TFLOPS) and accuracy vs native
 *   - Workspace size (MB) required by each variant
 *
 * Both variants compute C = A * A on a square matrix with condition 10^8.
 *
 * Usage:
 *   ./gemm_vs_gemmlt_benchmark <size1> [size2] [size3] ...
 *   Example: ./gemm_vs_gemmlt_benchmark 1024 2048 4096 8192
 */

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
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

#define HIPBLASLT_CHECK(call) do { \
    hipblasStatus_t status = call; \
    if (status != HIPBLAS_STATUS_SUCCESS) { \
        std::cerr << "hipBLASLt error: " << status << std::endl; \
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
static const bool FASTMODE = false;
static const std::vector<int> MODULI = {12, 16};

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

// Per-modulus comparison data.
struct ModuliResult {
    int moduli;
    double gemm_tflops;
    double gemm_error;
    double gemmLt_tflops;
    double gemmLt_error;
    size_t gemm_ws;
    size_t gemmLt_ws;
};

struct BenchmarkResult {
    double native_tflops;
    std::vector<ModuliResult> per_moduli;
};

BenchmarkResult run_benchmark(size_t n) {
    BenchmarkResult result;
    result.native_tflops = 0.0;

    size_t size = n * n;
    size_t bytes = size * sizeof(double);

    rocblas_handle rb_handle;
    hipblasHandle_t hb_handle;
    hipblasLtHandle_t hblt_handle;
    hiprandGenerator_t gen;
    hipStream_t stream = 0;

    ROCBLAS_CHECK(rocblas_create_handle(&rb_handle));
    HIPBLAS_CHECK(hipblasCreate(&hb_handle));
    HIPBLASLT_CHECK(hipblasLtCreate(&hblt_handle));
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

    // Native FP64 GEMM reference.
    std::cerr << "  Benchmarking native GEMM..." << std::flush;
    for (int i = 0; i < NUM_WARMUP; i++) {
        HIPBLAS_CHECK(hipblasDgemm(hb_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                   n, n, n, &alpha, d_A, n, d_A, n, &beta, d_C_native, n));
    }
    HIP_CHECK(hipDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        HIPBLAS_CHECK(hipblasDgemm(hb_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                   n, n, n, &alpha, d_A, n, d_A, n, &beta, d_C_native, n));
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    double native_time = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
    result.native_tflops = (2.0 * n * n * n) / (native_time * 1e12);
    std::cerr << " " << result.native_tflops << " TFLOPS" << std::endl;

    std::vector<double> h_C_native(size), h_C_ozaki(size);
    HIP_CHECK(hipMemcpy(h_C_native.data(), d_C_native, bytes, hipMemcpyDeviceToHost));

    // Allocate a workspace large enough for the largest modulus count tested.
    int max_moduli = 0;
    for (int m : MODULI) max_moduli = std::max(max_moduli, m);
    size_t worksize = gemmul8::workSize<false, gemmul8::Backend::INT8>(n, n, n, max_moduli);
    void* d_work;
    HIP_CHECK(hipMalloc(&d_work, worksize));

    for (int moduli : MODULI) {
        ModuliResult mr = {moduli, 0, 0, 0, 0, 0, 0};
        mr.gemm_ws   = gemmul8::workSize<false, gemmul8::Backend::INT8>(n, n, n, moduli);
        mr.gemmLt_ws = mr.gemm_ws; // INT8 workspace is identical for gemm and gemmLt

        // --- gemm (hipBLAS handle) ---
        std::cerr << "  Benchmarking gemm   INT8 (" << moduli << " moduli)..." << std::flush;
        for (int i = 0; i < NUM_WARMUP; i++) {
            gemmul8::gemm<double, gemmul8::Backend::INT8>(hb_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                  n, n, n, &alpha, d_A, n, d_A, n, &beta, d_C_ozaki, n,
                                  moduli, FASTMODE, d_work);
        }
        HIP_CHECK(hipDeviceSynchronize());

        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            gemmul8::gemm<double, gemmul8::Backend::INT8>(hb_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                  n, n, n, &alpha, d_A, n, d_A, n, &beta, d_C_ozaki, n,
                                  moduli, FASTMODE, d_work);
        }
        HIP_CHECK(hipDeviceSynchronize());
        end = std::chrono::high_resolution_clock::now();

        double gemm_time = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
        mr.gemm_tflops = (2.0 * n * n * n) / (gemm_time * 1e12);
        HIP_CHECK(hipMemcpy(h_C_ozaki.data(), d_C_ozaki, bytes, hipMemcpyDeviceToHost));
        mr.gemm_error = compute_frobenius_rel_error(h_C_native.data(), h_C_ozaki.data(), size);
        std::cerr << " " << mr.gemm_tflops << " TFLOPS, error=" << mr.gemm_error << std::endl;

        // --- gemmLt (hipBLASLt handle) ---
        std::cerr << "  Benchmarking gemmLt INT8 (" << moduli << " moduli)..." << std::flush;
        for (int i = 0; i < NUM_WARMUP; i++) {
            gemmul8::gemmLt<double, gemmul8::Backend::INT8>(hblt_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                  n, n, n, &alpha, d_A, n, d_A, n, &beta, d_C_ozaki, n,
                                  moduli, FASTMODE, d_work,
                                  nullptr, nullptr, false, false, false, false, stream);
        }
        HIP_CHECK(hipDeviceSynchronize());

        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            gemmul8::gemmLt<double, gemmul8::Backend::INT8>(hblt_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                  n, n, n, &alpha, d_A, n, d_A, n, &beta, d_C_ozaki, n,
                                  moduli, FASTMODE, d_work,
                                  nullptr, nullptr, false, false, false, false, stream);
        }
        HIP_CHECK(hipDeviceSynchronize());
        end = std::chrono::high_resolution_clock::now();

        double gemmLt_time = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
        mr.gemmLt_tflops = (2.0 * n * n * n) / (gemmLt_time * 1e12);
        HIP_CHECK(hipMemcpy(h_C_ozaki.data(), d_C_ozaki, bytes, hipMemcpyDeviceToHost));
        mr.gemmLt_error = compute_frobenius_rel_error(h_C_native.data(), h_C_ozaki.data(), size);
        std::cerr << " " << mr.gemmLt_tflops << " TFLOPS, error=" << mr.gemmLt_error << std::endl;

        result.per_moduli.push_back(mr);
    }

    HIP_CHECK(hipFree(d_work));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_C_native));
    HIP_CHECK(hipFree(d_C_ozaki));
    HIPRAND_CHECK(hiprandDestroyGenerator(gen));
    HIPBLASLT_CHECK(hipblasLtDestroy(hblt_handle));
    HIPBLAS_CHECK(hipblasDestroy(hb_handle));
    ROCBLAS_CHECK(rocblas_destroy_handle(rb_handle));

    return result;
}

void print_summary(const std::vector<size_t>& sizes, const std::vector<BenchmarkResult>& results) {
    // One row per matrix size, with the per-moduli gemm/gemmLt TFLOPS side by side.
    // Assumes the same MODULI list was used for every size.
    const std::vector<int>& mods = MODULI;

    std::cout << std::endl;
    std::cout << "INT8 gemm vs gemmLt  (" << NUM_ITERATIONS << " runs each, cond=10^" << LOG10_COND << ")" << std::endl;
    std::cout << "All values in TFLOPS unless noted." << std::endl;
    std::cout << std::endl;

    // Build separator and headers dynamically based on number of moduli.
    auto sep = [&]() {
        std::cout << "+--------+---------+";
        for (size_t m = 0; m < mods.size(); m++) std::cout << "---------+---------+";
        for (size_t m = 0; m < mods.size(); m++) std::cout << "----------+";
        std::cout << std::endl;
    };

    // Top header: grouped labels.
    sep();
    std::cout << "| Size   | Native  |";
    for (int mod : mods) {
        std::cout << " " << std::setw(7) << ("mod=" + std::to_string(mod)) << " "
                  << "|" << std::setw(9) << " " << "|";
    }
    for (int mod : mods) std::cout << std::setw(10) << ("WS m" + std::to_string(mod)) << "|";
    std::cout << std::endl;

    // Sub header: gemm / gemmLt under each moduli group.
    std::cout << "|        | FP64    |";
    for (size_t m = 0; m < mods.size(); m++) std::cout << "  gemm   | gemmLt  |";
    for (size_t m = 0; m < mods.size(); m++) std::cout << "   (MB)   |";
    std::cout << std::endl;
    sep();

    for (size_t i = 0; i < sizes.size(); i++) {
        const BenchmarkResult& r = results[i];
        std::cout << "| " << std::setw(6) << sizes[i]
                  << " | " << std::setw(7) << std::fixed << std::setprecision(2) << r.native_tflops
                  << " |";
        for (const ModuliResult& mr : r.per_moduli) {
            std::cout << " " << std::setw(7) << std::fixed << std::setprecision(2) << mr.gemm_tflops
                      << " | " << std::setw(7) << std::setprecision(2) << mr.gemmLt_tflops
                      << " |";
        }
        for (const ModuliResult& mr : r.per_moduli) {
            double gemm_mb = mr.gemm_ws / (1024.0 * 1024.0);
            std::cout << " " << std::setw(8) << std::setprecision(1) << gemm_mb << " |";
        }
        std::cout << std::endl;
    }
    sep();

    std::cout << std::endl;
    std::cout << "Legend:" << std::endl;
    std::cout << "  - Native FP64: hipBLAS FP64 reference TFLOPS" << std::endl;
    std::cout << "  - mod=N gemm / gemmLt: INT8 Ozaki-II TFLOPS with N moduli via hipBLAS / hipBLASLt handle" << std::endl;
    std::cout << "  - WS mN (MB): GEMMul8 workspace size in MB for N moduli (identical for gemm and gemmLt with INT8)" << std::endl;
    std::cout << std::endl;

    // Accuracy reported separately (identical for gemm and gemmLt, so listed once).
    std::cout << "Accuracy (relative Frobenius error vs native FP64):" << std::endl;
    auto acc_sep = [&]() {
        std::cout << "+--------+";
        for (size_t m = 0; m < mods.size(); m++) std::cout << "----------+";
        std::cout << std::endl;
    };
    acc_sep();
    std::cout << "| Size   |";
    for (int mod : mods) std::cout << " " << std::setw(8) << ("mod=" + std::to_string(mod)) << " |";
    std::cout << std::endl;
    acc_sep();
    for (size_t i = 0; i < sizes.size(); i++) {
        std::cout << "| " << std::setw(6) << sizes[i] << " |";
        for (const ModuliResult& mr : results[i].per_moduli) {
            std::cout << " " << std::setw(8) << std::scientific << std::setprecision(2) << mr.gemmLt_error << " |";
        }
        std::cout << std::endl;
    }
    acc_sep();
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

    std::cerr << "Backend: INT8 (comparing gemm vs gemmLt)" << std::endl;
    std::cerr << "Condition number: 10^" << LOG10_COND << std::endl;
    std::cerr << "Iterations: " << NUM_ITERATIONS << " (warmup " << NUM_WARMUP << ")" << std::endl;
    std::cerr << "Moduli: ";
    for (int m : MODULI) std::cerr << m << " ";
    std::cerr << std::endl << "Matrix sizes: ";
    for (size_t s : sizes) std::cerr << s << " ";
    std::cerr << std::endl << std::endl;

    std::vector<BenchmarkResult> results;
    for (size_t n : sizes) {
        std::cerr << "Processing " << n << "x" << n << "..." << std::endl;
        results.push_back(run_benchmark(n));
        std::cerr << std::endl;
    }

    print_summary(sizes, results);
    return 0;
}
