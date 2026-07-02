/**
 * INT8 vs FP8 Table Benchmark (gemmLt)
 * ------------------------------------
 * Compares the two low-precision Ozaki-II backends of GEMMul8 head to head,
 * both driven through gemmul8::gemmLt with a single hipBLASLt handle:
 *   - gemmul8::Backend::INT8  (INT8 matrix engine)
 *   - gemmul8::Backend::FP8   (FP8  matrix engine)
 *
 * For each requested matrix size and each modulus count it runs 10 warmup +
 * 50 timed iterations of each backend and reports, side by side:
 *   - Native FP64 GEMM performance (reference)
 *   - INT8 / FP8 performance (TFLOPS)
 *   - INT8 / FP8 workspace size (MB)
 *   - INT8 / FP8 accuracy (relative Frobenius error vs native FP64)
 *
 * All variants compute C = A * A on a square matrix with condition 10^8.
 *
 * Note: FP8 matrix engines are not available on all AMD GPUs / ROCm builds.
 * If an FP8 run fails on this device, FP8 columns are reported as "N/A"
 * (printed as "-") and the rest of the benchmark continues.
 *
 * Usage:
 *   ./gemm_int8_vs_fp8_benchmark <size1> [size2] [size3] ...
 *   Example: ./gemm_int8_vs_fp8_benchmark 1024 2048 4096 8192
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
#include <sstream>
#include <vector>
#include <string>
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
// Moduli are chosen independently per backend (FP8 carries fewer bits/modulus).
static const std::vector<int> INT8_MODULI = {12, 16};
static const std::vector<int> FP8_MODULI  = {10, 12};

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

// Result for one backend at one modulus count.
struct ModuliResult {
    int moduli;
    bool ok;          // false if the backend was unsupported / failed (FP8 only)
    double tflops;
    double error;
    size_t ws;
};

struct BenchmarkResult {
    double native_tflops;
    std::vector<ModuliResult> int8;   // one entry per INT8_MODULI
    std::vector<ModuliResult> fp8;    // one entry per FP8_MODULI
};

// Runs one gemmLt call for the given backend; returns true on success.
// Probes for FP8 support by checking the HIP error state after a single call.
template <gemmul8::Backend BACKEND>
bool try_gemmLt_once(hipblasLtHandle_t handle, size_t n,
                     const double* alpha, const double* d_A,
                     const double* beta, double* d_C,
                     int moduli, void* d_work, hipStream_t stream) {
    (void)hipGetLastError(); // clear any stale error
    gemmul8::gemmLt<double, BACKEND>(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                          n, n, n, alpha, d_A, n, d_A, n, beta, d_C, n,
                          moduli, FASTMODE, d_work,
                          nullptr, nullptr, false, false, false, false, stream);
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        (void)hipGetLastError(); // clear so subsequent work isn't poisoned
        return false;
    }
    return true;
}

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

    // Workspace sized for the largest moduli of either backend.
    int max_moduli = 0;
    for (int m : INT8_MODULI) max_moduli = std::max(max_moduli, m);
    for (int m : FP8_MODULI)  max_moduli = std::max(max_moduli, m);
    size_t ws_int8_max = gemmul8::workSize<false, gemmul8::Backend::INT8>(n, n, n, max_moduli);
    size_t ws_fp8_max  = gemmul8::workSize<false, gemmul8::Backend::FP8>(n, n, n, max_moduli);
    size_t worksize = std::max(ws_int8_max, ws_fp8_max);
    void* d_work;
    HIP_CHECK(hipMalloc(&d_work, worksize));

    // --- INT8 backend ---
    for (int moduli : INT8_MODULI) {
        ModuliResult mr = {moduli, true, 0, 0, 0};
        mr.ws = gemmul8::workSize<false, gemmul8::Backend::INT8>(n, n, n, moduli);

        std::cerr << "  Benchmarking INT8 (" << moduli << " moduli)..." << std::flush;
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

        double int8_time = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
        mr.tflops = (2.0 * n * n * n) / (int8_time * 1e12);
        HIP_CHECK(hipMemcpy(h_C_ozaki.data(), d_C_ozaki, bytes, hipMemcpyDeviceToHost));
        mr.error = compute_frobenius_rel_error(h_C_native.data(), h_C_ozaki.data(), size);
        std::cerr << " " << mr.tflops << " TFLOPS, error=" << mr.error << std::endl;

        result.int8.push_back(mr);
    }

    // --- FP8 backend (probe support first for each modulus count) ---
    for (int moduli : FP8_MODULI) {
        ModuliResult mr = {moduli, false, 0, 0, 0};
        mr.ws = gemmul8::workSize<false, gemmul8::Backend::FP8>(n, n, n, moduli);

        std::cerr << "  Benchmarking FP8  (" << moduli << " moduli)..." << std::flush;
        mr.ok = try_gemmLt_once<gemmul8::Backend::FP8>(
            hblt_handle, n, &alpha, d_A, &beta, d_C_ozaki, moduli, d_work, stream);

        if (!mr.ok) {
            std::cerr << " UNSUPPORTED / failed on this device" << std::endl;
        } else {
            for (int i = 1; i < NUM_WARMUP; i++) {
                gemmul8::gemmLt<double, gemmul8::Backend::FP8>(hblt_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                      n, n, n, &alpha, d_A, n, d_A, n, &beta, d_C_ozaki, n,
                                      moduli, FASTMODE, d_work,
                                      nullptr, nullptr, false, false, false, false, stream);
            }
            HIP_CHECK(hipDeviceSynchronize());

            start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < NUM_ITERATIONS; i++) {
                gemmul8::gemmLt<double, gemmul8::Backend::FP8>(hblt_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                      n, n, n, &alpha, d_A, n, d_A, n, &beta, d_C_ozaki, n,
                                      moduli, FASTMODE, d_work,
                                      nullptr, nullptr, false, false, false, false, stream);
            }
            HIP_CHECK(hipDeviceSynchronize());
            end = std::chrono::high_resolution_clock::now();

            double fp8_time = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
            mr.tflops = (2.0 * n * n * n) / (fp8_time * 1e12);
            HIP_CHECK(hipMemcpy(h_C_ozaki.data(), d_C_ozaki, bytes, hipMemcpyDeviceToHost));
            mr.error = compute_frobenius_rel_error(h_C_native.data(), h_C_ozaki.data(), size);
            std::cerr << " " << mr.tflops << " TFLOPS, error=" << mr.error << std::endl;
        }

        result.fp8.push_back(mr);
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

// Format a TFLOPS value, or "-" if the backend was unavailable.
static std::string fmt_tflops(double v, bool ok) {
    if (!ok) return "-";
    std::ostringstream os;
    os << std::fixed << std::setprecision(2) << v;
    return os.str();
}

static std::string fmt_err(double v, bool ok) {
    if (!ok) return "-";
    std::ostringstream os;
    os << std::scientific << std::setprecision(2) << v;
    return os.str();
}

// Selects which ModuliResult vector to read from a BenchmarkResult.
static const std::vector<ModuliResult>& pick(const BenchmarkResult& r, bool int8) {
    return int8 ? r.int8 : r.fp8;
}

// Prints performance / workspace / accuracy tables for a single backend.
void print_backend_tables(const std::string& name,
                          const std::vector<int>& mods,
                          bool int8,
                          const std::vector<size_t>& sizes,
                          const std::vector<BenchmarkResult>& results) {
    std::cout << std::endl;
    std::cout << name << " backend  (gemmLt, " << NUM_ITERATIONS
              << " runs each, cond=10^" << LOG10_COND << ")" << std::endl;
    std::cout << std::endl;

    auto sep = [&]() {
        std::cout << "+--------+---------+";
        for (size_t m = 0; m < mods.size(); m++) std::cout << "----------+";
        std::cout << std::endl;
    };

    // Performance.
    std::cout << "Performance (TFLOPS):" << std::endl;
    sep();
    std::cout << "| Size   | Native  |";
    for (int mod : mods) std::cout << std::setw(9) << ("mod=" + std::to_string(mod)) << " |";
    std::cout << std::endl;
    sep();
    for (size_t i = 0; i < sizes.size(); i++) {
        const std::vector<ModuliResult>& v = pick(results[i], int8);
        std::cout << "| " << std::setw(6) << sizes[i]
                  << " | " << std::setw(7) << std::fixed << std::setprecision(2) << results[i].native_tflops
                  << " |";
        for (const ModuliResult& mr : v)
            std::cout << " " << std::setw(8) << fmt_tflops(mr.tflops, mr.ok) << " |";
        std::cout << std::endl;
    }
    sep();

    // Workspace.
    std::cout << std::endl << "Workspace size (MB):" << std::endl;
    auto wsep = [&]() {
        std::cout << "+--------+";
        for (size_t m = 0; m < mods.size(); m++) std::cout << "----------+";
        std::cout << std::endl;
    };
    wsep();
    std::cout << "| Size   |";
    for (int mod : mods) std::cout << std::setw(9) << ("mod=" + std::to_string(mod)) << " |";
    std::cout << std::endl;
    wsep();
    for (size_t i = 0; i < sizes.size(); i++) {
        const std::vector<ModuliResult>& v = pick(results[i], int8);
        std::cout << "| " << std::setw(6) << sizes[i] << " |";
        for (const ModuliResult& mr : v)
            std::cout << " " << std::setw(8) << std::fixed << std::setprecision(1)
                      << (mr.ws / (1024.0 * 1024.0)) << " |";
        std::cout << std::endl;
    }
    wsep();

    // Accuracy.
    std::cout << std::endl << "Accuracy (relative Frobenius error vs native FP64):" << std::endl;
    auto asep = [&]() {
        std::cout << "+--------+";
        for (size_t m = 0; m < mods.size(); m++) std::cout << "------------+";
        std::cout << std::endl;
    };
    asep();
    std::cout << "| Size   |";
    for (int mod : mods) std::cout << " " << std::setw(9) << ("mod=" + std::to_string(mod)) << " |";
    std::cout << std::endl;
    asep();
    for (size_t i = 0; i < sizes.size(); i++) {
        const std::vector<ModuliResult>& v = pick(results[i], int8);
        std::cout << "| " << std::setw(6) << sizes[i] << " |";
        for (const ModuliResult& mr : v)
            std::cout << " " << std::setw(10) << fmt_err(mr.error, mr.ok) << " |";
        std::cout << std::endl;
    }
    asep();
}

enum class Metric { PERF, WS, ERR };

// Prints a single combined table with all INT8 and FP8 (backend, moduli)
// columns side by side, one row per matrix size.
void print_combined_table(Metric metric,
                          const std::vector<size_t>& sizes,
                          const std::vector<BenchmarkResult>& results) {
    // Column labels in display order: INT8 first, then FP8.
    std::vector<std::string> labels;
    for (int mod : INT8_MODULI) labels.push_back("INT8 m" + std::to_string(mod));
    for (int mod : FP8_MODULI)  labels.push_back("FP8 m"  + std::to_string(mod));
    const bool show_native = (metric == Metric::PERF);

    const int colw = (metric == Metric::ERR) ? 10 : 9;

    auto cell_value = [&](const BenchmarkResult& r, size_t col) -> std::string {
        // Map flat column index back to (backend, moduli index).
        bool int8 = col < INT8_MODULI.size();
        const std::vector<ModuliResult>& v = int8 ? r.int8 : r.fp8;
        size_t idx = int8 ? col : col - INT8_MODULI.size();
        const ModuliResult& mr = v[idx];
        switch (metric) {
            case Metric::PERF: return fmt_tflops(mr.tflops, mr.ok);
            case Metric::ERR:  return fmt_err(mr.error, mr.ok);
            case Metric::WS: {
                if (!mr.ok) return "-";
                std::ostringstream os;
                os << std::fixed << std::setprecision(1) << (mr.ws / (1024.0 * 1024.0));
                return os.str();
            }
        }
        return "";
    };

    auto sep = [&]() {
        std::cout << "+--------+";
        if (show_native) std::cout << "---------+";
        for (size_t c = 0; c < labels.size(); c++) {
            std::cout << std::string(colw + 2, '-') << "+";
        }
        std::cout << std::endl;
    };

    const char* title = (metric == Metric::PERF) ? "Performance (TFLOPS)"
                      : (metric == Metric::WS)   ? "Workspace size (MB)"
                      : "Accuracy (relative Frobenius error vs native FP64)";
    std::cout << std::endl << title << ":" << std::endl;
    sep();
    std::cout << "| Size   |";
    if (show_native) std::cout << " Native  |";
    for (const std::string& lab : labels) std::cout << " " << std::setw(colw) << lab << " |";
    std::cout << std::endl;
    sep();
    for (size_t i = 0; i < sizes.size(); i++) {
        std::cout << "| " << std::setw(6) << sizes[i] << " |";
        if (show_native)
            std::cout << " " << std::setw(7) << std::fixed << std::setprecision(2)
                      << results[i].native_tflops << " |";
        for (size_t c = 0; c < labels.size(); c++)
            std::cout << " " << std::setw(colw) << cell_value(results[i], c) << " |";
        std::cout << std::endl;
    }
    sep();
}

void print_summary(const std::vector<size_t>& sizes, const std::vector<BenchmarkResult>& results) {
    std::cout << std::endl;
    std::cout << "=================== INT8 vs FP8 comparison ===================" << std::endl;

    print_backend_tables("INT8", INT8_MODULI, true,  sizes, results);
    print_backend_tables("FP8",  FP8_MODULI,  false, sizes, results);

    // Combined side-by-side comparison tables.
    std::cout << std::endl;
    std::cout << "=================== INT8 vs FP8 (combined) ===================" << std::endl;
    print_combined_table(Metric::PERF, sizes, results);
    print_combined_table(Metric::WS,   sizes, results);
    print_combined_table(Metric::ERR,  sizes, results);

    std::cout << std::endl;
    std::cout << "Legend:" << std::endl;
    std::cout << "  - Native FP64: hipBLAS FP64 reference TFLOPS" << std::endl;
    std::cout << "  - INT8 / FP8: Ozaki-II TFLOPS via gemmLt with the given backend" << std::endl;
    std::cout << "  - FP8 is run at 10 and 12 moduli; INT8 at 12 and 16 moduli" << std::endl;
    std::cout << "  - '-' : FP8 backend unsupported or failed on this device" << std::endl;
    std::cout << "==============================================================" << std::endl;
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

    std::cerr << "Backends: INT8 vs FP8 (both via gemmLt / hipBLASLt)" << std::endl;
    std::cerr << "Condition number: 10^" << LOG10_COND << std::endl;
    std::cerr << "Iterations: " << NUM_ITERATIONS << " (warmup " << NUM_WARMUP << ")" << std::endl;
    std::cerr << "INT8 moduli: ";
    for (int m : INT8_MODULI) std::cerr << m << " ";
    std::cerr << std::endl << "FP8 moduli: ";
    for (int m : FP8_MODULI) std::cerr << m << " ";
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
