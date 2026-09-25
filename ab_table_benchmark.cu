/**
 * A/B Table Benchmark: C = A * B with independent condition numbers on A and B.
 *
 * Sweeps log10(cond(A)) and log10(cond(B)) from 0 .. 17 (18 x 18 = 324 rows in CSV).
 * Matrix size is fixed at 4096 x 4096. Matrices are generated on the GPU each pass.
 *
 * Methods (vs native FP64 reference):
 *   - FP64 (hipblasDgemm)
 *   - INT8 Ozaki-II, 12 and 16 moduli
 *   - FP8  Ozaki-II, 10 and 12 moduli
 *
 * Relative Frobenius error: ||C_ozaki - C_fp64||_F / ||C_fp64||_F
 *
 * Warmup (NUM_WARMUP iterations) runs only for the first (k,l) = (0,0) case.
 *
 * Ozaki fastmode is selected at compile time:
 *   ab_table_benchmark        — accurate (fastmode=false)
 *   ab_table_benchmark_fast   — fast (fastmode=true)
 *
 * Usage:
 *   ./ab_table_benchmark [output_csv]
 *   ./ab_table_benchmark_fast [output_csv]
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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define HIP_CHECK(call)                                                         \
    do {                                                                        \
        hipError_t err = call;                                                  \
        if (err != hipSuccess) {                                                \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__         \
                      << " code=" << err << " \"" << hipGetErrorString(err)     \
                      << "\"" << std::endl;                                     \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define HIPBLAS_CHECK(call)                                                     \
    do {                                                                        \
        hipblasStatus_t status = call;                                          \
        if (status != HIPBLAS_STATUS_SUCCESS) {                                 \
            std::cerr << "hipBLAS error at " << __FILE__ << ":" << __LINE__     \
                      << " status=" << status << std::endl;                     \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define HIPBLASLT_CHECK(call)                                                   \
    do {                                                                        \
        hipblasStatus_t status = call;                                          \
        if (status != HIPBLAS_STATUS_SUCCESS) {                                 \
            std::cerr << "hipBLASLt error at " << __FILE__ << ":" << __LINE__   \
                      << " status=" << status << std::endl;                     \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define ROCBLAS_CHECK(call)                                                     \
    do {                                                                        \
        rocblas_status status = call;                                           \
        if (status != rocblas_status_success) {                                 \
            std::cerr << "rocBLAS error at " << __FILE__ << ":" << __LINE__     \
                      << " status=" << status << std::endl;                     \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define HIPRAND_CHECK(call)                                                     \
    do {                                                                        \
        hiprandStatus_t status = call;                                          \
        if (status != HIPRAND_STATUS_SUCCESS) {                                 \
            std::cerr << "hipRAND error at " << __FILE__ << ":" << __LINE__     \
                      << " status=" << status << std::endl;                     \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

static constexpr int kN = 4096;
static constexpr int kLog10CondMin = 0;
static constexpr int kLog10CondMax = 17;
static constexpr int kNumWarmupFirstOnly = 10;
static constexpr int kNumTimedIters = 5;
#ifdef AB_TABLE_OZAKI_FAST_MODE
static constexpr bool kFastMode = true;
static constexpr const char* kOzakiModeLabel = "fast";
#else
static constexpr bool kFastMode = false;
static constexpr const char* kOzakiModeLabel = "accurate";
#endif

static constexpr int kInt8Moduli12 = 12;
static constexpr int kInt8Moduli16 = 16;
static constexpr int kFp8Moduli10 = 10;
static constexpr int kFp8Moduli12 = 12;

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

void generate_conditioned_matrix(rocblas_handle rb_handle, hiprandGenerator_t gen, size_t rows,
                                 size_t cols, int log10_cond, double* d_A) {
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
    hipLaunchKernelGGL(compute_singular_values_kernel, dim3(num_blocks), dim3(block_size), 0, 0, d_sv,
                       min_dim, (double)log10_cond);
    hipLaunchKernelGGL(set_diagonal_kernel, dim3(num_blocks), dim3(block_size), 0, 0, d_S, min_dim,
                       min_dim, min_dim, d_sv);
    HIP_CHECK(hipDeviceSynchronize());

    double alpha = 1.0, beta = 0.0;
    ROCBLAS_CHECK(rocblas_dgemm(rb_handle, rocblas_operation_none, rocblas_operation_none, rows, min_dim,
                                min_dim, &alpha, d_U, rows, d_S, min_dim, &beta, d_temp, rows));
    ROCBLAS_CHECK(rocblas_dgemm(rb_handle, rocblas_operation_none, rocblas_operation_transpose, rows,
                                cols, min_dim, &alpha, d_temp, rows, d_V, cols, &beta, d_A, rows));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_U));
    HIP_CHECK(hipFree(d_V));
    HIP_CHECK(hipFree(d_S));
    HIP_CHECK(hipFree(d_temp));
    HIP_CHECK(hipFree(d_tau));
    HIP_CHECK(hipFree(d_sv));
}

double compute_frobenius_rel_error(const double* C_ref, const double* C_test, size_t num_elements) {
    double diff_sum = 0.0, norm_sum = 0.0;
    for (size_t i = 0; i < num_elements; i++) {
        double diff = C_ref[i] - C_test[i];
        diff_sum += diff * diff;
        norm_sum += C_ref[i] * C_ref[i];
    }
    if (norm_sum <= 0.0) return 0.0;
    return std::sqrt(diff_sum) / std::sqrt(norm_sum);
}

std::string get_device_name() {
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::string name = prop.name;
    for (char& c : name) {
        if (c == ' ' || c == '/' || c == '\\') c = '_';
    }
    return name;
}

std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d_%H-%M-%S");
    return ss.str();
}

struct MethodTiming {
    double time_ms = 0.0;
    double tflops = 0.0;
    double frob_rel_err = 0.0;
    double workspace_mb = 0.0;
    std::vector<double> run_time_ms;
};

struct SweepRow {
    int log10_cond_A;
    int log10_cond_B;
    MethodTiming fp64;
    MethodTiming int8_12;
    MethodTiming int8_16;
    MethodTiming fp8_10;
    MethodTiming fp8_12;
};

/** Timed GEMM only (device sync inside window); D2H copy for error check is outside timing. */
double benchmark_dgemm(hipblasHandle_t hb, int n, double* d_A, double* d_B, double* d_C, double alpha,
                       double beta, int num_iters, std::vector<double>& h_out,
                       std::vector<double>* run_times_ms) {
    const size_t size_C = static_cast<size_t>(n) * static_cast<size_t>(n);
    double total_sec = 0.0;
    if (run_times_ms) {
        run_times_ms->clear();
        run_times_ms->reserve(static_cast<size_t>(num_iters));
    }
    for (int i = 0; i < num_iters; i++) {
        HIP_CHECK(hipDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        HIPBLAS_CHECK(hipblasDgemm(hb, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, &alpha, d_A, n, d_B, n, &beta,
                                   d_C, n));
        HIP_CHECK(hipDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(end - start).count();
        total_sec += sec;
        if (run_times_ms) run_times_ms->push_back(sec * 1000.0);
    }
    HIP_CHECK(hipMemcpy(h_out.data(), d_C, size_C * sizeof(double), hipMemcpyDeviceToHost));
    return total_sec / num_iters;
}

double benchmark_int8_ozaki(hipblasHandle_t hb, int n, double* d_A, double* d_B, double* d_C, double alpha,
                            double beta, int moduli, void* d_work, int num_iters, std::vector<double>& h_out,
                            std::vector<double>* run_times_ms) {
    const size_t size_C = static_cast<size_t>(n) * static_cast<size_t>(n);
    double total_sec = 0.0;
    if (run_times_ms) {
        run_times_ms->clear();
        run_times_ms->reserve(static_cast<size_t>(num_iters));
    }
    for (int i = 0; i < num_iters; i++) {
        HIP_CHECK(hipDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        gemmul8::gemm<double, gemmul8::Backend::INT8>(hb, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, &alpha, d_A,
                                                      n, d_B, n, &beta, d_C, n, moduli, kFastMode, d_work);
        HIP_CHECK(hipDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(end - start).count();
        total_sec += sec;
        if (run_times_ms) run_times_ms->push_back(sec * 1000.0);
    }
    HIP_CHECK(hipMemcpy(h_out.data(), d_C, size_C * sizeof(double), hipMemcpyDeviceToHost));
    return total_sec / num_iters;
}

double benchmark_fp8_ozaki(hipblasLtHandle_t hblt, int n, double* d_A, double* d_B, double* d_C, double alpha,
                           double beta, int moduli, void* d_work, int num_iters, std::vector<double>& h_out,
                           std::vector<double>* run_times_ms) {
    const size_t size_C = static_cast<size_t>(n) * static_cast<size_t>(n);
    double total_sec = 0.0;
    if (run_times_ms) {
        run_times_ms->clear();
        run_times_ms->reserve(static_cast<size_t>(num_iters));
    }
    for (int i = 0; i < num_iters; i++) {
        HIP_CHECK(hipDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        gemmul8::gemmLt<double, gemmul8::Backend::FP8>(hblt, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, &alpha,
                                                       d_A, n, d_B, n, &beta, d_C, n, moduli, kFastMode,
                                                       d_work);
        HIP_CHECK(hipDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(end - start).count();
        total_sec += sec;
        if (run_times_ms) run_times_ms->push_back(sec * 1000.0);
    }
    HIP_CHECK(hipMemcpy(h_out.data(), d_C, size_C * sizeof(double), hipMemcpyDeviceToHost));
    return total_sec / num_iters;
}

void run_warmup_first_case(hipblasHandle_t hb, hipblasLtHandle_t hblt, int n, double* d_A, double* d_B,
                           double* d_C, void* d_work, double alpha, double beta) {
    for (int w = 0; w < kNumWarmupFirstOnly; w++) {
        HIPBLAS_CHECK(hipblasDgemm(hb, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, &alpha, d_A, n, d_B, n, &beta,
                                   d_C, n));
        gemmul8::gemm<double, gemmul8::Backend::INT8>(hb, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, &alpha, d_A,
                                                      n, d_B, n, &beta, d_C, n, kInt8Moduli12, kFastMode,
                                                      d_work);
        gemmul8::gemm<double, gemmul8::Backend::INT8>(hb, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, &alpha, d_A,
                                                      n, d_B, n, &beta, d_C, n, kInt8Moduli16, kFastMode,
                                                      d_work);
        gemmul8::gemmLt<double, gemmul8::Backend::FP8>(hblt, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, &alpha,
                                                       d_A, n, d_B, n, &beta, d_C, n, kFp8Moduli10, kFastMode,
                                                       d_work);
        gemmul8::gemmLt<double, gemmul8::Backend::FP8>(hblt, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, &alpha,
                                                       d_A, n, d_B, n, &beta, d_C, n, kFp8Moduli12, kFastMode,
                                                       d_work);
    }
    HIP_CHECK(hipDeviceSynchronize());
}

void write_csv_header(std::ofstream& out) {
    // Each method: time_ms and tflops are from kNumTimedIters GPU GEMM calls (host memcpy excluded).
    out << "log10_cond_A,log10_cond_B,cond_A_label,cond_B_label,matrix_n,timed_iters,ozaki_fastmode,"
        << "fp64_time_ms,fp64_tflops,fp64_frob_rel_err,"
        << "int8_ozaki12_time_ms,int8_ozaki12_tflops,int8_ozaki12_frob_rel_err,int8_ozaki12_workspace_mb,"
        << "int8_ozaki16_time_ms,int8_ozaki16_tflops,int8_ozaki16_frob_rel_err,int8_ozaki16_workspace_mb,"
        << "fp8_ozaki10_time_ms,fp8_ozaki10_tflops,fp8_ozaki10_frob_rel_err,fp8_ozaki10_workspace_mb,"
        << "fp8_ozaki12_time_ms,fp8_ozaki12_tflops,fp8_ozaki12_frob_rel_err,fp8_ozaki12_workspace_mb\n";
}

void print_pair_results(const SweepRow& row) {
    std::cout << "\n--- cond(A)=10^" << row.log10_cond_A << "  cond(B)=10^" << row.log10_cond_B
              << "  (N=" << kN << ", Ozaki " << kOzakiModeLabel << ", " << kNumTimedIters
              << " timed GEMMs each) ---\n";
    std::cout << std::left << std::setw(22) << "Method";
    for (int r = 1; r <= kNumTimedIters; r++) {
        std::cout << std::right << std::setw(10) << ("run" + std::to_string(r) + "_ms");
    }
    std::cout << std::setw(12) << "avg_ms" << std::setw(12) << "TFLOPS" << std::setw(14) << "frob_rel_err"
              << "\n";
    std::cout << std::string(22 + 10 * kNumTimedIters + 12 + 12 + 14, '-') << "\n";

    auto print_method = [](const char* name, const MethodTiming& m) {
        std::cout << std::left << std::setw(22) << name;
        for (double t_ms : m.run_time_ms) {
            std::cout << std::right << std::fixed << std::setprecision(2) << std::setw(10) << t_ms;
        }
        std::cout << std::setw(12) << m.time_ms << std::setw(12) << std::setprecision(2) << m.tflops
                  << std::setw(14) << std::scientific << std::setprecision(4) << m.frob_rel_err << std::fixed
                  << "\n";
    };

    print_method("FP64 (native)", row.fp64);
    print_method("INT8 Ozaki 12", row.int8_12);
    print_method("INT8 Ozaki 16", row.int8_16);
    print_method("FP8 Ozaki 10", row.fp8_10);
    print_method("FP8 Ozaki 12", row.fp8_12);
    std::cout << std::flush;
}

void write_csv_row(std::ofstream& out, const SweepRow& row) {
    auto cond_label = [](int k) {
        std::stringstream ss;
        ss << "1e" << k;
        return ss.str();
    };

    auto write_method = [&](const MethodTiming& m, bool with_workspace) {
        out << "," << m.time_ms << "," << m.tflops << "," << std::scientific << m.frob_rel_err << std::fixed;
        if (with_workspace) {
            out << "," << m.workspace_mb;
        }
    };

    out << row.log10_cond_A << "," << row.log10_cond_B << "," << cond_label(row.log10_cond_A) << ","
        << cond_label(row.log10_cond_B) << "," << kN << "," << kNumTimedIters << ","
        << (kFastMode ? 1 : 0);

    write_method(row.fp64, false);
    write_method(row.int8_12, true);
    write_method(row.int8_16, true);
    write_method(row.fp8_10, true);
    write_method(row.fp8_12, true);
    out << "\n";
}

int main(int argc, char* argv[]) {
    const char* prof = getenv("GEMMUL8_PROFILE");
    if (prof && std::string(prof) == "1") {
        oz2::g_profiling_enabled = true;
        std::cerr << "[GEMMUL8] Profiling enabled" << std::endl;
    }

    std::string csv_path;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::cerr << "Usage: " << argv[0] << " [output.csv]\n"
                      << "  Ozaki fastmode: " << kOzakiModeLabel << " (compile-time; use ab_table_benchmark_fast for fast)\n";
            return 0;
        }
        if (csv_path.empty()) {
            csv_path = arg;
        }
    }
    if (csv_path.empty()) {
        csv_path = std::string("ab_table_benchmark_") + kOzakiModeLabel + "_" + get_device_name() + "_" +
                   get_timestamp() + ".csv";
    }

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cerr << "Device: " << prop.name << std::endl;
    std::cerr << "Matrix: " << kN << "x" << kN << " GEMM C = A * B" << std::endl;
    std::cerr << "Condition sweep: log10(cond) = " << kLog10CondMin << ".." << kLog10CondMax
              << " for A and B (" << (kLog10CondMax - kLog10CondMin + 1) << " x "
              << (kLog10CondMax - kLog10CondMin + 1) << " rows)" << std::endl;
    std::cerr << "Ozaki fastmode: " << kOzakiModeLabel << std::endl;
    std::cerr << "Output CSV: " << csv_path << std::endl;

    rocblas_handle rb_handle;
    hipblasHandle_t hb_handle;
    hipblasLtHandle_t hblt_handle;
    hiprandGenerator_t gen;

    ROCBLAS_CHECK(rocblas_create_handle(&rb_handle));
    HIPBLAS_CHECK(hipblasCreate(&hb_handle));
    HIPBLASLT_CHECK(hipblasLtCreate(&hblt_handle));
    HIPRAND_CHECK(hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT));

    const int n = kN;
    const size_t size = static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t bytes = size * sizeof(double);

    double *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, bytes));
    HIP_CHECK(hipMalloc(&d_B, bytes));
    HIP_CHECK(hipMalloc(&d_C, bytes));

    size_t ws_max = std::max({gemmul8::workSize<false, gemmul8::Backend::INT8>(n, n, n, kInt8Moduli12),
                              gemmul8::workSize<false, gemmul8::Backend::INT8>(n, n, n, kInt8Moduli16),
                              gemmul8::workSize<false, gemmul8::Backend::FP8>(n, n, n, kFp8Moduli10),
                              gemmul8::workSize<false, gemmul8::Backend::FP8>(n, n, n, kFp8Moduli12)});
    void* d_work;
    HIP_CHECK(hipMalloc(&d_work, ws_max));

    std::vector<double> h_C_ref(size), h_C_test(size);
    double alpha = 1.0, beta = 0.0;

    std::ofstream csv(csv_path);
    if (!csv.is_open()) {
        std::cerr << "Failed to open " << csv_path << std::endl;
        return 1;
    }
    csv << std::fixed << std::setprecision(6);
    write_csv_header(csv);

    const double ws_int8_12_mb =
        gemmul8::workSize<false, gemmul8::Backend::INT8>(n, n, n, kInt8Moduli12) / (1024.0 * 1024.0);
    const double ws_int8_16_mb =
        gemmul8::workSize<false, gemmul8::Backend::INT8>(n, n, n, kInt8Moduli16) / (1024.0 * 1024.0);
    const double ws_fp8_10_mb =
        gemmul8::workSize<false, gemmul8::Backend::FP8>(n, n, n, kFp8Moduli10) / (1024.0 * 1024.0);
    const double ws_fp8_12_mb =
        gemmul8::workSize<false, gemmul8::Backend::FP8>(n, n, n, kFp8Moduli12) / (1024.0 * 1024.0);

    const double flop_count = 2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
    bool first_case = true;
    int row_count = 0;

    for (int log10_a = kLog10CondMin; log10_a <= kLog10CondMax; log10_a++) {
        for (int log10_b = kLog10CondMin; log10_b <= kLog10CondMax; log10_b++) {
            std::cerr << "log10(cond A)=" << log10_a << " log10(cond B)=" << log10_b << ": generating..."
                      << std::flush;

            unsigned long long seed = 12345ULL + static_cast<unsigned long long>(log10_a) * 1000ULL +
                                      static_cast<unsigned long long>(log10_b);
            HIPRAND_CHECK(hiprandSetPseudoRandomGeneratorSeed(gen, seed));

            generate_conditioned_matrix(rb_handle, gen, n, n, log10_a, d_A);
            generate_conditioned_matrix(rb_handle, gen, n, n, log10_b, d_B);
            std::cerr << " benchmark..." << std::flush;

            if (first_case) {
                run_warmup_first_case(hb_handle, hblt_handle, n, d_A, d_B, d_C, d_work, alpha, beta);
                first_case = false;
            }

            SweepRow row{};
            row.log10_cond_A = log10_a;
            row.log10_cond_B = log10_b;

            auto fill_ozaki = [&](double sec, const std::vector<double>& h_result, MethodTiming& slot,
                                  const std::vector<double>& runs_ms, double workspace_mb) {
                slot.time_ms = sec * 1000.0;
                slot.tflops = flop_count / (sec * 1e12);
                slot.frob_rel_err = compute_frobenius_rel_error(h_C_ref.data(), h_result.data(), size);
                slot.run_time_ms = runs_ms;
                slot.workspace_mb = workspace_mb;
            };

            std::vector<double> runs_ms;
            double fp64_sec = benchmark_dgemm(hb_handle, n, d_A, d_B, d_C, alpha, beta, kNumTimedIters,
                                              h_C_ref, &runs_ms);
            row.fp64.time_ms = fp64_sec * 1000.0;
            row.fp64.tflops = flop_count / (fp64_sec * 1e12);
            row.fp64.frob_rel_err = 0.0;
            row.fp64.run_time_ms = runs_ms;

            fill_ozaki(benchmark_int8_ozaki(hb_handle, n, d_A, d_B, d_C, alpha, beta, kInt8Moduli12, d_work,
                                            kNumTimedIters, h_C_test, &runs_ms),
                       h_C_test, row.int8_12, runs_ms, ws_int8_12_mb);
            fill_ozaki(benchmark_int8_ozaki(hb_handle, n, d_A, d_B, d_C, alpha, beta, kInt8Moduli16, d_work,
                                            kNumTimedIters, h_C_test, &runs_ms),
                       h_C_test, row.int8_16, runs_ms, ws_int8_16_mb);
            fill_ozaki(benchmark_fp8_ozaki(hblt_handle, n, d_A, d_B, d_C, alpha, beta, kFp8Moduli10, d_work,
                                             kNumTimedIters, h_C_test, &runs_ms),
                       h_C_test, row.fp8_10, runs_ms, ws_fp8_10_mb);
            fill_ozaki(benchmark_fp8_ozaki(hblt_handle, n, d_A, d_B, d_C, alpha, beta, kFp8Moduli12, d_work,
                                             kNumTimedIters, h_C_test, &runs_ms),
                       h_C_test, row.fp8_12, runs_ms, ws_fp8_12_mb);

            write_csv_row(csv, row);
            csv.flush();
            row_count++;

            std::cerr << " done\n";
            print_pair_results(row);
        }
    }

    csv.close();

    HIP_CHECK(hipFree(d_work));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    HIPRAND_CHECK(hiprandDestroyGenerator(gen));
    HIPBLASLT_CHECK(hipblasLtDestroy(hblt_handle));
    HIPBLAS_CHECK(hipblasDestroy(hb_handle));
    ROCBLAS_CHECK(rocblas_destroy_handle(rb_handle));

    std::cerr << "Wrote " << row_count << " data rows to " << csv_path << std::endl;
    if (row_count != (kLog10CondMax - kLog10CondMin + 1) * (kLog10CondMax - kLog10CondMin + 1)) {
        std::cerr << "Warning: expected "
                  << (kLog10CondMax - kLog10CondMin + 1) * (kLog10CondMax - kLog10CondMin + 1) << " rows\n";
        return 1;
    }
    return 0;
}
