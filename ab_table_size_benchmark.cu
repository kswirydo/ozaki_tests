/**
 * A/B Table Benchmark by size: C = A * B with two *different* square matrices.
 *
 * Same table layout as table_benchmark, with three differences:
 *   - condition number is 10^3 (both A and B)
 *   - A and B are independent matrices (table_benchmark squares a single A)
 *   - each method is timed over 50 separate runs, summarized as
 *     mean / median / min / max / stddev (warmup still runs before the timed loop)
 *
 * Methods: native FP64 (hipblasDgemm), INT8 Ozaki-II 12 moduli, INT8 Ozaki-II
 * 16 moduli. Accuracy is the relative Frobenius error against native FP64.
 *
 * Usage:
 *   ./ab_table_size_benchmark <size>
 *   ./ab_table_size_benchmark <size1> <size2> <size3> ...
 */

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <hiprand/hiprand.h>
#include "gemmul8.hpp"
#include <cstdlib>

// Enable GEMMUL8_PROFILE support
namespace oz2 { bool g_profiling_enabled = false; }

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

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
static const int NUM_RUNS = 50;     // timed runs per method
static const int LOG10_COND = 3;    // condition number 10^3 for both A and B
static const int MODULI_LOW = 12;
static const int MODULI_HIGH = 16;
static const unsigned long long SEED = 12345ULL;

// Kernels for matrix generation
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

    // Generate random U and V
    HIPRAND_CHECK(hiprandGenerateNormalDouble(gen, d_U, size, 0.0, 1.0));
    HIPRAND_CHECK(hiprandGenerateNormalDouble(gen, d_V, size, 0.0, 1.0));

    // QR decomposition to get orthogonal matrices
    rocsolver_dgeqrf(rb_handle, n, n, d_U, n, d_tau);
    rocsolver_dorgqr(rb_handle, n, n, n, d_U, n, d_tau);

    rocsolver_dgeqrf(rb_handle, n, n, d_V, n, d_tau);
    rocsolver_dorgqr(rb_handle, n, n, n, d_V, n, d_tau);

    // Zero S and set diagonal
    int block_size = 256;
    size_t num_blocks = (size + block_size - 1) / block_size;
    hipLaunchKernelGGL(zero_matrix_kernel, dim3(num_blocks), dim3(block_size), 0, 0, d_S, size);

    num_blocks = (n + block_size - 1) / block_size;
    hipLaunchKernelGGL(compute_singular_values_kernel, dim3(num_blocks), dim3(block_size),
                       0, 0, d_sv, n, (double)log10_cond);
    hipLaunchKernelGGL(set_diagonal_kernel, dim3(num_blocks), dim3(block_size),
                       0, 0, d_S, n, n, d_sv);
    HIP_CHECK(hipDeviceSynchronize());

    // A = U * S * V'
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

struct MethodResult {
    std::string label;
    std::vector<double> run_ms;   // NUM_RUNS individual timings
    double mean_ms = 0.0;
    double median_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double stddev_ms = 0.0;
    double mean_tflops = 0.0;
    double min_tflops = 0.0;
    double max_tflops = 0.0;
    double error = 0.0;           // relative Frobenius error vs native FP64
};

void summarize(MethodResult& m, size_t n) {
    const double flop = 2.0 * (double)n * (double)n * (double)n;
    const size_t count = m.run_ms.size();

    m.mean_ms = std::accumulate(m.run_ms.begin(), m.run_ms.end(), 0.0) / count;
    m.min_ms = *std::min_element(m.run_ms.begin(), m.run_ms.end());
    m.max_ms = *std::max_element(m.run_ms.begin(), m.run_ms.end());

    std::vector<double> sorted = m.run_ms;
    std::sort(sorted.begin(), sorted.end());
    m.median_ms = (count % 2 == 1) ? sorted[count / 2]
                                   : 0.5 * (sorted[count / 2 - 1] + sorted[count / 2]);

    double sq = 0.0;
    for (double ms : m.run_ms) sq += (ms - m.mean_ms) * (ms - m.mean_ms);
    m.stddev_ms = (count > 1) ? std::sqrt(sq / (count - 1)) : 0.0;

    m.mean_tflops = flop / (m.mean_ms * 1e-3 * 1e12);
    m.max_tflops = flop / (m.min_ms * 1e-3 * 1e12);
    m.min_tflops = flop / (m.max_ms * 1e-3 * 1e12);
}

struct BenchmarkResult {
    MethodResult native;
    MethodResult ozaki12;
    MethodResult ozaki16;
};

BenchmarkResult run_benchmark(size_t n) {
    BenchmarkResult result;
    result.native.label = "Native FP64";
    result.ozaki12.label = "Ozaki II (12 moduli)";
    result.ozaki16.label = "Ozaki II (16 moduli)";

    size_t size = n * n;
    size_t bytes = size * sizeof(double);

    rocblas_handle rb_handle;
    hipblasHandle_t hb_handle;
    hiprandGenerator_t gen;

    ROCBLAS_CHECK(rocblas_create_handle(&rb_handle));
    HIPBLAS_CHECK(hipblasCreate(&hb_handle));
    HIPRAND_CHECK(hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT));
    HIPRAND_CHECK(hiprandSetPseudoRandomGeneratorSeed(gen, SEED));

    double *d_A, *d_B, *d_C_native, *d_C_ozaki;
    HIP_CHECK(hipMalloc(&d_A, bytes));
    HIP_CHECK(hipMalloc(&d_B, bytes));
    HIP_CHECK(hipMalloc(&d_C_native, bytes));
    HIP_CHECK(hipMalloc(&d_C_ozaki, bytes));

    // A and B both have condition number 10^LOG10_COND but are drawn
    // consecutively from one RNG stream, so they are different matrices.
    std::cerr << "  Generating A (" << n << "x" << n << ", cond=10^" << LOG10_COND << ")..." << std::flush;
    generate_conditioned_matrix(rb_handle, gen, n, LOG10_COND, d_A);
    std::cerr << " B..." << std::flush;
    generate_conditioned_matrix(rb_handle, gen, n, LOG10_COND, d_B);
    std::cerr << " done" << std::endl;

    double alpha = 1.0, beta = 0.0;

    auto time_runs = [&](MethodResult& m, auto&& launch) {
        for (int i = 0; i < NUM_WARMUP; i++) launch();
        HIP_CHECK(hipDeviceSynchronize());

        m.run_ms.clear();
        for (int i = 0; i < NUM_RUNS; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            launch();
            HIP_CHECK(hipDeviceSynchronize());
            auto end = std::chrono::high_resolution_clock::now();
            m.run_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }
        summarize(m, n);
    };

    // Native FP64 reference: C = A * B
    std::cerr << "  Benchmarking native GEMM (" << NUM_WARMUP << " warmup + " << NUM_RUNS
              << " runs)..." << std::flush;
    time_runs(result.native, [&]() {
        HIPBLAS_CHECK(hipblasDgemm(hb_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                   n, n, n, &alpha, d_A, n, d_B, n, &beta, d_C_native, n));
    });
    std::cerr << " " << result.native.mean_tflops << " TFLOPS" << std::endl;

    std::vector<double> h_C_native(size), h_C_ozaki(size);
    HIP_CHECK(hipMemcpy(h_C_native.data(), d_C_native, bytes, hipMemcpyDeviceToHost));

    size_t worksize = std::max(gemmul8::workSize<false, gemmul8::Backend::INT8>(n, n, n, MODULI_LOW),
                               gemmul8::workSize<false, gemmul8::Backend::INT8>(n, n, n, MODULI_HIGH));
    void* d_work;
    HIP_CHECK(hipMalloc(&d_work, worksize));

    for (int moduli : {MODULI_LOW, MODULI_HIGH}) {
        MethodResult& m = (moduli == MODULI_LOW) ? result.ozaki12 : result.ozaki16;
        std::cerr << "  Benchmarking Ozaki-II (" << moduli << " moduli, " << NUM_WARMUP
                  << " warmup + " << NUM_RUNS << " runs)..." << std::flush;
        time_runs(m, [&]() {
            gemmul8::gemm<double>(hb_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                  n, n, n, &alpha, d_A, n, d_B, n, &beta, d_C_ozaki, n,
                                  moduli, false, d_work);
        });
        HIP_CHECK(hipMemcpy(h_C_ozaki.data(), d_C_ozaki, bytes, hipMemcpyDeviceToHost));
        m.error = compute_frobenius_rel_error(h_C_native.data(), h_C_ozaki.data(), size);
        std::cerr << " " << m.mean_tflops << " TFLOPS, error=" << m.error << std::endl;
    }

    HIP_CHECK(hipFree(d_work));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C_native));
    HIP_CHECK(hipFree(d_C_ozaki));
    HIPRAND_CHECK(hiprandDestroyGenerator(gen));
    HIPBLAS_CHECK(hipblasDestroy(hb_handle));
    ROCBLAS_CHECK(rocblas_destroy_handle(rb_handle));

    return result;
}

void print_runs_table(size_t n, const BenchmarkResult& r) {
    const MethodResult* methods[] = {&r.native, &r.ozaki12, &r.ozaki16};
    const char* sep = "+----------------------+----------+----------+----------+----------+----------+";

    std::cout << std::endl;
    std::cout << "Run statistics for " << n << "x" << n << " (" << NUM_RUNS
              << " timed runs after " << NUM_WARMUP << " warmup iterations), times in ms" << std::endl;
    std::cout << sep << std::endl;
    std::cout << "| " << std::setw(20) << std::left << "Method" << std::right << " |"
              << std::setw(9) << "mean" << " |" << std::setw(9) << "median" << " |"
              << std::setw(9) << "min" << " |" << std::setw(9) << "max" << " |"
              << std::setw(9) << "stddev" << " |" << std::endl;
    std::cout << sep << std::endl;

    for (const MethodResult* m : methods) {
        std::cout << "| " << std::setw(20) << std::left << m->label << std::right
                  << std::fixed << std::setprecision(3)
                  << " |" << std::setw(9) << m->mean_ms
                  << " |" << std::setw(9) << m->median_ms
                  << " |" << std::setw(9) << m->min_ms
                  << " |" << std::setw(9) << m->max_ms
                  << " |" << std::setw(9) << m->stddev_ms
                  << " |" << std::endl;
    }
    std::cout << sep << std::endl;

    // Full list of individual timings, wrapped at 10 values per line.
    for (const MethodResult* m : methods) {
        std::cout << "  " << m->label << " runs (ms):" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        for (size_t i = 0; i < m->run_ms.size(); i++) {
            if (i % 10 == 0) std::cout << "   ";
            std::cout << std::setw(9) << m->run_ms[i];
            if (i % 10 == 9 || i + 1 == m->run_ms.size()) std::cout << std::endl;
        }
    }
}

void print_table_header() {
    std::cout << "+-------------+----------+----------+----------+-----------------------+-----------------------------------------+-----------------------------------------+" << std::endl;
    std::cout << "| Matrix Size | Matrices | WS 12spl | WS 16spl | Native GEMM           | Ozaki II (12 splits)                    | Ozaki II (16 splits)                    |" << std::endl;
    std::cout << "+-------------+----------+----------+----------+-----------------------+-----------------------------------------+-----------------------------------------+" << std::endl;
    std::cout << "|             |   (MB)   |   (MB)   |   (MB)   |  mean TF (min-max)    |  mean TF (min-max)      | Accuracy      |  mean TF (min-max)      | Accuracy      |" << std::endl;
    std::cout << "+-------------+----------+----------+----------+-----------------------+-------------------------+---------------+-------------------------+---------------+" << std::endl;
}

void print_table_row(size_t n, const BenchmarkResult& r) {
    // A, B and C are resident for the A*B product.
    size_t matrix_mem = 3 * n * n * sizeof(double);
    size_t ozaki_ws_12 = gemmul8::workSize<false, gemmul8::Backend::INT8>(n, n, n, MODULI_LOW);
    size_t ozaki_ws_16 = gemmul8::workSize<false, gemmul8::Backend::INT8>(n, n, n, MODULI_HIGH);
    double matrix_mb = matrix_mem / (1024.0 * 1024.0);
    double ozaki_mb_12 = ozaki_ws_12 / (1024.0 * 1024.0);
    double ozaki_mb_16 = ozaki_ws_16 / (1024.0 * 1024.0);

    auto perf = [](const MethodResult& m) {
        std::ostringstream os;
        os << std::fixed << std::setprecision(2) << m.mean_tflops
           << " (" << m.min_tflops << "-" << m.max_tflops << ")";
        return os.str();
    };

    std::cout << "| " << std::setw(11) << n
              << " | " << std::setw(8) << std::fixed << std::setprecision(1) << matrix_mb
              << " | " << std::setw(8) << std::setprecision(1) << ozaki_mb_12
              << " | " << std::setw(8) << std::setprecision(1) << ozaki_mb_16
              << " | " << std::setw(21) << perf(r.native)
              << " | " << std::setw(23) << perf(r.ozaki12)
              << " | " << std::setw(13) << std::scientific << std::setprecision(2) << r.ozaki12.error
              << " | " << std::setw(23) << perf(r.ozaki16)
              << " | " << std::setw(13) << std::scientific << std::setprecision(2) << r.ozaki16.error
              << " |" << std::endl;
}

void print_table_footer() {
    std::cout << "+-------------+----------+----------+----------+-----------------------+-------------------------+---------------+-------------------------+---------------+" << std::endl;
}

void print_summary_table(const std::vector<size_t>& sizes, const std::vector<BenchmarkResult>& results) {
    std::cout << std::endl;
    std::cout << "==================================================================================================================================================" << std::endl;
    std::cout << "                                                        SUMMARY TABLE  (C = A * B, A != B)" << std::endl;
    std::cout << "                              Condition Number: 10^" << LOG10_COND
              << "   |   " << NUM_WARMUP << " warmup + " << NUM_RUNS << " timed runs per method" << std::endl;
    std::cout << "==================================================================================================================================================" << std::endl;
    std::cout << std::endl;

    print_table_header();
    for (size_t i = 0; i < sizes.size(); i++) {
        print_table_row(sizes[i], results[i]);
    }
    print_table_footer();

    std::cout << std::endl;
    std::cout << "Legend:" << std::endl;
    std::cout << "  - Matrices: Memory for A, B and C (A*B computation, two distinct inputs)" << std::endl;
    std::cout << "  - WS 12spl: Ozaki-II workspace memory for 12 splits" << std::endl;
    std::cout << "  - WS 16spl: Ozaki-II workspace memory for 16 splits" << std::endl;
    std::cout << "  - Performance in TFLOPS: mean over " << NUM_RUNS
              << " runs, with (slowest-fastest) run in parentheses" << std::endl;
    std::cout << "  - Accuracy: Relative Frobenius error vs native FP64" << std::endl;
    std::cout << "==================================================================================================================================================" << std::endl;
}

void print_latex_table(const std::vector<size_t>& sizes, const std::vector<BenchmarkResult>& results) {
    std::cout << std::endl;
    std::cout << "% ---- LaTeX table (copy into your document) ----" << std::endl;
    std::cout << "\\begin{table}[htbp]" << std::endl;
    std::cout << "  \\centering" << std::endl;
    std::cout << "  \\caption{GEMMul8 Ozaki~II performance and workspace for $C = AB$ with $A \\neq B$, "
              << "both $\\kappa = 10^{" << LOG10_COND << "}$. Performance is the mean of " << NUM_RUNS
              << " timed runs after " << NUM_WARMUP
              << " warmup iterations; speedup is Ozaki~II throughput divided by native FP64 throughput.}" << std::endl;
    std::cout << "  \\label{tab:ab_size_ozaki}" << std::endl;
    std::cout << "  \\begin{tabular}{rrrrrrrr}" << std::endl;
    std::cout << "    \\toprule" << std::endl;
    std::cout << "    & Native & \\multicolumn{3}{c}{Ozaki~II (12 moduli)} & \\multicolumn{3}{c}{Ozaki~II (16 moduli)} \\\\" << std::endl;
    std::cout << "    \\cmidrule(lr){3-5} \\cmidrule(lr){6-8}" << std::endl;
    std::cout << "    Size & TFLOP/s & TFLOP/s & Workspace (MB) & Speedup & TFLOP/s & Workspace (MB) & Speedup \\\\" << std::endl;
    std::cout << "    \\midrule" << std::endl;

    for (size_t i = 0; i < sizes.size(); i++) {
        size_t n = sizes[i];
        const BenchmarkResult& r = results[i];
        double ws12_mb = gemmul8::workSize<false, gemmul8::Backend::INT8>(n, n, n, MODULI_LOW) / (1024.0 * 1024.0);
        double ws16_mb = gemmul8::workSize<false, gemmul8::Backend::INT8>(n, n, n, MODULI_HIGH) / (1024.0 * 1024.0);
        double speedup12 = r.ozaki12.mean_tflops / r.native.mean_tflops;
        double speedup16 = r.ozaki16.mean_tflops / r.native.mean_tflops;

        std::cout << "    " << n
                  << " & " << std::fixed << std::setprecision(2) << r.native.mean_tflops
                  << " & " << r.ozaki12.mean_tflops
                  << " & " << std::setprecision(1) << ws12_mb
                  << " & " << std::setprecision(2) << speedup12 << "$\\times$"
                  << " & " << r.ozaki16.mean_tflops
                  << " & " << std::setprecision(1) << ws16_mb
                  << " & " << std::setprecision(2) << speedup16 << "$\\times$"
                  << " \\\\" << std::endl;
    }

    std::cout << "    \\bottomrule" << std::endl;
    std::cout << "  \\end{tabular}" << std::endl;
    std::cout << "\\end{table}" << std::endl;
    std::cout << "% ---- end LaTeX table ----" << std::endl;
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

    std::cerr << "C = A * B with A != B, both cond = 10^" << LOG10_COND << std::endl;
    std::cerr << NUM_WARMUP << " warmup iterations + " << NUM_RUNS
              << " timed runs per method (native FP64, Ozaki-II " << MODULI_LOW
              << ", Ozaki-II " << MODULI_HIGH << ")" << std::endl;
    std::cerr << "Matrix sizes: ";
    for (size_t s : sizes) std::cerr << s << " ";
    std::cerr << std::endl << std::endl;

    std::vector<BenchmarkResult> results;
    for (size_t n : sizes) {
        std::cerr << "Processing " << n << "x" << n << "..." << std::endl;
        BenchmarkResult result = run_benchmark(n);
        results.push_back(result);
        print_runs_table(n, result);
        std::cerr << std::endl;
    }

    print_summary_table(sizes, results);
    print_latex_table(sizes, results);

    return 0;
}
