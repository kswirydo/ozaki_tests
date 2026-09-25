/**
 * A/B Table LaTeX Benchmark (CUDA)
 * --------------------------------
 * For each requested matrix size N, generates two DISTINCT N x N matrices A and
 * B, each with 2-norm condition number ~1e3, computes C = A * B with several
 * methods, prints on-screen run statistics + summary tables, and finally a
 * LaTeX table (booktabs) with one row per size.
 *
 * Methods (accuracy measured vs native FP64 reference):
 *   1. Native FP64                (cublasDgemm)
 *   2. INT8 Ozaki-II, 12 moduli   (gemmul8::gemm,   Backend::INT8)
 *   3. INT8 Ozaki-II, 16 moduli   (gemmul8::gemm,   Backend::INT8)
 *   4. FP8  Ozaki-II, 10 moduli   (gemmul8::gemmLt, Backend::FP8)
 *   5. FP8  Ozaki-II, 12 moduli   (gemmul8::gemmLt, Backend::FP8)
 *
 * Relative Frobenius error: ||C_method - C_fp64||_F / ||C_fp64||_F
 *
 * A and B are made distinct by seeding the RNG differently for each.
 * The condition number is fixed at 1e3 (log10_cond = 3).
 *
 * Build (from this directory):
 *   make ab_table_latex_benchmark
 *
 * Usage:
 *   ./ab_table_latex_benchmark [--tex out.tex] <size1> [size2] [size3] ...
 *     e.g. ./ab_table_latex_benchmark 1024 2048 4096 8192
 *   With no sizes given, defaults to 4096.
 */

#include "cuda_helpers.cuh"
#include "cuda_conditioned_matrix.cuh"

#include <cublasLt.h>
#include "gemmul8.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#define CUBLASLT_CHECK(call)                                                     \
    do {                                                                         \
        cublasStatus_t status = (call);                                          \
        if (status != CUBLAS_STATUS_SUCCESS) {                                   \
            std::cerr << "cuBLASLt error at " << __FILE__ << ":" << __LINE__     \
                      << " status=" << status << std::endl;                      \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

static constexpr int kLog10Cond   = 3;      // condition number 1e3
static constexpr int kNumWarmup   = 10;
static constexpr int kNumTimed    = 50;     // timed runs per method
static constexpr bool kFastMode   = false;  // Ozaki accurate mode

static constexpr int kInt8Moduli12 = 12;
static constexpr int kInt8Moduli16 = 16;
static constexpr int kFp8Moduli10  = 10;
static constexpr int kFp8Moduli12  = 12;

struct MethodResult {
    std::string name;
    std::vector<double> run_ms;   // kNumTimed individual timings
    double mean_ms      = 0.0;
    double median_ms    = 0.0;
    double min_ms       = 0.0;
    double max_ms       = 0.0;
    double stddev_ms    = 0.0;
    double mean_tflops  = 0.0;
    double min_tflops   = 0.0;
    double max_tflops   = 0.0;
    double frob_rel_err = 0.0;    // relative Frobenius error vs native FP64
    double workspace_mb = 0.0;
    bool   is_reference = false;
};

struct BenchmarkResult {
    int n;
    std::vector<MethodResult> methods;   // [FP64, INT8-12, INT8-16, FP8-10, FP8-12]
};

void summarize(MethodResult& m, int n) {
    const double flop = 2.0 * (double)n * (double)n * (double)n;
    const size_t count = m.run_ms.size();
    if (count == 0) return;

    m.mean_ms = std::accumulate(m.run_ms.begin(), m.run_ms.end(), 0.0) / count;
    m.min_ms  = *std::min_element(m.run_ms.begin(), m.run_ms.end());
    m.max_ms  = *std::max_element(m.run_ms.begin(), m.run_ms.end());

    std::vector<double> sorted = m.run_ms;
    std::sort(sorted.begin(), sorted.end());
    m.median_ms = (count % 2 == 1) ? sorted[count / 2]
                                   : 0.5 * (sorted[count / 2 - 1] + sorted[count / 2]);

    double sq = 0.0;
    for (double ms : m.run_ms) sq += (ms - m.mean_ms) * (ms - m.mean_ms);
    m.stddev_ms = (count > 1) ? std::sqrt(sq / (count - 1)) : 0.0;

    m.mean_tflops = flop / (m.mean_ms * 1e-3 * 1e12);
    m.max_tflops  = flop / (m.min_ms * 1e-3 * 1e12);
    m.min_tflops  = flop / (m.max_ms * 1e-3 * 1e12);
}

double compute_frobenius_rel_error(const double* C_ref, const double* C_test, size_t n) {
    double diff_sum = 0.0, norm_sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = C_ref[i] - C_test[i];
        diff_sum += diff * diff;
        norm_sum += C_ref[i] * C_ref[i];
    }
    if (norm_sum <= 0.0) return 0.0;
    return std::sqrt(diff_sum) / std::sqrt(norm_sum);
}

/** Record kNumTimed synchronized cublasDgemm timings (ms) into run_ms; copy last C to host. */
void time_dgemm(cublasHandle_t h, int n, const double* d_A, const double* d_B, double* d_C,
                double alpha, double beta, std::vector<double>& run_ms, std::vector<double>& h_out) {
    const size_t size_C = static_cast<size_t>(n) * static_cast<size_t>(n);
    run_ms.clear();
    run_ms.reserve(kNumTimed);
    for (int i = 0; i < kNumTimed; i++) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        CUBLAS_CHECK(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_A, n, d_B, n, &beta,
                                 d_C, n));
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        run_ms.push_back(std::chrono::duration<double>(end - start).count() * 1000.0);
    }
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_C, size_C * sizeof(double), cudaMemcpyDeviceToHost));
}

void time_int8_ozaki(cublasHandle_t h, int n, double* d_A, double* d_B, double* d_C, double alpha,
                     double beta, int moduli, void* d_work, std::vector<double>& run_ms,
                     std::vector<double>& h_out) {
    const size_t size_C = static_cast<size_t>(n) * static_cast<size_t>(n);
    run_ms.clear();
    run_ms.reserve(kNumTimed);
    for (int i = 0; i < kNumTimed; i++) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        gemmul8::gemm<double, gemmul8::Backend::INT8>(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_A,
                                                      n, d_B, n, &beta, d_C, n, moduli, kFastMode, d_work);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        run_ms.push_back(std::chrono::duration<double>(end - start).count() * 1000.0);
    }
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_C, size_C * sizeof(double), cudaMemcpyDeviceToHost));
}

void time_fp8_ozaki(cublasLtHandle_t hlt, int n, double* d_A, double* d_B, double* d_C, double alpha,
                    double beta, int moduli, void* d_work, std::vector<double>& run_ms,
                    std::vector<double>& h_out) {
    const size_t size_C = static_cast<size_t>(n) * static_cast<size_t>(n);
    run_ms.clear();
    run_ms.reserve(kNumTimed);
    for (int i = 0; i < kNumTimed; i++) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        gemmul8::gemmLt<double, gemmul8::Backend::FP8>(hlt, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                                                       d_A, n, d_B, n, &beta, d_C, n, moduli, kFastMode,
                                                       d_work);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        run_ms.push_back(std::chrono::duration<double>(end - start).count() * 1000.0);
    }
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_C, size_C * sizeof(double), cudaMemcpyDeviceToHost));
}

/** Generate A, B (distinct, both cond 1e3), warm up, and time all methods for one size. */
BenchmarkResult run_benchmark(int n, cublasHandle_t cublas, cublasLtHandle_t cublaslt,
                              cusolverDnHandle_t cusolver, curandGenerator_t gen) {
    const size_t size  = static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t bytes = size * sizeof(double);
    const double alpha = 1.0, beta = 0.0;
    const double mb = 1024.0 * 1024.0;

    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    std::cerr << "  Generating A (" << n << "x" << n << ", cond 1e" << kLog10Cond << ")..." << std::flush;
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 12345ULL));
    generate_conditioned_matrix_cuda(cublas, cusolver, gen, n, n, kLog10Cond, d_A);
    std::cerr << " B..." << std::flush;
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 67890ULL));
    generate_conditioned_matrix_cuda(cublas, cusolver, gen, n, n, kLog10Cond, d_B);
    std::cerr << " done" << std::endl;

    const size_t ws_int8_12 = gemmul8::workSize<false, gemmul8::Backend::INT8>(n, n, n, kInt8Moduli12);
    const size_t ws_int8_16 = gemmul8::workSize<false, gemmul8::Backend::INT8>(n, n, n, kInt8Moduli16);
    const size_t ws_fp8_10  = gemmul8::workSize<false, gemmul8::Backend::FP8>(n, n, n, kFp8Moduli10);
    const size_t ws_fp8_12  = gemmul8::workSize<false, gemmul8::Backend::FP8>(n, n, n, kFp8Moduli12);
    const size_t ws_max     = std::max({ws_int8_12, ws_int8_16, ws_fp8_10, ws_fp8_12});

    void* d_work;
    CUDA_CHECK(cudaMalloc(&d_work, ws_max));

    std::vector<double> h_C_ref(size), h_C_test(size);

    std::cerr << "  Warmup..." << std::flush;
    for (int w = 0; w < kNumWarmup; w++) {
        CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_A, n, d_B, n,
                                 &beta, d_C, n));
        gemmul8::gemm<double, gemmul8::Backend::INT8>(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                                                      d_A, n, d_B, n, &beta, d_C, n, kInt8Moduli12,
                                                      kFastMode, d_work);
        gemmul8::gemmLt<double, gemmul8::Backend::FP8>(cublaslt, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                                                       d_A, n, d_B, n, &beta, d_C, n, kFp8Moduli10,
                                                       kFastMode, d_work);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cerr << " benchmarking (" << kNumTimed << " runs/method)..." << std::flush;

    BenchmarkResult result;
    result.n = n;

    MethodResult fp64;
    fp64.name = "FP64 (native)";
    fp64.is_reference = true;
    time_dgemm(cublas, n, d_A, d_B, d_C, alpha, beta, fp64.run_ms, h_C_ref);
    summarize(fp64, n);
    result.methods.push_back(fp64);

    auto add_int8 = [&](const char* name, int moduli, size_t ws) {
        MethodResult r;
        r.name = name;
        r.workspace_mb = ws / mb;
        time_int8_ozaki(cublas, n, d_A, d_B, d_C, alpha, beta, moduli, d_work, r.run_ms, h_C_test);
        r.frob_rel_err = compute_frobenius_rel_error(h_C_ref.data(), h_C_test.data(), size);
        summarize(r, n);
        result.methods.push_back(r);
    };
    auto add_fp8 = [&](const char* name, int moduli, size_t ws) {
        MethodResult r;
        r.name = name;
        r.workspace_mb = ws / mb;
        time_fp8_ozaki(cublaslt, n, d_A, d_B, d_C, alpha, beta, moduli, d_work, r.run_ms, h_C_test);
        r.frob_rel_err = compute_frobenius_rel_error(h_C_ref.data(), h_C_test.data(), size);
        summarize(r, n);
        result.methods.push_back(r);
    };

    add_int8("INT8 Ozaki-II (12)", kInt8Moduli12, ws_int8_12);
    add_int8("INT8 Ozaki-II (16)", kInt8Moduli16, ws_int8_16);
    add_fp8("FP8 Ozaki-II (10)", kFp8Moduli10, ws_fp8_10);
    add_fp8("FP8 Ozaki-II (12)", kFp8Moduli12, ws_fp8_12);
    std::cerr << " done" << std::endl;

    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return result;
}

/** Per-method run statistics + full list of individual timings (like the HIP file). */
void print_run_stats_table(const BenchmarkResult& r) {
    const char* sep =
        "+----------------------+----------+----------+----------+----------+----------+";

    std::cout << std::endl;
    std::cout << "Run statistics for " << r.n << "x" << r.n << " (" << kNumTimed
              << " timed runs after " << kNumWarmup << " warmup iterations), times in ms" << std::endl;
    std::cout << sep << std::endl;
    std::cout << "| " << std::setw(20) << std::left << "Method" << std::right << " |"
              << std::setw(9) << "mean" << " |" << std::setw(9) << "median" << " |"
              << std::setw(9) << "min" << " |" << std::setw(9) << "max" << " |"
              << std::setw(9) << "stddev" << " |" << std::endl;
    std::cout << sep << std::endl;

    for (const MethodResult& m : r.methods) {
        std::cout << "| " << std::setw(20) << std::left << m.name << std::right
                  << std::fixed << std::setprecision(3)
                  << " |" << std::setw(9) << m.mean_ms
                  << " |" << std::setw(9) << m.median_ms
                  << " |" << std::setw(9) << m.min_ms
                  << " |" << std::setw(9) << m.max_ms
                  << " |" << std::setw(9) << m.stddev_ms
                  << " |" << std::endl;
    }
    std::cout << sep << std::endl;

    for (const MethodResult& m : r.methods) {
        std::cout << "  " << m.name << " runs (ms):" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        for (size_t i = 0; i < m.run_ms.size(); i++) {
            if (i % 10 == 0) std::cout << "   ";
            std::cout << std::setw(9) << m.run_ms[i];
            if (i % 10 == 9 || i + 1 == m.run_ms.size()) std::cout << std::endl;
        }
    }
}

/** Summary tables (one row per size): throughput/speedup, then accuracy. */
void print_summary_tables(const std::vector<BenchmarkResult>& results, const std::string& device) {
    std::cout << std::endl;
    std::cout << "==================================================================================="
                 "=============="
              << std::endl;
    std::cout << "                              SUMMARY  (C = A * B, A != B)" << std::endl;
    std::cout << "   cond(A) = cond(B) = 10^" << kLog10Cond << "   |   " << kNumWarmup << " warmup + "
              << kNumTimed << " timed runs per method   |   " << device << std::endl;
    std::cout << "==================================================================================="
                 "=============="
              << std::endl;

    // --- Throughput / speedup table ---
    const char* psep =
        "+---------+------------+---------------------+---------------------+---------------------+---------------------+";
    std::cout << "\nThroughput: mean TFLOP/s (speedup vs native FP64)\n";
    std::cout << psep << std::endl;
    std::cout << "| " << std::setw(7) << std::left << "Size" << std::right << " |"
              << std::setw(11) << "Native" << " |"
              << std::setw(20) << "INT8 Ozaki (12)" << " |"
              << std::setw(20) << "INT8 Ozaki (16)" << " |"
              << std::setw(20) << "FP8 Ozaki (10)" << " |"
              << std::setw(20) << "FP8 Ozaki (12)" << " |" << std::endl;
    std::cout << psep << std::endl;
    for (const BenchmarkResult& r : results) {
        const double nat = r.methods[0].mean_tflops;
        std::cout << "| " << std::setw(7) << std::left << r.n << std::right << std::fixed
                  << std::setprecision(2) << " |" << std::setw(11) << nat << " |";
        for (size_t i = 1; i < r.methods.size(); i++) {
            std::ostringstream cell;
            cell << std::fixed << std::setprecision(2) << r.methods[i].mean_tflops << " ("
                 << std::setprecision(2) << (r.methods[i].mean_tflops / nat) << "x)";
            std::cout << std::setw(20) << cell.str() << " |";
        }
        std::cout << std::endl;
    }
    std::cout << psep << std::endl;

    // --- Accuracy + workspace table ---
    const char* asep =
        "+---------+--------------+--------------+--------------+--------------+";
    std::cout << "\nAccuracy: relative Frobenius error vs native FP64\n";
    std::cout << asep << std::endl;
    std::cout << "| " << std::setw(7) << std::left << "Size" << std::right << " |"
              << std::setw(13) << "INT8 (12)" << " |"
              << std::setw(13) << "INT8 (16)" << " |"
              << std::setw(13) << "FP8 (10)" << " |"
              << std::setw(13) << "FP8 (12)" << " |" << std::endl;
    std::cout << asep << std::endl;
    for (const BenchmarkResult& r : results) {
        std::cout << "| " << std::setw(7) << std::left << r.n << std::right;
        for (size_t i = 1; i < r.methods.size(); i++) {
            std::ostringstream cell;
            cell << std::scientific << std::setprecision(2) << r.methods[i].frob_rel_err;
            std::cout << " |" << std::setw(13) << cell.str();
        }
        std::cout << " |" << std::endl;
    }
    std::cout << asep << std::endl;

    std::cout << "\nOzaki-II workspace (MB), independent of size class shown above:" << std::endl;
    if (!results.empty()) {
        const BenchmarkResult& r = results.back();
        std::cout << "  INT8(12)=" << std::fixed << std::setprecision(1) << r.methods[1].workspace_mb
                  << "  INT8(16)=" << r.methods[2].workspace_mb
                  << "  FP8(10)=" << r.methods[3].workspace_mb
                  << "  FP8(12)=" << r.methods[4].workspace_mb << "  (for N=" << r.n << ")" << std::endl;
    }
}

/** Booktabs LaTeX table (one row per size) matching ../ab_table_size_benchmark.cu.
 *  methods order per row: [FP64, INT8-12, INT8-16, FP8-10, FP8-12]. Speedup vs native FP64. */
void write_latex_table(std::ostream& os, const std::vector<BenchmarkResult>& results,
                       const std::string& device) {
    os << "% ---- LaTeX table (copy into your document; needs \\usepackage{booktabs}) ----\n";
    os << "\\begin{table}[htbp]\n";
    os << "  \\centering\n";
    os << "  \\caption{GEMMul8 Ozaki~II performance and workspace for $C = AB$ with $A \\neq B$, "
       << "both $\\kappa = 10^{" << kLog10Cond << "}$ on " << device
       << ". Performance is the mean of " << kNumTimed << " timed runs after " << kNumWarmup
       << " warmup iterations; speedup is Ozaki~II throughput divided by native FP64 throughput.}\n";
    os << "  \\label{tab:ab_table_ozaki_cond1e" << kLog10Cond << "}\n";
    os << "  \\begin{tabular}{rrrrrrrrrrrrrr}\n";
    os << "    \\toprule\n";
    os << "    & Native & \\multicolumn{3}{c}{INT8 Ozaki~II (12 moduli)} "
       << "& \\multicolumn{3}{c}{INT8 Ozaki~II (16 moduli)} "
       << "& \\multicolumn{3}{c}{FP8 Ozaki~II (10 moduli)} "
       << "& \\multicolumn{3}{c}{FP8 Ozaki~II (12 moduli)} \\\\\n";
    os << "    \\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11} \\cmidrule(lr){12-14}\n";
    os << "    Size & TFLOP/s & TFLOP/s & Workspace (MB) & Speedup "
       << "& TFLOP/s & Workspace (MB) & Speedup "
       << "& TFLOP/s & Workspace (MB) & Speedup "
       << "& TFLOP/s & Workspace (MB) & Speedup \\\\\n";
    os << "    \\midrule\n";

    for (const BenchmarkResult& r : results) {
        const double nat = r.methods[0].mean_tflops;
        auto emit_method = [&](const MethodResult& m) {
            os << " & " << std::fixed << std::setprecision(2) << m.mean_tflops
               << " & " << std::setprecision(1) << m.workspace_mb
               << " & " << std::setprecision(2) << (m.mean_tflops / nat) << "$\\times$";
        };
        os << "    " << r.n << " & " << std::fixed << std::setprecision(2) << nat;
        emit_method(r.methods[1]);
        emit_method(r.methods[2]);
        emit_method(r.methods[3]);
        emit_method(r.methods[4]);
        os << " \\\\\n";
    }

    os << "    \\bottomrule\n";
    os << "  \\end{tabular}\n";
    os << "\\end{table}\n";
    os << "% ---- end LaTeX table ----\n";
}

int main(int argc, char* argv[]) {
    std::string tex_path;
    std::vector<int> sizes;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [--tex out.tex] <size1> [size2] ...\n"
                      << "  For each size: generate two distinct matrices with cond = 1e" << kLog10Cond
                      << ", benchmark C = A*B\n  (FP64 + INT8/FP8 Ozaki, " << kNumTimed
                      << " runs each), print run stats, summary tables and a LaTeX table.\n"
                      << "  Default size if none given: 4096.\n";
            return 0;
        }
        if (arg == "--tex" || arg == "-o") {
            if (i + 1 >= argc) {
                std::cerr << "error: " << arg << " requires a filename" << std::endl;
                return 1;
            }
            tex_path = argv[++i];
            continue;
        }
        int s = std::atoi(arg.c_str());
        if (s <= 0) {
            std::cerr << "Invalid size: " << arg << std::endl;
            return 1;
        }
        sizes.push_back(s);
    }
    if (sizes.empty()) sizes.push_back(4096);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::string device = prop.name;
    std::cerr << "Device: " << device << std::endl;
    std::cerr << "C = A * B with A != B, both cond = 10^" << kLog10Cond << std::endl;
    std::cerr << kNumWarmup << " warmup + " << kNumTimed << " timed runs per method (FP64, INT8 "
              << kInt8Moduli12 << "/" << kInt8Moduli16 << ", FP8 " << kFp8Moduli10 << "/"
              << kFp8Moduli12 << ")" << std::endl;
    std::cerr << "Ozaki mode: " << (kFastMode ? "fast" : "accurate") << std::endl;
    std::cerr << "Matrix sizes:";
    for (int s : sizes) std::cerr << " " << s;
    std::cerr << std::endl << std::endl;

    cublasHandle_t     cublas_handle;
    cublasLtHandle_t   cublaslt_handle;
    cusolverDnHandle_t cusolver_handle;
    curandGenerator_t  gen;

    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUBLASLT_CHECK(cublasLtCreate(&cublaslt_handle));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

    std::vector<BenchmarkResult> results;
    for (int n : sizes) {
        std::cerr << "Processing " << n << "x" << n << "..." << std::endl;
        BenchmarkResult r = run_benchmark(n, cublas_handle, cublaslt_handle, cusolver_handle, gen);
        print_run_stats_table(r);
        results.push_back(std::move(r));
    }

    print_summary_tables(results, device);
    std::cout << std::endl;
    write_latex_table(std::cout, results, device);

    if (!tex_path.empty()) {
        std::ofstream tex(tex_path);
        if (!tex.is_open()) {
            std::cerr << "Failed to open " << tex_path << std::endl;
        } else {
            write_latex_table(tex, results, device);
            std::cerr << "Wrote LaTeX table to " << tex_path << std::endl;
        }
    }

    CURAND_CHECK(curandDestroyGenerator(gen));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
    CUBLASLT_CHECK(cublasLtDestroy(cublaslt_handle));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    return 0;
}
