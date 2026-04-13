/**
 * Native cuBLAS GEMM benchmark with a sweep of matrix condition numbers.
 * Similar to table_benchmark_native: square GEMM C = A * A (column-major), conditioned A.
 * For each problem size N, iterates log10(cond(A)) from 1 to 16.
 *
 * Correctness: CPU double GEMM reference vs GPU; relative Frobenius error matches table_benchmark.cu:
 *   sqrt(sum (C_ref - C_gpu)^2) / sqrt(sum C_ref^2)
 *
 * Warmup: the first (N, cond) pair runs kNumWarmupFirstOnly GEMMs before timing.
 */

#include "cuda_conditioned_matrix.cuh"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kNumWarmupFirstOnly = 10;
constexpr int kNumTimedIters = 50;
constexpr int kLog10CondMin = 1;
constexpr int kLog10CondMax = 16;

/** Same formula as table_benchmark.cu::compute_frobenius_rel_error */
double compute_frobenius_rel_error(const double* C_ref, const double* C_test, size_t num_elements) {
    double diff_sum = 0.0, norm_sum = 0.0;
    for (size_t i = 0; i < num_elements; i++) {
        double diff = C_ref[i] - C_test[i];
        diff_sum += diff * diff;
        norm_sum += C_ref[i] * C_ref[i];
    }
    return std::sqrt(diff_sum) / std::sqrt(norm_sum);
}

/** Column-major C = A * B (double), lda = ldb = ldc = n (square n x n). */
void dgemm_cpu_colmajor_nn(int n, const double* A, const double* B, double* C) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i + k * n] * B[k + j * n];
            }
            C[i + j * n] = sum;
        }
    }
}

struct CondResult {
    int log10_cond;
    double tflops;
    double frob_rel_err;
    double d11;
};

void print_summary_table(int n, const std::vector<CondResult>& rows) {
    if (rows.empty()) {
        return;
    }
    constexpr int kColC = 14;
    constexpr int kColTf = 14;
    constexpr int kColErr = 16;
    constexpr int kColD = 18;
    const auto line = [&] {
        std::cout << '+' << std::string(kColC + 2,
            '-'
        ) << '+' << std::string(kColTf + 2,
            '-'
        ) << '+'
                  << std::string(kColErr + 2,
                      '-'
                  ) << '+' << std::string(kColD + 2,
                      '-'
                  ) << '+' << '\n';
    };
    std::cout << std::endl;
    std::cout << "N=" << n << "  Native GEMM (double, C = A * A) vs CPU ref; cond(A) ~ 10^k, k=" << kLog10CondMin
              << ".." << kLog10CondMax << '\n';
    line();
    std::cout << "| " << std::setw(kColC) << std::left << "log10(cond)" << " | " << std::setw(kColTf) << std::left
              << "TFLOPS"
              << " | " << std::setw(kColErr) << std::left << "Frob rel err"
              << " | " << std::setw(kColD) << std::left << "D(1,1) GPU"
              << " |\n";
    line();
    for (const auto& r : rows) {
        std::cout << "| " << std::setw(kColC) << std::left << r.log10_cond << " | " << std::setw(kColTf) << std::left
                  << std::fixed << std::setprecision(3) << r.tflops << " | " << std::setw(kColErr) << std::left
                  << std::scientific << std::setprecision(4) << r.frob_rel_err << " | " << std::setw(kColD) << std::left
                  << std::fixed << std::setprecision(6) << r.d11 << " |\n";
    }
    line();
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <n1> [n2] ..." << std::endl;
        std::cerr << "  Each n is the dimension for square GEMM C = A * A (column-major).\n";
        std::cerr << "  For each n, runs cond(A) with log10(cond) = " << kLog10CondMin << ".." << kLog10CondMax << ".\n";
        return 1;
    }

    std::vector<int> sizes;
    for (int i = 1; i < argc; i++) {
        long long v = std::strtoll(argv[i],
            nullptr,
            10
        );
        if (v <= 0) {
            std::cerr << "Invalid size: " << argv[i] << std::endl;
            return 1;
        }
        sizes.push_back(static_cast<int>(v));
    }

    cublasHandle_t cublas{};
    cusolverDnHandle_t cusolver{};
    curandGenerator_t gen{};
    CUBLAS_CHECK(cublasCreate(&cublas));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
    CURAND_CHECK(curandCreateGenerator(&gen,
        CURAND_RNG_PSEUDO_DEFAULT
    ));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen,
        12345ULL
    ));

    const double alpha = 1.0;
    const double beta = 0.0;

    bool first_case = true;

    for (size_t si = 0; si < sizes.size(); si++) {
        const int n = sizes[si];
        const size_t nn = static_cast<size_t>(n) * static_cast<size_t>(n);
        const size_t bytes = nn * sizeof(double);

        std::vector<CondResult> summary;
        summary.reserve(static_cast<size_t>(kLog10CondMax - kLog10CondMin + 1));

        for (int log10_cond = kLog10CondMin; log10_cond <= kLog10CondMax; log10_cond++) {
            double* d_A = nullptr;
            double* d_C = nullptr;
            CUDA_CHECK(cudaMalloc(&d_A,
                bytes
            ));
            CUDA_CHECK(cudaMalloc(&d_C,
                bytes
            ));

            std::cerr << "N=" << n << "  log10(cond)=" << log10_cond << ": generating conditioned matrix..."
                      << std::flush;
            generate_conditioned_matrix_cuda(cublas,
                cusolver,
                gen,
                static_cast<size_t>(n),
                static_cast<size_t>(n),
                log10_cond,
                d_A
            );
            std::cerr << " done\n";

            if (first_case) {
                for (int w = 0; w < kNumWarmupFirstOnly; w++) {
                    CUBLAS_CHECK(cublasDgemm(cublas,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        n,
                        n,
                        n,
                        &alpha,
                        d_A,
                        n,
                        d_A,
                        n,
                        &beta,
                        d_C,
                        n
                    ));
                }
                CUDA_CHECK(cudaDeviceSynchronize());
            }

            CUDA_CHECK(cudaDeviceSynchronize());
            const auto t0 = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < kNumTimedIters; it++) {
                CUBLAS_CHECK(cublasDgemm(cublas,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    n,
                    n,
                    n,
                    &alpha,
                    d_A,
                    n,
                    d_A,
                    n,
                    &beta,
                    d_C,
                    n
                ));
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            const auto t1 = std::chrono::high_resolution_clock::now();

            const double elapsed = std::chrono::duration<double>(t1 - t0).count();
            const double time_per = elapsed / static_cast<double>(kNumTimedIters);
            const double tflops =
                (2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n)) /
                (time_per * 1e12);

            std::vector<double> h_A(nn);
            std::vector<double> h_C_gpu(nn);
            std::vector<double> h_C_cpu(nn);
            CUDA_CHECK(cudaMemcpy(h_A.data(),
                d_A,
                bytes,
                cudaMemcpyDeviceToHost
            ));
            CUDA_CHECK(cudaMemcpy(h_C_gpu.data(),
                d_C,
                bytes,
                cudaMemcpyDeviceToHost
            ));

            dgemm_cpu_colmajor_nn(n,
                h_A.data(),
                h_A.data(),
                h_C_cpu.data()
            );
            const double frob_err = compute_frobenius_rel_error(h_C_cpu.data(),
                h_C_gpu.data(),
                nn
            );
            const double d11 = h_C_gpu[0];

            summary.push_back({log10_cond, tflops, frob_err, d11});

            std::cerr << std::fixed << std::setprecision(6) << "  D(1,1)=" << d11 << "  time=" << time_per * 1e3
                      << " ms/iter"
                      << "  TFLOPS=" << std::setprecision(3) << tflops << "  CPU-GPU Frob rel err=" << std::scientific
                      << std::setprecision(4) << frob_err;
            if (first_case) {
                std::cerr << "  [warmup " << kNumWarmupFirstOnly << " GEMMs before timing]";
            }
            std::cerr << std::endl;

            first_case = false;

            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_C));
        }

        print_summary_table(n,
            summary
        );
    }

    CURAND_CHECK(curandDestroyGenerator(gen));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver));
    CUBLAS_CHECK(cublasDestroy(cublas));
    return 0;
}
