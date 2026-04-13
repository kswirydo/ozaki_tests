/**
 * Native cuBLAS GEMM benchmark (no emulation / Ozaki).
 * For each problem size N (square: M=N=K), builds a random double matrix with condition
 * number ~10^8, runs C = A * A, copies C to the host, and prints D(1,1), timing, and TFLOPS.
 *
 * Warmup: exactly the first problem size in the list runs 10 GEMM kernel calls before timing.
 */

#include "cuda_conditioned_matrix.cuh"
#include <cuda_profiler_api.h>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {
constexpr int kNumWarmupFirstOnly = 10;
constexpr int kNumTimedIters = 50;
constexpr int kLog10Cond = 8;

struct SizeResult {
    int n;
    double tflops;
};

void print_summary_table(const std::vector<SizeResult>& rows) {
    if (rows.empty()) {
        return;
    }
    constexpr int kColN = 12;
    constexpr int kColTf = 14;
    const auto line = [&] {
        std::cout << '+' << std::string(kColN + 2,
            '-'
        ) << '+' << std::string(kColTf + 2,
            '-'
        ) << '+' << '\n';
    };
    std::cout << std::endl;
    std::cout << "Native GEMM (double, C = A * A; cond(A) ~ 10^" << kLog10Cond << ")\n";
    line();
    std::cout << "| " << std::setw(kColN) << std::left << "N" << " | " << std::setw(kColTf) << std::left << "TFLOPS"
              << " |\n";
    line();
    for (const auto& r : rows) {
        std::cout << "| " << std::setw(kColN) << std::left << r.n << " | " << std::setw(kColTf) << std::left
                  << std::fixed << std::setprecision(3) << r.tflops << " |\n";
    }
    line();
}
}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <n1> [n2] ..." << std::endl;
        std::cerr << "  Each n is the dimension for square GEMM C = A * A (column-major).\n";
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
    std::vector<SizeResult> summary;
    summary.reserve(sizes.size());

    for (size_t si = 0; si < sizes.size(); si++) {
        const int n = sizes[si];
        const size_t nn = static_cast<size_t>(n) * static_cast<size_t>(n);
        const size_t bytes = nn * sizeof(double);

        double *d_A = nullptr;
        double *d_C = nullptr;
        CUDA_CHECK(cudaMalloc(&d_A,
            bytes
        ));
        CUDA_CHECK(cudaMalloc(&d_C,
            bytes
        ));

        std::cerr << "N=" << n << ": generating conditioned matrix (cond≈10^" << kLog10Cond << ")..."
                  << std::flush;
        generate_conditioned_matrix_cuda(cublas,
            cusolver,
            gen,
            static_cast<size_t>(n),
            static_cast<size_t>(n),
            kLog10Cond,
            d_A
        );
        std::cerr << " done\n";

        if (si == 0) {
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

        cudaProfilerStart();
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
        cudaProfilerStop();
	CUDA_CHECK(cudaDeviceSynchronize());
        const auto t1 = std::chrono::high_resolution_clock::now();

        const double elapsed = std::chrono::duration<double>(t1 - t0).count();
        const double time_per = elapsed / static_cast<double>(kNumTimedIters);
        const double tflops = (2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n)) /
                              (time_per * 1e12);

        std::vector<double> h_C(nn);
        CUDA_CHECK(cudaMemcpy(h_C.data(),
            d_C,
            bytes,
            cudaMemcpyDeviceToHost
        ));

        const double d11 = h_C[0];  // D(1,1), column-major storage

        summary.push_back({n, tflops});

        std::cerr << std::fixed << std::setprecision(6) << "N=" << n << "  D(1,1)=" << d11
                  << "  time=" << time_per * 1e3 << " ms/iter"
                  << "  TFLOPS=" << std::setprecision(3) << tflops;
        if (si == 0) {
            std::cerr << "  [warmup " << kNumWarmupFirstOnly << " GEMMs applied before timing]";
        }
        std::cerr << std::endl;

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_C));
    }

    print_summary_table(summary);

    CURAND_CHECK(curandDestroyGenerator(gen));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver));
    CUBLAS_CHECK(cublasDestroy(cublas));
    return 0;
}
