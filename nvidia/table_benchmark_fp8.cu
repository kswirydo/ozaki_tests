/**
 * Table benchmark — FP8 emulation path (cuBLASLt + Ozaki-II), same workload as table_benchmark.cu.
 *
 * Uses GEMMul8's FP8 backend (see sibling ../GEMMul8/GEMMul8/sample/dgemm_cuBLASLt_fp8.cu).
 *
 * Build (from this directory): make table_benchmark_fp8
 *   or: nvcc -arch=sm_100 -O3 -std=c++17 table_benchmark_fp8.cu -o table_benchmark_fp8 \
 *        -I${GEMMUL8_PATH}/include -L${GEMMUL8_PATH}/lib -lgemmul8 \
 *        -lcublas -lcublasLt -lcusolver -lcurand -lcudart
 */

#include "cuda_conditioned_matrix.cuh"

#include <chrono>
#include <cublasLt.h>
#include <iomanip>
#include <iostream>
#include <vector>

#include "gemmul8.hpp"

#define CUBLASLT_CHECK(call)                                                                 \
    do {                                                                                     \
        cublasStatus_t status = (call);                                                      \
        if (status != CUBLAS_STATUS_SUCCESS) {                                               \
            std::cerr << "cuBLASLt error at " << __FILE__ << ":" << __LINE__ << " status="   \
                      << status << std::endl;                                                \
            std::exit(1);                                                                    \
        }                                                                                    \
    } while (0)

static constexpr gemmul8::Backend kEmuBackend = gemmul8::Backend::FP8;

static const int NUM_WARMUP = 10;
static const int NUM_ITERATIONS = 50;
static const int LOG10_COND = 8;
static const int OZAKI_MODULI[] = {10, 12};

double compute_frobenius_rel_error(const double* C_ref, const double* C_test, size_t n) {
    double diff_sum = 0.0, norm_sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = C_ref[i] - C_test[i];
        diff_sum += diff * diff;
        norm_sum += C_ref[i] * C_ref[i];
    }
    return std::sqrt(diff_sum) / std::sqrt(norm_sum);
}

struct BenchmarkResultFp8 {
    double native_tflops;
    double ozaki10_tflops;
    double ozaki10_error;
    double ozaki12_tflops;
    double ozaki12_error;
};

BenchmarkResultFp8 run_benchmark_fp8(size_t n) {
    BenchmarkResultFp8 result = {0, 0, 0, 0, 0};
    size_t size = n * n;
    size_t bytes = size * sizeof(double);
    cublasHandle_t cublas;
    cublasLtHandle_t cublas_lt;
    cusolverDnHandle_t cusolver;
    curandGenerator_t gen;
    CUBLAS_CHECK(cublasCreate(&cublas));
    CUBLASLT_CHECK(cublasLtCreate(&cublas_lt));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
    CURAND_CHECK(curandCreateGenerator(&gen,
        CURAND_RNG_PSEUDO_DEFAULT
    ));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen,
        12345ULL
    ));
    double *d_A, *d_C_native, *d_C_ozaki;
    CUDA_CHECK(cudaMalloc(&d_A,
        bytes
    ));
    CUDA_CHECK(cudaMalloc(&d_C_native,
        bytes
    ));
    CUDA_CHECK(cudaMalloc(&d_C_ozaki,
        bytes
    ));
    std::cerr << "  Generating " << n << "x" << n << " matrix (cond=10^" << LOG10_COND << ")..." << std::flush;
    generate_conditioned_matrix_cuda(cublas,
        cusolver,
        gen,
        n,
        n,
        LOG10_COND,
        d_A
    );
    std::cerr << " done" << std::endl;
    double alpha = 1.0, beta = 0.0;
    std::cerr << "  Benchmarking native GEMM..." << std::flush;
    for (int i = 0; i < NUM_WARMUP; i++) {
        CUBLAS_CHECK(cublasDgemm(cublas,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            (int)n,
            (int)n,
            (int)n,
            &alpha,
            d_A,
            (int)n,
            d_A,
            (int)n,
            &beta,
            d_C_native,
            (int)n
        ));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        CUBLAS_CHECK(cublasDgemm(cublas,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            (int)n,
            (int)n,
            (int)n,
            &alpha,
            d_A,
            (int)n,
            d_A,
            (int)n,
            &beta,
            d_C_native,
            (int)n
        ));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    double native_time = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
    result.native_tflops = (2.0 * n * n * n) / (native_time * 1e12);
    std::cerr << " " << result.native_tflops << " TFLOPS" << std::endl;
    std::vector<double> h_C_native(size), h_C_ozaki(size);
    CUDA_CHECK(cudaMemcpy(h_C_native.data(),
        d_C_native,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    for (int moduli : OZAKI_MODULI) {
        std::cerr << "  Benchmarking Ozaki-II FP8 (" << moduli << " moduli)..." << std::flush;
        size_t worksize = gemmul8::workSize<false, kEmuBackend>(n,
            n,
            n,
            static_cast<unsigned>(moduli)
        );
        void* d_work;
        CUDA_CHECK(cudaMalloc(&d_work,
            worksize
        ));
        for (int i = 0; i < NUM_WARMUP; i++) {
            gemmul8::gemmLt<double, kEmuBackend>(cublas_lt,
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
                d_C_ozaki,
                n,
                static_cast<unsigned>(moduli),
                false,
                d_work
            );
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            gemmul8::gemmLt<double, kEmuBackend>(cublas_lt,
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
                d_C_ozaki,
                n,
                static_cast<unsigned>(moduli),
                false,
                d_work
            );
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        end = std::chrono::high_resolution_clock::now();
        double ozaki_time = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
        double tflops = (2.0 * n * n * n) / (ozaki_time * 1e12);
        CUDA_CHECK(cudaMemcpy(h_C_ozaki.data(),
            d_C_ozaki,
            bytes,
            cudaMemcpyDeviceToHost
        ));
        double error = compute_frobenius_rel_error(h_C_native.data(),
            h_C_ozaki.data(),
            size
        );
        if (moduli == OZAKI_MODULI[0]) {
            result.ozaki10_tflops = tflops;
            result.ozaki10_error = error;
        } else {
            result.ozaki12_tflops = tflops;
            result.ozaki12_error = error;
        }
        std::cerr << " " << tflops << " TFLOPS, error=" << error << std::endl;
        CUDA_CHECK(cudaFree(d_work));
    }
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C_native));
    CUDA_CHECK(cudaFree(d_C_ozaki));
    CURAND_CHECK(curandDestroyGenerator(gen));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver));
    CUBLASLT_CHECK(cublasLtDestroy(cublas_lt));
    CUBLAS_CHECK(cublasDestroy(cublas));
    return result;
}

void print_table_header() {
    std::cout << "+-------------+----------+----------+----------+-------------+-----------------------------+-----------------------------+"
              << std::endl;
    std::cout << "| Matrix Size | Matrices | WS 10spl | WS 12spl | Native GEMM | Ozaki FP8 (10 splits)        | Ozaki FP8 (12 splits)        |"
              << std::endl;
    std::cout << "+-------------+----------+----------+----------+-------------+-----------------------------+-----------------------------+"
              << std::endl;
    std::cout << "|             |   (MB)   |   (MB)   |   (MB)   | Performance | Performance | Accuracy      | Performance | Accuracy      |"
              << std::endl;
    std::cout << "+-------------+----------+----------+----------+-------------+-------------+---------------+-------------+---------------+"
              << std::endl;
}

void print_table_row(size_t n, const BenchmarkResultFp8& r) {
    size_t matrix_mem = 2 * n * n * sizeof(double);
    size_t ozaki_ws_10 = gemmul8::workSize<false, kEmuBackend>(n,
        n,
        n,
        OZAKI_MODULI[0]
    );
    size_t ozaki_ws_12 = gemmul8::workSize<false, kEmuBackend>(n,
        n,
        n,
        OZAKI_MODULI[1]
    );
    double matrix_mb = matrix_mem / (1024.0 * 1024.0);
    double ozaki_mb_10 = ozaki_ws_10 / (1024.0 * 1024.0);
    double ozaki_mb_12 = ozaki_ws_12 / (1024.0 * 1024.0);
    std::cout << "| " << std::setw(11) << n << " | " << std::setw(8) << std::fixed << std::setprecision(1) << matrix_mb
              << " | " << std::setw(8) << std::setprecision(1) << ozaki_mb_10 << " | " << std::setw(8) << std::setprecision(1)
              << ozaki_mb_12 << " | " << std::setw(8) << std::setprecision(2) << r.native_tflops << " TF"
              << " | " << std::setw(8) << std::setprecision(2) << r.ozaki10_tflops << " TF"
              << " | " << std::scientific << std::setprecision(2) << r.ozaki10_error << " | " << std::fixed << std::setw(8)
              << std::setprecision(2) << r.ozaki12_tflops << " TF"
              << " | " << std::scientific << std::setprecision(2) << r.ozaki12_error << " |" << std::endl;
}

void print_table_footer() {
    std::cout << "+-------------+----------+----------+----------+-------------+-------------+---------------+-------------+---------------+"
              << std::endl;
}

void print_summary_table(const std::vector<size_t>& sizes, const std::vector<BenchmarkResultFp8>& results) {
    std::cout << std::endl;
    std::cout << "======================================================================================================================"
              << std::endl;
    std::cout << "                              SUMMARY TABLE — FP8 emulation (cuBLASLt), CUDA" << std::endl;
    std::cout << "                                             Condition Number: 10^" << LOG10_COND << std::endl;
    std::cout << "======================================================================================================================"
              << std::endl;
    std::cout << std::endl;
    print_table_header();
    for (size_t i = 0; i < sizes.size(); i++) {
        print_table_row(sizes[i],
            results[i]
        );
    }
    print_table_footer();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <size1> [size2] ..." << std::endl;
        return 1;
    }
    std::vector<size_t> sizes;
    for (int i = 1; i < argc; i++) {
        sizes.push_back(std::stoull(argv[i]));
    }
    std::vector<BenchmarkResultFp8> results;
    for (size_t n : sizes) {
        std::cerr << "Processing " << n << "x" << n << "..." << std::endl;
        results.push_back(run_benchmark_fp8(n));
        std::cerr << std::endl;
    }
    print_summary_table(sizes,
        results
    );
    return 0;
}
