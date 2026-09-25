/**
 * Table benchmark — CUDA port of ../table_benchmark.cu
 *
 * Compile (use -arch matching your GPU, e.g. sm_100 for Blackwell; see ../Makefile):
 *   nvcc -arch=sm_100 -O3 -std=c++17 table_benchmark.cu -o table_benchmark \
 *        -I${GEMMUL8_PATH}/include -L${GEMMUL8_PATH}/lib -lgemmul8 \
 *        -lcublas -lcusolver -lcurand -lcudart
 */

#include "cuda_conditioned_matrix.cuh"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "gemmul8.hpp"
#include "gemmul8_helpers.cuh"

static const int NUM_WARMUP = 10;
static const int NUM_ITERATIONS = 50;
static const int LOG10_COND = 8;

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
    cublasHandle_t cublas;
    cusolverDnHandle_t cusolver;
    curandGenerator_t gen;
    CUBLAS_CHECK(cublasCreate(&cublas));
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
    size_t worksize = gemmul8_gemm_worksize(n,
        n,
        n,
        16
    );
    void* d_work;
    CUDA_CHECK(cudaMalloc(&d_work,
        worksize
    ));
    for (int moduli : {12, 16}) {
        std::cerr << "  Benchmarking Ozaki-II (" << moduli << " moduli)..." << std::flush;
        for (int i = 0; i < NUM_WARMUP; i++) {
            gemmul8::gemm<double>(cublas,
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
                moduli,
                false,
                d_work
            );
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            gemmul8::gemm<double>(cublas,
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
                moduli,
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
        if (moduli == 12) {
            result.ozaki12_tflops = tflops;
            result.ozaki12_error = error;
        } else {
            result.ozaki16_tflops = tflops;
            result.ozaki16_error = error;
        }
        std::cerr << " " << tflops << " TFLOPS, error=" << error << std::endl;
    }
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C_native));
    CUDA_CHECK(cudaFree(d_C_ozaki));
    CURAND_CHECK(curandDestroyGenerator(gen));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver));
    CUBLAS_CHECK(cublasDestroy(cublas));
    return result;
}

void print_table_header() {
    std::cout << "+-------------+----------+----------+----------+-------------+-----------------------------+-----------------------------+"
              << std::endl;
    std::cout << "| Matrix Size | Matrices | WS 12spl | WS 16spl | Native GEMM | Ozaki II (12 splits)        | Ozaki II (16 splits)        |"
              << std::endl;
    std::cout << "+-------------+----------+----------+----------+-------------+-----------------------------+-----------------------------+"
              << std::endl;
    std::cout << "|             |   (MB)   |   (MB)   |   (MB)   | Performance | Performance | Accuracy      | Performance | Accuracy      |"
              << std::endl;
    std::cout << "+-------------+----------+----------+----------+-------------+-------------+---------------+-------------+---------------+"
              << std::endl;
}

void print_table_row(size_t n, const BenchmarkResult& r) {
    size_t matrix_mem = 2 * n * n * sizeof(double);
    size_t ozaki_ws_12 = gemmul8_gemm_worksize(n,
        n,
        n,
        12
    );
    size_t ozaki_ws_16 = gemmul8_gemm_worksize(n,
        n,
        n,
        16
    );
    double matrix_mb = matrix_mem / (1024.0 * 1024.0);
    double ozaki_mb_12 = ozaki_ws_12 / (1024.0 * 1024.0);
    double ozaki_mb_16 = ozaki_ws_16 / (1024.0 * 1024.0);
    std::cout << "| " << std::setw(11) << n << " | " << std::setw(8) << std::fixed << std::setprecision(1) << matrix_mb
              << " | " << std::setw(8) << std::setprecision(1) << ozaki_mb_12 << " | " << std::setw(8) << std::setprecision(1)
              << ozaki_mb_16 << " | " << std::setw(8) << std::setprecision(2) << r.native_tflops << " TF"
              << " | " << std::setw(8) << std::setprecision(2) << r.ozaki12_tflops << " TF"
              << " | " << std::scientific << std::setprecision(2) << r.ozaki12_error << " | " << std::fixed << std::setw(8)
              << std::setprecision(2) << r.ozaki16_tflops << " TF"
              << " | " << std::scientific << std::setprecision(2) << r.ozaki16_error << " |" << std::endl;
}

void print_table_footer() {
    std::cout << "+-------------+----------+----------+----------+-------------+-------------+---------------+-------------+---------------+"
              << std::endl;
}

void print_summary_table(const std::vector<size_t>& sizes, const std::vector<BenchmarkResult>& results) {
    std::cout << std::endl;
    std::cout << "======================================================================================================================"
              << std::endl;
    std::cout << "                                                  SUMMARY TABLE (CUDA)" << std::endl;
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
    std::vector<BenchmarkResult> results;
    for (size_t n : sizes) {
        std::cerr << "Processing " << n << "x" << n << "..." << std::endl;
        results.push_back(run_benchmark(n));
        std::cerr << std::endl;
    }
    print_summary_table(sizes,
        results
    );
    return 0;
}
