/**
 * Outer dimension benchmark — CUDA port of ../outer_dim_benchmark.cu
 *
 * Compile:
 *   nvcc -O3 -std=c++17 outer_dim_benchmark.cu -o outer_dim_benchmark \
 *        -I${GEMMUL8_PATH}/include -L${GEMMUL8_PATH}/lib -lgemmul8 \
 *        -lcublas -lcusolver -lcurand -lcudart
 */

#include "cuda_conditioned_matrix.cuh"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "gemmul8.hpp"

static const int NUM_WARMUP = 2;
static const int NUM_ITERATIONS = 50;
static const int LOG10_COND = 8;
static const size_t N_MULTIPLIER_MAX = 40;

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
    size_t N;
    size_t K;
    double native_tflops;
    double ozaki12_tflops;
    double ozaki12_error;
    double ozaki16_tflops;
    double ozaki16_error;
};

bool run_benchmark(size_t N, size_t K, cublasHandle_t cublas, cusolverDnHandle_t cusolver, curandGenerator_t gen,
                   BenchmarkResult& result) {
    result = {N, K, 0, 0, 0, 0, 0};
    size_t size_A = N * K;
    size_t size_B = K * N;
    size_t size_C = N * N;
    size_t mem_matrices = (size_A + size_B + 2 * size_C) * sizeof(double);
    size_t mem_workspace = gemmul8::workSize(N,
        N,
        K,
        16
    );
    size_t mem_total = mem_matrices + mem_workspace;
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem,
        &total_mem
    ));
    if (mem_total > free_mem * 0.9) {
        std::cerr << "  WARNING: Not enough GPU memory!" << std::endl;
        return false;
    }
    double *d_A = nullptr, *d_B = nullptr, *d_C_native = nullptr, *d_C_ozaki = nullptr;
    void* d_work = nullptr;
    if (cudaMalloc(&d_A,
        size_A * sizeof(double)
    ) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(&d_B,
        size_B * sizeof(double)
    ) != cudaSuccess) {
        cudaFree(d_A);
        return false;
    }
    if (cudaMalloc(&d_C_native,
        size_C * sizeof(double)
    ) != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        return false;
    }
    if (cudaMalloc(&d_C_ozaki,
        size_C * sizeof(double)
    ) != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C_native);
        return false;
    }
    std::cerr << "  Generating matrices A(" << N << "x" << K << ") and B(" << K << "x" << N << ")..." << std::flush;
    generate_conditioned_matrix_cuda(cublas,
        cusolver,
        gen,
        N,
        K,
        LOG10_COND,
        d_A
    );
    generate_conditioned_matrix_cuda(cublas,
        cusolver,
        gen,
        K,
        N,
        LOG10_COND,
        d_B
    );
    std::cerr << " done" << std::endl;
    double alpha = 1.0, beta = 0.0;
    std::cerr << "  Benchmarking native GEMM..." << std::flush;
    for (int i = 0; i < NUM_WARMUP; i++) {
        CUBLAS_CHECK(cublasDgemm(cublas,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            (int)N,
            (int)N,
            (int)K,
            &alpha,
            d_A,
            (int)N,
            d_B,
            (int)K,
            &beta,
            d_C_native,
            (int)N
        ));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        CUBLAS_CHECK(cublasDgemm(cublas,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            (int)N,
            (int)N,
            (int)K,
            &alpha,
            d_A,
            (int)N,
            d_B,
            (int)K,
            &beta,
            d_C_native,
            (int)N
        ));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    double native_time = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
    result.native_tflops = (2.0 * N * N * K) / (native_time * 1e12);
    std::cerr << " " << result.native_tflops << " TFLOPS" << std::endl;
    std::vector<double> h_C_native(size_C), h_C_ozaki(size_C);
    CUDA_CHECK(cudaMemcpy(h_C_native.data(),
        d_C_native,
        size_C * sizeof(double),
        cudaMemcpyDeviceToHost
    ));
    size_t worksize = gemmul8::workSize(N,
        N,
        K,
        16
    );
    if (cudaMalloc(&d_work,
        worksize
    ) != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C_native);
        cudaFree(d_C_ozaki);
        return false;
    }
    for (int moduli : {12, 16}) {
        std::cerr << "  Benchmarking Ozaki-II (" << moduli << " moduli)..." << std::flush;
        for (int i = 0; i < NUM_WARMUP; i++) {
            gemmul8::gemm<double>(cublas,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N,
                N,
                K,
                &alpha,
                d_A,
                N,
                d_B,
                K,
                &beta,
                d_C_ozaki,
                N,
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
                N,
                N,
                K,
                &alpha,
                d_A,
                N,
                d_B,
                K,
                &beta,
                d_C_ozaki,
                N,
                moduli,
                false,
                d_work
            );
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        end = std::chrono::high_resolution_clock::now();
        double ozaki_time = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
        double tflops = (2.0 * N * N * K) / (ozaki_time * 1e12);
        CUDA_CHECK(cudaMemcpy(h_C_ozaki.data(),
            d_C_ozaki,
            size_C * sizeof(double),
            cudaMemcpyDeviceToHost
        ));
        double error = compute_frobenius_rel_error(h_C_native.data(),
            h_C_ozaki.data(),
            size_C
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
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_native));
    CUDA_CHECK(cudaFree(d_C_ozaki));
    return true;
}

void print_table_header() {
    std::cout << "+-----------------------+-------------+-----------------------------+-----------------------------+"
              << std::endl;
    std::cout << "| Dimensions (NxKxN)    | Native GEMM | Ozaki II (12 splits)        | Ozaki II (16 splits)        |"
              << std::endl;
    std::cout << "+-----------------------+-------------+-----------------------------+-----------------------------+"
              << std::endl;
    std::cout << "|                       | Performance | Performance | Accuracy      | Performance | Accuracy      |"
              << std::endl;
    std::cout << "+-----------------------+-------------+-------------+---------------+-------------+---------------+"
              << std::endl;
}

void print_table_row(const BenchmarkResult& r) {
    std::ostringstream dims;
    dims << r.N << "x" << r.K << "x" << r.N;
    std::cout << "| " << std::setw(21) << dims.str() << " | " << std::setw(8) << std::fixed << std::setprecision(2)
              << r.native_tflops << " TF"
              << " | " << std::setw(8) << std::setprecision(2) << r.ozaki12_tflops << " TF"
              << " | " << std::scientific << std::setprecision(2) << r.ozaki12_error << " | " << std::fixed << std::setw(8)
              << std::setprecision(2) << r.ozaki16_tflops << " TF"
              << " | " << std::scientific << std::setprecision(2) << r.ozaki16_error << " |" << std::endl;
}

void print_table_footer() {
    std::cout << "+-----------------------+-------------+-------------+---------------+-------------+---------------+"
              << std::endl;
}

void print_summary_table(size_t K, const std::vector<BenchmarkResult>& results) {
    std::cout << std::endl;
    std::cout << "================================================================================================"
              << std::endl;
    std::cout << "                           SUMMARY TABLE (CUDA) — outer dimension sweep, K=" << K << std::endl;
    std::cout << "================================================================================================"
              << std::endl;
    print_table_header();
    for (const auto& r : results) {
        print_table_row(r);
    }
    print_table_footer();
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <K>" << std::endl;
        return 1;
    }
    size_t K = std::stoull(argv[1]);
    if (K < 128) {
        std::cerr << "Error: K must be at least 128" << std::endl;
        return 1;
    }
    std::vector<size_t> N_values;
    for (size_t mult = 1; mult <= N_MULTIPLIER_MAX; mult++) {
        N_values.push_back(mult * K);
    }
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
    std::vector<BenchmarkResult> results;
    bool oom = false;
    for (size_t N : N_values) {
        BenchmarkResult res;
        if (!run_benchmark(N,
            K,
            cublas,
            cusolver,
            gen,
            res
        )) {
            oom = true;
            break;
        }
        results.push_back(res);
    }
    CURAND_CHECK(curandDestroyGenerator(gen));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver));
    CUBLAS_CHECK(cublasDestroy(cublas));
    if (oom && results.empty()) {
        return 1;
    }
    print_summary_table(K,
        results
    );
    return 0;
}
