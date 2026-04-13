/**
 * Single GEMM benchmark — CUDA port of ../single_gemm_benchmark.cu
 *
 * Compile:
 *   nvcc -O3 -std=c++17 single_gemm_benchmark.cu -o single_gemm_benchmark \
 *        -I${GEMMUL8_PATH}/include -L${GEMMUL8_PATH}/lib -lgemmul8 \
 *        -lcublas -lcusolver -lcurand -lcudart
 */

#include "cuda_conditioned_matrix.cuh"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "gemmul8.hpp"

static const int NUM_WARMUP = 40;
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

double compute_max_abs_error(const double* C_ref, const double* C_test, size_t n) {
    double max_err = 0.0;
    for (size_t i = 0; i < n; i++) {
        double err = std::abs(C_ref[i] - C_test[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <M> <N> <K>" << std::endl;
        return 1;
    }
    size_t M = std::stoull(argv[1]);
    size_t N = std::stoull(argv[2]);
    size_t K = std::stoull(argv[3]);
    if (M < 128 || N < 128 || K < 128) {
        std::cerr << "Error: All dimensions must be at least 128" << std::endl;
        return 1;
    }
    size_t size_A = M * K;
    size_t size_B = K * N;
    size_t size_C = M * N;
    size_t mem_total = (size_A + size_B + 2 * size_C) * sizeof(double) + gemmul8::workSize(M,
        N,
        K,
        16
    );
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem,
        &total_mem
    ));
    if (mem_total > free_mem * 0.9) {
        std::cerr << "Error: Not enough GPU memory!" << std::endl;
        return 1;
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
    double *d_A, *d_B, *d_C_native, *d_C_ozaki;
    void* d_work;
    CUDA_CHECK(cudaMalloc(&d_A,
        size_A * sizeof(double)
    ));
    CUDA_CHECK(cudaMalloc(&d_B,
        size_B * sizeof(double)
    ));
    CUDA_CHECK(cudaMalloc(&d_C_native,
        size_C * sizeof(double)
    ));
    CUDA_CHECK(cudaMalloc(&d_C_ozaki,
        size_C * sizeof(double)
    ));
    CUDA_CHECK(cudaMalloc(&d_work,
        gemmul8::workSize(M,
            N,
            K,
            16
        )
    ));
    std::cout << "Generating matrices..." << std::flush;
    generate_conditioned_matrix_cuda(cublas,
        cusolver,
        gen,
        M,
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
    std::cout << " done" << std::endl;
    double alpha = 1.0, beta = 0.0;
    std::vector<double> h_C_native(size_C), h_C_ozaki(size_C);
    std::cout << "Running Native FP64 GEMM..." << std::flush;
    for (int i = 0; i < NUM_WARMUP; i++) {
        CUBLAS_CHECK(cublasDgemm(cublas,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            (int)M,
            (int)N,
            (int)K,
            &alpha,
            d_A,
            (int)M,
            d_B,
            (int)K,
            &beta,
            d_C_native,
            (int)M
        ));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        CUBLAS_CHECK(cublasDgemm(cublas,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            (int)M,
            (int)N,
            (int)K,
            &alpha,
            d_A,
            (int)M,
            d_B,
            (int)K,
            &beta,
            d_C_native,
            (int)M
        ));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    double native_time = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
    double native_tflops = (2.0 * M * N * K) / (native_time * 1e12);
    std::cout << " " << std::fixed << std::setprecision(2) << native_tflops << " TFLOPS" << std::endl;
    CUDA_CHECK(cudaMemcpy(h_C_native.data(),
        d_C_native,
        size_C * sizeof(double),
        cudaMemcpyDeviceToHost
    ));
    auto bench_ozaki = [&](int moduli, double& tflops, double& frel, double& maxabs) {
        for (int i = 0; i < NUM_WARMUP; i++) {
            gemmul8::gemm<double>(cublas,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                M,
                N,
                K,
                &alpha,
                d_A,
                M,
                d_B,
                K,
                &beta,
                d_C_ozaki,
                M,
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
                M,
                N,
                K,
                &alpha,
                d_A,
                M,
                d_B,
                K,
                &beta,
                d_C_ozaki,
                M,
                moduli,
                false,
                d_work
            );
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        end = std::chrono::high_resolution_clock::now();
        double t = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
        tflops = (2.0 * M * N * K) / (t * 1e12);
        CUDA_CHECK(cudaMemcpy(h_C_ozaki.data(),
            d_C_ozaki,
            size_C * sizeof(double),
            cudaMemcpyDeviceToHost
        ));
        frel = compute_frobenius_rel_error(h_C_native.data(),
            h_C_ozaki.data(),
            size_C
        );
        maxabs = compute_max_abs_error(h_C_native.data(),
            h_C_ozaki.data(),
            size_C
        );
    };
    double ozaki12_tflops = 0, ozaki12_error = 0, ozaki12_max_err = 0;
    double ozaki16_tflops = 0, ozaki16_error = 0, ozaki16_max_err = 0;
    std::cout << "Running Ozaki-II (12 splits)..." << std::flush;
    bench_ozaki(12,
        ozaki12_tflops,
        ozaki12_error,
        ozaki12_max_err
    );
    std::cout << " " << ozaki12_tflops << " TFLOPS" << std::endl;
    std::cout << "Running Ozaki-II (16 splits)..." << std::flush;
    bench_ozaki(16,
        ozaki16_tflops,
        ozaki16_error,
        ozaki16_max_err
    );
    std::cout << " " << ozaki16_tflops << " TFLOPS" << std::endl;
    std::cout << "\nNative: " << native_tflops << " TFLOPS\nOzaki 12: " << ozaki12_tflops << " TF, err=" << std::scientific
              << ozaki12_error << "\nOzaki 16: " << std::fixed << ozaki16_tflops << " TF, err=" << std::scientific
              << ozaki16_error << std::endl;
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_native));
    CUDA_CHECK(cudaFree(d_C_ozaki));
    CURAND_CHECK(curandDestroyGenerator(gen));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver));
    CUBLAS_CHECK(cublasDestroy(cublas));
    return 0;
}
