/**
 * Standalone GEMM benchmark — CUDA port of ../standalone_benchmark.cu
 *
 * Compile:
 *   nvcc -O3 -std=c++17 standalone_benchmark.cu -o standalone_benchmark \
 *        -I${GEMMUL8_PATH}/include -L${GEMMUL8_PATH}/lib -lgemmul8 \
 *        -lcublas -lcusolver -lcurand -lcudart
 */

#include "cuda_conditioned_matrix.cuh"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "gemmul8.hpp"

static const int NUM_WARMUP = 5;
static const int NUM_ITERATIONS = 50;
static const int LOG10_COND = 8;
static const int NUM_MODULI = 12;

double compute_frobenius_rel_error(const double* C_ref, const double* C_test, size_t n) {
    double diff_sum = 0.0, norm_sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = C_ref[i] - C_test[i];
        diff_sum += diff * diff;
        norm_sum += C_ref[i] * C_ref[i];
    }
    return std::sqrt(diff_sum) / std::sqrt(norm_sum);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <M> <N> <K>" << std::endl;
        return 1;
    }
    size_t M = std::stoull(argv[1]);
    size_t N = std::stoull(argv[2]);
    size_t K = std::stoull(argv[3]);
    size_t size_A = M * K;
    size_t size_B = K * N;
    size_t size_C = M * N;
    std::cout << "Standalone GEMM Benchmark (CUDA)" << std::endl;
    std::cout << "C(" << M << "x" << N << ") = A(" << M << "x" << K << ") * B(" << K << "x" << N << ")" << std::endl;
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
    double *d_A, *d_B, *d_C;
    void* d_work = nullptr;
    size_t worksize = gemmul8::workSize(M,
        N,
        K,
        NUM_MODULI
    );
    CUDA_CHECK(cudaMalloc(&d_A,
        size_A * sizeof(double)
    ));
    CUDA_CHECK(cudaMalloc(&d_B,
        size_B * sizeof(double)
    ));
    CUDA_CHECK(cudaMalloc(&d_C,
        size_C * sizeof(double)
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
    CUDA_CHECK(cudaMalloc(&d_work,
        worksize
    ));
    double alpha = 1.0, beta = 0.0;
    int block_size = 256;
    size_t num_blocks = (size_C + block_size - 1) / block_size;
    cm_zero_matrix_kernel<<<num_blocks, block_size>>>(d_C, size_C);
    CUDA_CHECK(cudaDeviceSynchronize());
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
            d_C,
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
            d_C,
            (int)M
        ));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    double native_time = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
    double native_tflops = (2.0 * M * N * K) / (native_time * 1e12);
    std::cout << " " << std::fixed << std::setprecision(2) << native_tflops << " TFLOPS" << std::endl;
    std::vector<double> h_C_native(size_C);
    CUDA_CHECK(cudaMemcpy(h_C_native.data(),
        d_C,
        size_C * sizeof(double),
        cudaMemcpyDeviceToHost
    ));
    cm_zero_matrix_kernel<<<num_blocks, block_size>>>(d_C, size_C);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Running Ozaki-II GEMM (" << NUM_MODULI << " moduli)..." << std::flush;
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
            d_C,
            M,
            NUM_MODULI,
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
            d_C,
            M,
            NUM_MODULI,
            false,
            d_work
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    double ozaki_time = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
    double ozaki_tflops = (2.0 * M * N * K) / (ozaki_time * 1e12);
    std::cout << " " << ozaki_tflops << " TFLOPS" << std::endl;
    std::vector<double> h_C_ozaki(size_C);
    CUDA_CHECK(cudaMemcpy(h_C_ozaki.data(),
        d_C,
        size_C * sizeof(double),
        cudaMemcpyDeviceToHost
    ));
    double rel_error = compute_frobenius_rel_error(h_C_native.data(),
        h_C_ozaki.data(),
        size_C
    );
    std::cout << "\nResults:\n  Native FP64:  " << native_tflops << " TFLOPS\n  Ozaki-II:     " << ozaki_tflops
              << " TFLOPS\n  Speedup:      " << (ozaki_tflops / native_tflops) << "x\n  Rel. Error:   "
              << std::scientific << rel_error << std::endl;
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CURAND_CHECK(curandDestroyGenerator(gen));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver));
    CUBLAS_CHECK(cublasDestroy(cublas));
    return 0;
}
