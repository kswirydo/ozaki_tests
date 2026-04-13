#pragma once

#include <algorithm>
#include <cmath>

#include "cuda_helpers.cuh"

__global__ void cm_zero_matrix_kernel(double* A, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] = 0.0;
    }
}

__global__ void cm_set_diagonal_kernel(double* S, size_t rows, size_t cols, size_t ld, double* sv) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t min_dim = (rows < cols) ? rows : cols;
    if (idx < min_dim) {
        S[idx * ld + idx] = sv[idx];
    }
}

__global__ void cm_compute_sv_kernel(double* sv, size_t n, double log10_cond) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double exponent = (n > 1) ? ((double)idx / (double)(n - 1) * log10_cond) : 0.0;
        sv[idx] = pow(10.0,
            exponent
        );
    }
}

/** Device column-major A(rows,cols) with 2-norm condition number ~10^log10_cond */
inline void generate_conditioned_matrix_cuda(cublasHandle_t cublas, cusolverDnHandle_t cusolver,
                                             curandGenerator_t gen, size_t rows, size_t cols, int log10_cond,
                                             double* d_A) {
    size_t min_dim = std::min(rows,
        cols
    );
    size_t size_U = rows * min_dim;
    size_t size_V = cols * min_dim;
    size_t size_S = min_dim * min_dim;
    double *d_U, *d_V, *d_S, *d_temp, *d_tau, *d_sv;
    CUDA_CHECK(cudaMalloc(&d_U,
        size_U * sizeof(double)
    ));
    CUDA_CHECK(cudaMalloc(&d_V,
        size_V * sizeof(double)
    ));
    CUDA_CHECK(cudaMalloc(&d_S,
        size_S * sizeof(double)
    ));
    CUDA_CHECK(cudaMalloc(&d_temp,
        rows * min_dim * sizeof(double)
    ));
    CUDA_CHECK(cudaMalloc(&d_tau,
        min_dim * sizeof(double)
    ));
    CUDA_CHECK(cudaMalloc(&d_sv,
        min_dim * sizeof(double)
    ));
    CURAND_CHECK(curandGenerateNormalDouble(gen,
        d_U,
        size_U,
        0.0,
        1.0
    ));
    CURAND_CHECK(curandGenerateNormalDouble(gen,
        d_V,
        size_V,
        0.0,
        1.0
    ));
    // cuRAND fills are asynchronous; cuSOLVER must not read U/V until generation completes.
    CUDA_CHECK(cudaDeviceSynchronize());
    cusolver_thin_orthogonal_q(cusolver,
        (int)rows,
        (int)min_dim,
        d_U,
        (int)rows,
        d_tau
    );
    cusolver_thin_orthogonal_q(cusolver,
        (int)cols,
        (int)min_dim,
        d_V,
        (int)cols,
        d_tau
    );
    const int block_size = 256;
    size_t num_blocks = (size_S + block_size - 1) / block_size;
    cm_zero_matrix_kernel<<<num_blocks, block_size>>>(d_S, size_S);
    num_blocks = (min_dim + block_size - 1) / block_size;
    cm_compute_sv_kernel<<<num_blocks, block_size>>>(d_sv, min_dim, (double)log10_cond);
    cm_set_diagonal_kernel<<<num_blocks, block_size>>>(d_S, min_dim, min_dim, min_dim, d_sv);
    CUDA_CHECK(cudaDeviceSynchronize());
    double alpha = 1.0, beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(cublas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        (int)rows,
        (int)min_dim,
        (int)min_dim,
        &alpha,
        d_U,
        (int)rows,
        d_S,
        (int)min_dim,
        &beta,
        d_temp,
        (int)rows
    ));
    CUBLAS_CHECK(cublasDgemm(cublas,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        (int)rows,
        (int)cols,
        (int)min_dim,
        &alpha,
        d_temp,
        (int)rows,
        d_V,
        (int)cols,
        &beta,
        d_A,
        (int)rows
    ));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_tau));
    CUDA_CHECK(cudaFree(d_sv));
}
