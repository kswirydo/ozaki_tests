#include "cuda_conditioned_matrix.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusolverDn.h>

__global__ void test_pow_kernel(double* out) {
    out[0] = pow(10.0, 0.0);
    out[1] = pow(10.0, 8.0);
    out[2] = (128 > 1) ? ((double)127 / (double)(128 - 1) * 8.0) : 0.0;
    out[3] = pow(10.0, out[2]);
}

static double host_norm_sq(const double* p, size_t len) {
    long double s = 0;
    for (size_t i = 0; i < len; ++i)
        s += (long double)p[i] * (long double)p[i];
    return (double)s;
}

int main() {
    const int n = 128;
    const size_t rows = (size_t)n, cols = (size_t)n;
    const size_t min_dim = std::min(rows, cols);
    const size_t size_U = rows * min_dim;
    const size_t size_V = cols * min_dim;
    const size_t size_S = min_dim * min_dim;
    const size_t size_A = rows * cols;

    cublasHandle_t cublas;
    cusolverDnHandle_t cusolver;
    curandGenerator_t gen;
    cublasCreate(&cublas);
    cusolverDnCreate(&cusolver);
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 12345ULL);

    double *d_U, *d_V, *d_S, *d_temp, *d_tau, *d_sv, *d_A;
    cudaMalloc(&d_U, size_U * sizeof(double));
    cudaMalloc(&d_V, size_V * sizeof(double));
    cudaMalloc(&d_S, size_S * sizeof(double));
    cudaMalloc(&d_temp, rows * min_dim * sizeof(double));
    cudaMalloc(&d_tau, min_dim * sizeof(double));
    cudaMalloc(&d_sv, min_dim * sizeof(double));
    cudaMalloc(&d_A, size_A * sizeof(double));

    {
        double *d_test;
        cudaMalloc(&d_test, 4 * sizeof(double));
        test_pow_kernel<<<1, 1>>>(d_test);
        cudaDeviceSynchronize();
        double ht[4];
        cudaMemcpy(ht, d_test, sizeof(ht), cudaMemcpyDeviceToHost);
        std::printf("device pow(10,0)=%g pow(10,8)=%g exp=%g pow(10,exp)=%g\n", ht[0], ht[1], ht[2], ht[3]);
        cudaFree(d_test);
    }

    curandGenerateNormalDouble(gen, d_U, size_U, 0.0, 1.0);
    curandGenerateNormalDouble(gen, d_V, size_V, 0.0, 1.0);
    cudaDeviceSynchronize();

    std::vector<double> h_U(size_U), h_V(size_V);
    cudaMemcpy(h_U.data(), d_U, size_U * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_V.data(), d_V, size_V * sizeof(double), cudaMemcpyDeviceToHost);
    std::printf("after curand ||U||_F^2=%g ||V||_F^2=%g\n", host_norm_sq(h_U.data(), size_U),
                host_norm_sq(h_V.data(), size_V));

    cusolver_thin_orthogonal_q(cusolver, (int)rows, (int)min_dim, d_U, (int)rows, d_tau);
    cudaMemcpy(h_U.data(), d_U, size_U * sizeof(double), cudaMemcpyDeviceToHost);
    std::printf("after QR U ||U||_F^2=%g\n", host_norm_sq(h_U.data(), size_U));

    cusolver_thin_orthogonal_q(cusolver, (int)cols, (int)min_dim, d_V, (int)cols, d_tau);
    cudaMemcpy(h_V.data(), d_V, size_V * sizeof(double), cudaMemcpyDeviceToHost);
    std::printf("after QR V ||V||_F^2=%g\n", host_norm_sq(h_V.data(), size_V));

    const int block_size = 256;
    size_t num_blocks = (size_S + block_size - 1) / block_size;
    cm_zero_matrix_kernel<<<num_blocks, block_size>>>(d_S, size_S);
    num_blocks = (min_dim + block_size - 1) / block_size;
    cm_compute_sv_kernel<<<num_blocks, block_size>>>(d_sv, min_dim, 8.0);
    cudaDeviceSynchronize();
    std::vector<double> hsv(min_dim);
    cudaMemcpy(hsv.data(), d_sv, min_dim * sizeof(double), cudaMemcpyDeviceToHost);
    std::printf("after sv kernel sv[0]=%g sv[last]=%g\n", hsv[0], hsv[min_dim - 1]);

    cm_set_diagonal_kernel<<<num_blocks, block_size>>>(d_S, min_dim, min_dim, min_dim, d_sv);
    cudaDeviceSynchronize();

    std::vector<double> hS(size_S);
    cudaMemcpy(hS.data(), d_S, size_S * sizeof(double), cudaMemcpyDeviceToHost);
    std::printf("after S diag ||S||_F^2=%g S[0]=%g S[end]=%g\n", host_norm_sq(hS.data(), size_S), hS[0],
                hS[min_dim * min_dim - 1]);

    double alpha = 1.0, beta = 0.0;
    cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, (int)rows, (int)min_dim, (int)min_dim, &alpha, d_U, (int)rows, d_S,
                (int)min_dim, &beta, d_temp, (int)rows);
    cudaDeviceSynchronize();
    std::vector<double> hT(rows * min_dim);
    cudaMemcpy(hT.data(), d_temp, rows * min_dim * sizeof(double), cudaMemcpyDeviceToHost);
    std::printf("after US ||temp||_F^2=%g\n", host_norm_sq(hT.data(), rows * min_dim));

    cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, (int)rows, (int)cols, (int)min_dim, &alpha, d_temp, (int)rows, d_V,
                (int)cols, &beta, d_A, (int)rows);
    cudaDeviceSynchronize();

    std::vector<double> hA(size_A);
    cudaMemcpy(hA.data(), d_A, size_A * sizeof(double), cudaMemcpyDeviceToHost);
    std::printf("final ||A||_F^2=%g maxabs=%g\n", host_norm_sq(hA.data(), size_A),
                *std::max_element(hA.begin(), hA.end()));

    cudaFree(d_U);
    cudaFree(d_V);
    cudaFree(d_S);
    cudaFree(d_temp);
    cudaFree(d_tau);
    cudaFree(d_sv);
    cudaFree(d_A);
    curandDestroyGenerator(gen);
    cusolverDnDestroy(cusolver);
    cublasDestroy(cublas);
    return 0;
}
