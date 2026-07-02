#pragma once

#include <algorithm>
#include <cstdlib>
#include <iostream>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusolverDn.h>

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__         \
                      << " code=" << err << " \"" << cudaGetErrorString(err)     \
                      << "\"" << std::endl;                                      \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

#define CUBLAS_CHECK(call)                                                       \
    do {                                                                         \
        cublasStatus_t status = (call);                                          \
        if (status != CUBLAS_STATUS_SUCCESS) {                                    \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__        \
                      << " status=" << status << std::endl;                      \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

#define CUSOLVER_CHECK(call)                                                     \
    do {                                                                         \
        cusolverStatus_t status = (call);                                        \
        if (status != CUSOLVER_STATUS_SUCCESS) {                                 \
            std::cerr << "cuSOLVER error at " << __FILE__ << ":" << __LINE__      \
                      << " status=" << status << std::endl;                      \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

#define CURAND_CHECK(call)                                                       \
    do {                                                                         \
        curandStatus_t status = (call);                                          \
        if (status != CURAND_STATUS_SUCCESS) {                                   \
            std::cerr << "cuRAND error at " << __FILE__ << ":" << __LINE__       \
                      << " status=" << status << std::endl;                      \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

/** QR + orgqr to form the thin Q factor (same role as rocsolver_dgeqrf + rocsolver_dorgqr). */
inline void cusolver_thin_orthogonal_q(cusolverDnHandle_t cusolver, int m, int n, double* A,
                                       int lda, double* tau) {
    if (m <= 0 || n <= 0) {
        return;
    }
    int lwork_geqrf = 0;
    CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(cusolver,
        m,
        n,
        A,
        lda,
        &lwork_geqrf
    ));
    int lwork_orgqr = 0;
    CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(cusolver,
        m,
        n,
        n,
        A,
        lda,
        tau,
        &lwork_orgqr
    ));
    const int lwork_buf = std::max(lwork_geqrf,
        lwork_orgqr
    );
    double* d_work = nullptr;
    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_work,
        sizeof(double) * std::max(1,
            lwork_buf
        )
    ));
    CUDA_CHECK(cudaMalloc(&d_info,
        sizeof(int)
    ));
    CUSOLVER_CHECK(cusolverDnDgeqrf(cusolver,
        m,
        n,
        A,
        lda,
        tau,
        d_work,
        lwork_geqrf,
        d_info
    ));
    CUSOLVER_CHECK(cusolverDnDorgqr(cusolver,
        m,
        n,
        n,
        A,
        lda,
        tau,
        d_work,
        lwork_orgqr,
        d_info
    ));
    // cuSOLVER calls are asynchronous; free'ing d_work before completion corrupts the factorization.
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_info));
}
