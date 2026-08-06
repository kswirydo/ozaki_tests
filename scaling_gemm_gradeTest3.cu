/**
 * scaling_gemm_gradeTest3
 * -----------------------
 * Dynamic-range (diagonal-scaling) robustness test for Ozaki-II emulation,
 * extending scaling_gemm_gradeTest2 with the FP8 backend.
 *
 *  (1) Generate the SAME two random matrices A, B as gradeTest1/gradeTest2
 *      (default 1024x1024, i.i.d. N(0,1), identical RNG seeds/streams).
 *
 *  (2) Build a scaling vector D = (2^r_1, ..., 2^r_K), where each r_k is an
 *      integer drawn uniformly from [-rexp_range, rexp_range] (default 53).
 *
 *  (3) Scale along the shared (contraction) dimension so the product is preserved:
 *          A_scal[:,k] = 2^{ r_k} * A[:,k]      (A_scal = A * D)
 *          B_scal[k,:] = 2^{-r_k} * B[k,:]      (B_scal = D^{-1} * B)
 *      In exact arithmetic, A_scal * B_scal = A * B.
 *
 *  (4) Compute, with reference C1 = A*B in FP64:
 *          C1        = A      * B        (FP64, hipblasDgemm)
 *          C2        = A_scal * B_scal   (FP64, hipblasDgemm)
 *          C3_int8   = A_scal * B_scal   (INT8 Ozaki-II, `moduli` moduli)
 *          C4_int8   = A      * B        (INT8 Ozaki-II, `moduli` moduli)
 *          C5_fp8    = A_scal * B_scal   (FP8  Ozaki-II, `fp8_moduli` moduli)
 *          C6_fp8    = A      * B        (FP8  Ozaki-II, `fp8_moduli` moduli)
 *      and report relative Frobenius errors against C1, plus the scaled-vs-unscaled
 *      emulation difference for each backend.
 *
 * Build (or `make scaling_gemm_gradeTest3`):
 *   hipcc -O3 -std=c++17 scaling_gemm_gradeTest3.cu -o scaling_gemm_gradeTest3 \
 *       -I/opt/rocm/include -I$(GEMMUL8_PATH)/include \
 *       -L/opt/rocm/lib -L$(GEMMUL8_PATH)/lib -Wl,-rpath,$(GEMMUL8_PATH)/lib \
 *       -lgemmul8 -lhipblas -lamdhip64 -lhipblaslt
 *
 * Usage:
 *   ./scaling_gemm_gradeTest3 [N] [moduli] [seed] [rexp_range] [fp8_moduli]
 *   defaults: N=1024, moduli=16, seed=12345, rexp_range=53, fp8_moduli=12
 */

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#include "gemmul8.hpp"

// GEMMul8 internal profiling flag (see other benchmarks in this repo).
namespace oz2 { bool g_profiling_enabled = false; }

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define HIP_CHECK(call)                                                         \
    do {                                                                        \
        hipError_t err = call;                                                  \
        if (err != hipSuccess) {                                                \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__         \
                      << " code=" << err << " \"" << hipGetErrorString(err)     \
                      << "\"" << std::endl;                                     \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define HIPBLAS_CHECK(call)                                                     \
    do {                                                                        \
        hipblasStatus_t status = call;                                          \
        if (status != HIPBLAS_STATUS_SUCCESS) {                                 \
            std::cerr << "hipBLAS error at " << __FILE__ << ":" << __LINE__     \
                      << " status=" << status << std::endl;                     \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

static const bool FASTMODE = false;

// Column-major indexing (BLAS convention, leading dim = number of rows).
static inline size_t idxA(size_t i, size_t k, size_t N) { return k * N + i; } // A is N x K
static inline size_t idxB(size_t k, size_t j, size_t K) { return j * K + k; } // B is K x M

// Relative Frobenius error ||X - Y||_F / ||X||_F.
static double rel_frob(const std::vector<double>& X, const std::vector<double>& Y) {
    double num = 0.0, den = 0.0;
    for (size_t t = 0; t < X.size(); ++t) {
        double d = X[t] - Y[t];
        num += d * d;
        den += X[t] * X[t];
    }
    return std::sqrt(num) / std::sqrt(den);
}

static double frob(const std::vector<double>& X) {
    double s = 0.0;
    for (double v : X) s += v * v;
    return std::sqrt(s);
}

int main(int argc, char** argv) {
    size_t N = 1024;
    int    moduli = 16;
    unsigned long seed = 12345UL;
    int    rexp_range = 53;
    int    fp8_moduli = 12;

    if (argc > 1) N = std::stoull(argv[1]);
    if (argc > 2) moduli = std::stoi(argv[2]);
    if (argc > 3) seed = std::stoul(argv[3]);
    if (argc > 4) rexp_range = std::stoi(argv[4]);
    if (argc > 5) fp8_moduli = std::stoi(argv[5]);

    const size_t K = N, M = N;
    const size_t size_A = N * K, size_B = K * M, size_C = N * M;

    std::cout << "============================================================\n";
    std::cout << " scaling_gemm_gradeTest3 : diagonal scaling, INT8 + FP8\n";
    std::cout << "============================================================\n";
    std::cout << "  N = K = M      : " << N << "\n";
    std::cout << "  INT8 moduli    : " << moduli << "\n";
    std::cout << "  FP8  moduli    : " << fp8_moduli << "\n";
    std::cout << "  RNG seed       : " << seed << "\n";
    std::cout << "  D_k = 2^r_k, r_k in [" << -rexp_range << ", " << rexp_range
              << "] (integer), applied to shared dim\n\n";

    // ---- (1) Same A, B as gradeTest1/gradeTest2 (no zero-planting) ---------
    std::vector<double> h_A(size_A), h_B(size_B);
    std::mt19937_64 gen_A(seed);
    std::mt19937_64 gen_B(seed + 0x9E3779B97F4A7C15ULL);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (size_t t = 0; t < size_A; ++t) h_A[t] = nd(gen_A);
    for (size_t t = 0; t < size_B; ++t) h_B[t] = nd(gen_B);

    // ---- (2) Scaling vector D = 2^r, r in [-rexp_range, rexp_range] --------
    std::mt19937_64 gen_D(seed + 2024);
    std::uniform_int_distribution<int> rdist(-rexp_range, rexp_range);
    std::vector<double> D(K), Dinv(K);
    std::vector<int> rexp(K);
    for (size_t k = 0; k < K; ++k) {
        rexp[k] = rdist(gen_D);
        D[k]    = std::ldexp(1.0, rexp[k]);   // 2^r_k (exact)
        Dinv[k] = std::ldexp(1.0, -rexp[k]);  // 2^-r_k (exact)
    }
    {
        int rmin = rexp[0], rmax = rexp[0];
        for (int r : rexp) { rmin = std::min(rmin, r); rmax = std::max(rmax, r); }
        std::cout << "Scaling vector D: length " << K
                  << ", exponent range [" << rmin << ", " << rmax << "]\n\n";
    }

    // ---- (3) Build A_scal = A*D (scale col k), B_scal = D^{-1}*B (scale row k)
    std::vector<double> h_As(size_A), h_Bs(size_B);
    for (size_t k = 0; k < K; ++k) {
        double dk = D[k], dik = Dinv[k];
        for (size_t i = 0; i < N; ++i) h_As[idxA(i, k, N)] = h_A[idxA(i, k, N)] * dk;
        for (size_t j = 0; j < M; ++j) h_Bs[idxB(k, j, K)] = h_B[idxB(k, j, K)] * dik;
    }

    // ---- Device setup ------------------------------------------------------
    double *d_A, *d_B, *d_As, *d_Bs, *d_C;
    HIP_CHECK(hipMalloc(&d_A,  size_A * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_B,  size_B * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_As, size_A * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_Bs, size_B * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_C,  size_C * sizeof(double)));
    HIP_CHECK(hipMemcpy(d_A,  h_A.data(),  size_A * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B,  h_B.data(),  size_B * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_As, h_As.data(), size_A * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_Bs, h_Bs.data(), size_B * sizeof(double), hipMemcpyHostToDevice));

    hipDeviceProp_t dprop;
    HIP_CHECK(hipGetDeviceProperties(&dprop, 0));
    std::cout << "Device: " << dprop.name << "\n\n";

    hipblasHandle_t hb;
    HIPBLAS_CHECK(hipblasCreate(&hb));
    hipblasLtHandle_t hblt;
    if (hipblasLtCreate(&hblt) != HIPBLAS_STATUS_SUCCESS) {
        std::cerr << "hipblasLtCreate failed\n";
        return 1;
    }
    const double alpha = 1.0, beta = 0.0;

    std::vector<double> h_C1(size_C), h_C2(size_C), h_C3(size_C), h_C4(size_C),
        h_C5(size_C), h_C6(size_C);

    // ---- (4) C1 = A*B (FP64) ----------------------------------------------
    std::cout << "[1/6] C1 = A * B            (FP64) ..." << std::flush;
    HIPBLAS_CHECK(hipblasDgemm(hb, HIPBLAS_OP_N, HIPBLAS_OP_N, (int)N, (int)M, (int)K,
                               &alpha, d_A, (int)N, d_B, (int)K, &beta, d_C, (int)N));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_C1.data(), d_C, size_C * sizeof(double), hipMemcpyDeviceToHost));
    std::cout << " done\n";

    // ---- C2 = A_scal * B_scal (FP64) --------------------------------------
    std::cout << "[2/6] C2 = A_scal * B_scal  (FP64) ..." << std::flush;
    HIPBLAS_CHECK(hipblasDgemm(hb, HIPBLAS_OP_N, HIPBLAS_OP_N, (int)N, (int)M, (int)K,
                               &alpha, d_As, (int)N, d_Bs, (int)K, &beta, d_C, (int)N));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_C2.data(), d_C, size_C * sizeof(double), hipMemcpyDeviceToHost));
    std::cout << " done\n";

    // Workspace large enough for both backends.
    size_t ws = std::max(gemmul8::workSize<false, gemmul8::Backend::INT8>(N, M, K, moduli),
                         gemmul8::workSize<false, gemmul8::Backend::FP8>(N, M, K, fp8_moduli));
    void* d_work;
    HIP_CHECK(hipMalloc(&d_work, ws));

    // ---- C3 = A_scal * B_scal (INT8 Ozaki-II) -----------------------------
    std::cout << "[3/6] C3 = A_scal * B_scal  (INT8 Ozaki-II, " << moduli
              << " moduli) ..." << std::flush;
    gemmul8::gemm<double, gemmul8::Backend::INT8>(
        hb, HIPBLAS_OP_N, HIPBLAS_OP_N, (int)N, (int)M, (int)K,
        &alpha, d_As, (int)N, d_Bs, (int)K, &beta, d_C, (int)N,
        moduli, FASTMODE, d_work);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_C3.data(), d_C, size_C * sizeof(double), hipMemcpyDeviceToHost));
    std::cout << " done\n";

    // ---- C4 = A * B (INT8 Ozaki-II, unscaled) -----------------------------
    std::cout << "[4/6] C4 = A * B            (INT8 Ozaki-II, " << moduli
              << " moduli) ..." << std::flush;
    gemmul8::gemm<double, gemmul8::Backend::INT8>(
        hb, HIPBLAS_OP_N, HIPBLAS_OP_N, (int)N, (int)M, (int)K,
        &alpha, d_A, (int)N, d_B, (int)K, &beta, d_C, (int)N,
        moduli, FASTMODE, d_work);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_C4.data(), d_C, size_C * sizeof(double), hipMemcpyDeviceToHost));
    std::cout << " done\n";

    // ---- C5 = A_scal * B_scal (FP8 Ozaki-II) ------------------------------
    std::cout << "[5/6] C5 = A_scal * B_scal  (FP8  Ozaki-II, " << fp8_moduli
              << " moduli) ..." << std::flush;
    gemmul8::gemmLt<double, gemmul8::Backend::FP8>(
        hblt, HIPBLAS_OP_N, HIPBLAS_OP_N, (int)N, (int)M, (int)K,
        &alpha, d_As, (int)N, d_Bs, (int)K, &beta, d_C, (int)N,
        fp8_moduli, FASTMODE, d_work);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_C5.data(), d_C, size_C * sizeof(double), hipMemcpyDeviceToHost));
    std::cout << " done\n";

    // ---- C6 = A * B (FP8 Ozaki-II, unscaled) ------------------------------
    std::cout << "[6/6] C6 = A * B            (FP8  Ozaki-II, " << fp8_moduli
              << " moduli) ..." << std::flush;
    gemmul8::gemmLt<double, gemmul8::Backend::FP8>(
        hblt, HIPBLAS_OP_N, HIPBLAS_OP_N, (int)N, (int)M, (int)K,
        &alpha, d_A, (int)N, d_B, (int)K, &beta, d_C, (int)N,
        fp8_moduli, FASTMODE, d_work);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_C6.data(), d_C, size_C * sizeof(double), hipMemcpyDeviceToHost));
    std::cout << " done\n\n";

    // ---- 6-way Frobenius comparison ---------------------------------------
    std::cout << "------------------------------------------------------------\n";
    std::cout << "6-way Frobenius comparison (reference C1 = A*B, FP64)\n";
    std::cout << "  C1 = A*B (FP64)            C2 = A_scal*B_scal (FP64)\n";
    std::cout << "  C3 = A_scal*B_scal (INT8)  C4 = A*B (INT8)\n";
    std::cout << "  C5 = A_scal*B_scal (FP8)   C6 = A*B (FP8)\n";
    std::cout << "------------------------------------------------------------\n";
    std::printf("  ||C1||_F = %16.16e\n", frob(h_C1));
    std::printf("  ||C2||_F = %16.16e\n", frob(h_C2));
    std::printf("  ||C3||_F = %16.16e\n", frob(h_C3));
    std::printf("  ||C4||_F = %16.16e\n", frob(h_C4));
    std::printf("  ||C5||_F = %16.16e\n", frob(h_C5));
    std::printf("  ||C6||_F = %16.16e\n\n", frob(h_C6));
    std::printf("  ||C1 - C2||_F / ||C1||_F  (FP64 scaled   vs FP64 ref) = %16.16e\n",
                rel_frob(h_C1, h_C2));
    std::printf("  ||C1 - C3||_F / ||C1||_F  (INT8 scaled   vs FP64 ref) = %16.16e\n",
                rel_frob(h_C1, h_C3));
    std::printf("  ||C1 - C4||_F / ||C1||_F  (INT8 unscaled vs FP64 ref) = %16.16e\n",
                rel_frob(h_C1, h_C4));
    std::printf("  ||C1 - C5||_F / ||C1||_F  (FP8  scaled   vs FP64 ref) = %16.16e\n",
                rel_frob(h_C1, h_C5));
    std::printf("  ||C1 - C6||_F / ||C1||_F  (FP8  unscaled vs FP64 ref) = %16.16e\n",
                rel_frob(h_C1, h_C6));
    std::printf("  ||C3 - C4||_F / ||C4||_F  (INT8 scaled   vs INT8 unscaled) = %16.16e\n",
                rel_frob(h_C4, h_C3));
    std::printf("  ||C5 - C6||_F / ||C6||_F  (FP8  scaled   vs FP8  unscaled) = %16.16e\n",
                rel_frob(h_C6, h_C5));
    std::cout << "------------------------------------------------------------\n";

    HIP_CHECK(hipFree(d_work));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_As));
    HIP_CHECK(hipFree(d_Bs));
    HIP_CHECK(hipFree(d_C));
    hipblasLtDestroy(hblt);
    HIPBLAS_CHECK(hipblasDestroy(hb));
    return 0;
}
