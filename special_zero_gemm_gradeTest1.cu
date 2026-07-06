/**
 * special_zero_gemm_gradeTest1
 * ----------------------------
 * Self-contained experiment requested for rocEMU:
 *
 *  (1) Generate two DIFFERENT random square matrices A, B (default 1024x1024)
 *      with i.i.d. normal entries N(mean=0, std=1).
 *
 *  (2) Plant 5 "special" (row, column) pairs whose dot product is EXACTLY zero:
 *        - pick 5 random rows of A; in each, zero 50% of the entries (randomly
 *          chosen positions).
 *        - pick 5 random columns of B (one paired to each special row of A) and
 *          zero the COMPLEMENTARY positions, i.e. B[k, j] = 0 wherever A[i, k] != 0.
 *      Then for every k either A[i,k]==0 or B[k,j]==0, so
 *        C[i, j] = sum_k A[i,k] * B[k,j] = 0  exactly.
 *
 *  (3) Compute C = A * B with
 *        (a) native FP64 GEMM        (hipblasDgemm)
 *        (b) Ozaki-II GEMM via GEMMul8, INT8 backend, 16 moduli (gemmul8::gemm)
 *      and report:
 *        - the relative Frobenius residual
 *              (||C_fp64||_F - ||C_emu||_F) / ||C_fp64||_F           (as requested)
 *          and, for reference, the standard
 *              ||C_fp64 - C_emu||_F / ||C_fp64||_F
 *        - the 5 planted-zero locations: C_fp64[i,j] and C_emu[i,j], which
 *          should both be ~0.
 *
 * Build (or `make special_zero_gemm_gradeTest1`):
 *   hipcc -O3 -std=c++17 special_zero_gemm_gradeTest1.cu -o special_zero_gemm_gradeTest1 \
 *       -I/opt/rocm/include -I$(GEMMUL8_PATH)/include \
 *       -L/opt/rocm/lib -L$(GEMMUL8_PATH)/lib -Wl,-rpath,$(GEMMUL8_PATH)/lib \
 *       -lgemmul8 -lhipblas -lamdhip64
 *
 * Usage:
 *   ./special_zero_gemm_gradeTest1 [N] [moduli] [seed]
 *   defaults: N=1024, moduli=16, seed=12345
 */

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include "gemmul8.hpp"

// GEMMul8 internal profiling flag (see other benchmarks in this repo).
namespace oz2 { bool g_profiling_enabled = false; }

#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <string>

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

// Ozaki scaling path: accurate (false) for best accuracy.
static const bool FASTMODE = false;

// Column-major indexing helpers (BLAS convention, leading dim = number of rows).
static inline size_t idxA(size_t i, size_t k, size_t N) { return k * N + i; } // A is N x K
static inline size_t idxB(size_t k, size_t j, size_t K) { return j * K + k; } // B is K x M
static inline size_t idxC(size_t i, size_t j, size_t N) { return j * N + i; } // C is N x M

int main(int argc, char** argv) {
    // ---- Parameters --------------------------------------------------------
    size_t N = 1024;          // square: A is NxN, B is NxN, C is NxN (K = M = N)
    int    moduli = 16;       // INT8 Ozaki-II moduli
    unsigned long seed = 12345UL;
    const int NUM_SPECIAL = 5;

    if (argc > 1) N = std::stoull(argv[1]);
    if (argc > 2) moduli = std::stoi(argv[2]);
    if (argc > 3) seed = std::stoul(argv[3]);

    const size_t K = N, M = N;
    const size_t size_A = N * K, size_B = K * M, size_C = N * M;

    std::cout << "============================================================\n";
    std::cout << " special_zero_gemm : FP64 vs INT8 Ozaki-II (GEMMul8)\n";
    std::cout << "============================================================\n";
    std::cout << "  N = K = M      : " << N << "\n";
    std::cout << "  INT8 moduli    : " << moduli << "\n";
    std::cout << "  RNG seed       : " << seed << "\n";
    std::cout << "  special pairs  : " << NUM_SPECIAL << " (dot product = 0 by construction)\n\n";

    // ---- (1) Generate A and B (different) with N(0,1) entries --------------
    std::vector<double> h_A(size_A), h_B(size_B);

    std::mt19937_64 gen_A(seed);
    std::mt19937_64 gen_B(seed + 0x9E3779B97F4A7C15ULL); // different stream => A != B
    std::normal_distribution<double> nd(0.0, 1.0);

    for (size_t t = 0; t < size_A; ++t) h_A[t] = nd(gen_A);
    for (size_t t = 0; t < size_B; ++t) h_B[t] = nd(gen_B);

    // ---- (2) Plant 5 special rows/columns with complementary zeros --------
    std::mt19937_64 gen_sel(seed + 777);

    // Pick NUM_SPECIAL distinct rows of A and distinct columns of B.
    std::vector<size_t> all_rows(N), all_cols(M);
    std::iota(all_rows.begin(), all_rows.end(), 0);
    std::iota(all_cols.begin(), all_cols.end(), 0);
    std::shuffle(all_rows.begin(), all_rows.end(), gen_sel);
    std::shuffle(all_cols.begin(), all_cols.end(), gen_sel);

    std::vector<size_t> special_rows(all_rows.begin(), all_rows.begin() + NUM_SPECIAL);
    std::vector<size_t> special_cols(all_cols.begin(), all_cols.begin() + NUM_SPECIAL);

    const size_t num_zero = K / 2; // 50% of the K entries along the row

    std::cout << "Planted special (row of A, col of B) pairs:\n";
    for (int s = 0; s < NUM_SPECIAL; ++s) {
        size_t i = special_rows[s];
        size_t j = special_cols[s];

        // Random 50% positions in [0,K) to zero within row i of A.
        std::vector<size_t> pos(K);
        std::iota(pos.begin(), pos.end(), 0);
        std::shuffle(pos.begin(), pos.end(), gen_sel);

        // First `num_zero` positions -> zero in A's row i.
        // Complementary positions (A nonzero) -> zero in B's column j.
        std::vector<char> zeroInA(K, 0);
        for (size_t z = 0; z < num_zero; ++z) zeroInA[pos[z]] = 1;

        size_t a_zeros = 0, b_zeros = 0;
        for (size_t k = 0; k < K; ++k) {
            if (zeroInA[k]) {
                h_A[idxA(i, k, N)] = 0.0;      // A row zeroed here
                ++a_zeros;
            } else {
                h_B[idxB(k, j, K)] = 0.0;      // B col zeroed on the complement
                ++b_zeros;
            }
        }
        std::cout << "  pair " << s << ": row A[" << i << ",:] (" << a_zeros
                  << " zeros)  x  col B[:," << j << "] (" << b_zeros
                  << " zeros)\n";
    }
    std::cout << "\n";

    // ---- Device buffers ----------------------------------------------------
    double *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, size_A * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_B, size_B * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_C, size_C * sizeof(double)));
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_A * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B * sizeof(double), hipMemcpyHostToDevice));

    hipDeviceProp_t dprop;
    HIP_CHECK(hipGetDeviceProperties(&dprop, 0));
    std::cout << "Device: " << dprop.name << "\n\n";

    hipblasHandle_t hb;
    HIPBLAS_CHECK(hipblasCreate(&hb));

    const double alpha = 1.0, beta = 0.0;

    // ---- (3a) Native FP64 GEMM --------------------------------------------
    std::cout << "[1/2] Native FP64 GEMM (hipblasDgemm) ..." << std::flush;
    HIPBLAS_CHECK(hipblasDgemm(hb, HIPBLAS_OP_N, HIPBLAS_OP_N,
                               (int)N, (int)M, (int)K,
                               &alpha, d_A, (int)N, d_B, (int)K,
                               &beta, d_C, (int)N));
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<double> h_C_fp64(size_C);
    HIP_CHECK(hipMemcpy(h_C_fp64.data(), d_C, size_C * sizeof(double), hipMemcpyDeviceToHost));
    std::cout << " done\n";

    // ---- (3b) INT8 Ozaki-II GEMM (GEMMul8), `moduli` moduli ---------------
    std::cout << "[2/2] INT8 Ozaki-II GEMM (GEMMul8, " << moduli
              << " moduli) ..." << std::flush;
    size_t ws = gemmul8::workSize<false, gemmul8::Backend::INT8>(N, M, K, moduli);
    void* d_work;
    HIP_CHECK(hipMalloc(&d_work, ws));
    gemmul8::gemm<double, gemmul8::Backend::INT8>(
        hb, HIPBLAS_OP_N, HIPBLAS_OP_N, (int)N, (int)M, (int)K,
        &alpha, d_A, (int)N, d_B, (int)K, &beta, d_C, (int)N,
        moduli, FASTMODE, d_work);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<double> h_C_emu(size_C);
    HIP_CHECK(hipMemcpy(h_C_emu.data(), d_C, size_C * sizeof(double), hipMemcpyDeviceToHost));
    std::cout << " done\n\n";

    // ---- Residuals ---------------------------------------------------------
    double norm_fp64 = 0.0, norm_emu = 0.0, norm_diff = 0.0;
    for (size_t t = 0; t < size_C; ++t) {
        double a = h_C_fp64[t], b = h_C_emu[t], d = a - b;
        norm_fp64 += a * a;
        norm_emu  += b * b;
        norm_diff += d * d;
    }
    norm_fp64 = std::sqrt(norm_fp64);
    norm_emu  = std::sqrt(norm_emu);
    norm_diff = std::sqrt(norm_diff);

    std::cout << "------------------------------------------------------------\n";
    std::cout << "Frobenius-norm residuals\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "  ||C_fp64||_F                                = " << norm_fp64 << "\n";
    std::cout << "  ||C_emu ||_F                                = " << norm_emu  << "\n";
    std::cout << "  ||C_fp64 - C_emu||_F / ||C_fp64||_F         = "
              << norm_diff / norm_fp64 << "   (elementwise, for reference)\n\n";

    // ---- Planted-zero locations -------------------------------------------
    std::cout << "------------------------------------------------------------\n";
    std::cout << "Planted-zero entries of C (should be ~0 for both methods)\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "   #   (i, j)            C_fp64            C_emu(INT8)\n";
    for (int s = 0; s < NUM_SPECIAL; ++s) {
        size_t i = special_rows[s], j = special_cols[s];
        size_t p = idxC(i, j, N);
        std::cout << "  " << s << "   (" << std::setw(5) << i << "," << std::setw(5) << j
                  << ")   " << std::setw(15) << h_C_fp64[p]
                  << "   " << std::setw(15) << h_C_emu[p] << "\n";
    }
    std::cout << "------------------------------------------------------------\n";

    HIP_CHECK(hipFree(d_work));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    HIPBLAS_CHECK(hipblasDestroy(hb));
    return 0;
}
