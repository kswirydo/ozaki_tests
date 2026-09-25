/**
 * aspect_ratio_test_alt
 * ---------------------
 * Dynamic-range (aspect ratio) accuracy test for INT8 Ozaki-II emulation.
 *
 * Unlike aspect_ratio_generator.cu, which writes matrix files and forces both row
 * and column extremes into the same matrix, this test is self-contained and puts
 * the dynamic range where it matters for C = A * B:
 *
 *   - every ROW    of A (M x K) spans max|a| / min|a| = 10^r
 *   - every COLUMN of B (K x N) spans max|b| / min|b| = 10^r
 *
 * so every dot product C[i,j] = sum_k A[i,k] * B[k,j] mixes terms whose magnitudes
 * differ by up to 10^r. A and B are drawn from independent random streams.
 *
 * Element magnitudes are 10^(r*(u - 1/2)) with u ~ U[0,1] and a random sign; two
 * positions per row of A (and per column of B) are pinned to exactly 10^(+/-r/2)
 * so the requested ratio is attained, not just approached.
 *
 * For each aspect ratio the test reports the relative Frobenius error
 *   ||C_ozaki - C_fp64||_F / ||C_fp64||_F
 * for INT8 Ozaki-II (GEMMul8) in both accurate and fast mode, plus the measured
 * worst-case row/column ratio as a sanity check.
 *
 * Build (or `make aspect_ratio_test_alt`):
 *   hipcc -O3 -std=c++17 aspect_ratio_test_alt.cu -o aspect_ratio_test_alt \
 *       -I/opt/rocm/include -I$(GEMMUL8_PATH)/include \
 *       -L/opt/rocm/lib -L$(GEMMUL8_PATH)/lib -Wl,-rpath,$(GEMMUL8_PATH)/lib \
 *       -lgemmul8 -lhipblas -lamdhip64
 *
 * Usage:
 *   ./aspect_ratio_test_alt [N] [ratio_max] [seed] [output.csv]
 *   defaults: N=4096, ratio_max=31, seed=12345
 */

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hiprand/hiprand.h>
#include "gemmul8.hpp"

// GEMMul8 internal profiling flag (see other benchmarks in this repo).
namespace oz2 { bool g_profiling_enabled = false; }

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
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

#define HIPRAND_CHECK(call)                                                     \
    do {                                                                        \
        hiprandStatus_t status = call;                                          \
        if (status != HIPRAND_STATUS_SUCCESS) {                                 \
            std::cerr << "hipRAND error at " << __FILE__ << ":" << __LINE__     \
                      << " status=" << status << std::endl;                     \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

static const int NUM_WARMUP = 2;
static const int NUM_ITERATIONS = 5;
static const std::vector<unsigned> MODULI_LIST = {12, 16};

/** Magnitude 10^(log10_ratio * (u - 1/2)) with a random sign. */
__global__ void apply_aspect_ratio_kernel(double* data, const double* uniform, const double* signs,
                                          size_t size, double log10_ratio) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double magnitude = pow(10.0, log10_ratio * (uniform[idx] - 0.5));
        double sign = (signs[idx] > 0.5) ? 1.0 : -1.0;
        data[idx] = sign * magnitude;
    }
}

/** Column-major A (rows x cols): pin the extremes inside each ROW. */
__global__ void pin_row_extremes_kernel(double* A, size_t rows, size_t cols, double log10_ratio) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && cols >= 2) {
        double max_val = pow(10.0, log10_ratio / 2.0);
        double min_val = pow(10.0, -log10_ratio / 2.0);
        size_t max_idx = i;                 // column 0
        size_t min_idx = i + rows;          // column 1
        A[max_idx] = (A[max_idx] >= 0.0 ? 1.0 : -1.0) * max_val;
        A[min_idx] = (A[min_idx] >= 0.0 ? 1.0 : -1.0) * min_val;
    }
}

/** Column-major B (rows x cols): pin the extremes inside each COLUMN. */
__global__ void pin_col_extremes_kernel(double* B, size_t rows, size_t cols, double log10_ratio) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < cols && rows >= 2) {
        double max_val = pow(10.0, log10_ratio / 2.0);
        double min_val = pow(10.0, -log10_ratio / 2.0);
        size_t max_idx = j * rows;          // row 0
        size_t min_idx = j * rows + 1;      // row 1
        B[max_idx] = (B[max_idx] >= 0.0 ? 1.0 : -1.0) * max_val;
        B[min_idx] = (B[min_idx] >= 0.0 ? 1.0 : -1.0) * min_val;
    }
}

enum class Extremes { Rows, Cols };

static void generate_aspect_ratio_matrix(hiprandGenerator_t gen, size_t rows, size_t cols,
                                         int log10_ratio, Extremes where, double* d_M) {
    size_t size = rows * cols;

    double *d_uniform, *d_signs;
    HIP_CHECK(hipMalloc(&d_uniform, size * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_signs, size * sizeof(double)));

    HIPRAND_CHECK(hiprandGenerateUniformDouble(gen, d_uniform, size));
    HIPRAND_CHECK(hiprandGenerateUniformDouble(gen, d_signs, size));

    int block_size = 256;
    size_t num_blocks = (size + block_size - 1) / block_size;
    hipLaunchKernelGGL(apply_aspect_ratio_kernel, dim3(num_blocks), dim3(block_size), 0, 0, d_M,
                       d_uniform, d_signs, size, (double)log10_ratio);

    if (where == Extremes::Rows) {
        num_blocks = (rows + block_size - 1) / block_size;
        hipLaunchKernelGGL(pin_row_extremes_kernel, dim3(num_blocks), dim3(block_size), 0, 0, d_M, rows,
                           cols, (double)log10_ratio);
    } else {
        num_blocks = (cols + block_size - 1) / block_size;
        hipLaunchKernelGGL(pin_col_extremes_kernel, dim3(num_blocks), dim3(block_size), 0, 0, d_M, rows,
                           cols, (double)log10_ratio);
    }
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_uniform));
    HIP_CHECK(hipFree(d_signs));
}

static double frobenius_rel_error(const std::vector<double>& ref, const std::vector<double>& test) {
    double diff_sum = 0.0, norm_sum = 0.0;
    for (size_t i = 0; i < ref.size(); i++) {
        double d = ref[i] - test[i];
        diff_sum += d * d;
        norm_sum += ref[i] * ref[i];
    }
    if (norm_sum <= 0.0) return 0.0;
    return std::sqrt(diff_sum) / std::sqrt(norm_sum);
}

/** Smallest observed max|.|/min|.| over all rows (column-major). */
static double min_row_ratio(const std::vector<double>& A, size_t rows, size_t cols) {
    double worst = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < rows; i++) {
        double lo = std::numeric_limits<double>::infinity(), hi = 0.0;
        for (size_t j = 0; j < cols; j++) {
            double v = std::abs(A[i + j * rows]);
            if (v > 0.0) {
                lo = std::min(lo, v);
                hi = std::max(hi, v);
            }
        }
        if (lo > 0.0 && std::isfinite(lo)) worst = std::min(worst, hi / lo);
    }
    return worst;
}

/** Smallest observed max|.|/min|.| over all columns (column-major). */
static double min_col_ratio(const std::vector<double>& B, size_t rows, size_t cols) {
    double worst = std::numeric_limits<double>::infinity();
    for (size_t j = 0; j < cols; j++) {
        double lo = std::numeric_limits<double>::infinity(), hi = 0.0;
        for (size_t i = 0; i < rows; i++) {
            double v = std::abs(B[i + j * rows]);
            if (v > 0.0) {
                lo = std::min(lo, v);
                hi = std::max(hi, v);
            }
        }
        if (lo > 0.0 && std::isfinite(lo)) worst = std::min(worst, hi / lo);
    }
    return worst;
}

static std::string get_device_name() {
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::string name = prop.name;
    for (char& c : name) {
        if (c == ' ' || c == '/' || c == '\\') c = '_';
    }
    return name;
}

static std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&t), "%Y-%m-%d_%H-%M-%S");
    return oss.str();
}

int main(int argc, char* argv[]) {
    if (argc >= 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [N] [ratio_max] [seed] [output.csv]\n"
                  << "  N          square matrix dimension (default: 4096)\n"
                  << "  ratio_max  sweep aspect ratio 10^0 .. 10^ratio_max (default: 31)\n"
                  << "  seed       RNG seed (default: 12345)\n"
                  << "  Large dynamic range is placed in every ROW of A and every COLUMN of B.\n";
        return 0;
    }

    size_t n = (argc >= 2) ? std::stoull(argv[1]) : 4096;
    int ratio_max = (argc >= 3) ? std::stoi(argv[2]) : 31;
    unsigned long long seed = (argc >= 4) ? std::stoull(argv[3]) : 12345ULL;
    std::string csv_path = (argc >= 5) ? argv[4]
                                       : ("aspect_ratio_test_alt_" + get_device_name() + "_" +
                                          get_timestamp() + ".csv");

    const size_t elems = n * n;
    const size_t bytes = elems * sizeof(double);
    const double flops = 2.0 * (double)n * (double)n * (double)n;

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cerr << "aspect_ratio_test_alt\n"
              << "  Device: " << prop.name << "\n"
              << "  Matrix: " << n << " x " << n << "  C = A * B\n"
              << "  Aspect ratio: 10^0 .. 10^" << ratio_max
              << "  (rows of A, columns of B)\n"
              << "  CSV: " << csv_path << std::endl;

    hipblasHandle_t handle;
    HIPBLAS_CHECK(hipblasCreate(&handle));

    hiprandGenerator_t gen;
    HIPRAND_CHECK(hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT));

    double *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, bytes));
    HIP_CHECK(hipMalloc(&d_B, bytes));
    HIP_CHECK(hipMalloc(&d_C, bytes));

    unsigned max_moduli = *std::max_element(MODULI_LIST.begin(), MODULI_LIST.end());
    size_t worksize = gemmul8::workSize<false, gemmul8::Backend::INT8>(n, n, n, max_moduli);
    void* d_work;
    HIP_CHECK(hipMalloc(&d_work, worksize));

    std::vector<double> h_A(elems), h_B(elems), h_ref(elems), h_test(elems);
    const double alpha = 1.0, beta = 0.0;

    std::ofstream csv(csv_path);
    if (!csv.is_open()) {
        std::cerr << "Failed to open " << csv_path << std::endl;
        return 1;
    }
    csv << "log10_aspect_ratio,matrix_n,measured_min_row_ratio_A,measured_min_col_ratio_B,"
        << "fp64_time_ms,fp64_tflops";
    for (unsigned m : MODULI_LIST) {
        csv << ",int8_ozaki" << m << "_accurate_frob_rel_err"
            << ",int8_ozaki" << m << "_fast_frob_rel_err";
    }
    csv << "\n";

    for (int r = 0; r <= ratio_max; r++) {
        std::cerr << "aspect ratio 10^" << r << " ..." << std::flush;

        HIPRAND_CHECK(hiprandSetPseudoRandomGeneratorSeed(gen, seed + (unsigned long long)r));
        generate_aspect_ratio_matrix(gen, n, n, r, Extremes::Rows, d_A);
        generate_aspect_ratio_matrix(gen, n, n, r, Extremes::Cols, d_B);

        HIP_CHECK(hipMemcpy(h_A.data(), d_A, bytes, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_B.data(), d_B, bytes, hipMemcpyDeviceToHost));
        double row_ratio_A = min_row_ratio(h_A, n, n);
        double col_ratio_B = min_col_ratio(h_B, n, n);

        for (int i = 0; i < NUM_WARMUP; i++) {
            HIPBLAS_CHECK(hipblasDgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, &alpha, d_A, n, d_B,
                                       n, &beta, d_C, n));
        }
        HIP_CHECK(hipDeviceSynchronize());

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            HIPBLAS_CHECK(hipblasDgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, &alpha, d_A, n, d_B,
                                       n, &beta, d_C, n));
        }
        HIP_CHECK(hipDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();

        double fp64_sec = std::chrono::duration<double>(t1 - t0).count() / NUM_ITERATIONS;
        HIP_CHECK(hipMemcpy(h_ref.data(), d_C, bytes, hipMemcpyDeviceToHost));

        csv << r << "," << n << "," << std::scientific << std::setprecision(6) << row_ratio_A << ","
            << col_ratio_B << "," << std::fixed << std::setprecision(6) << fp64_sec * 1000.0 << ","
            << flops / (fp64_sec * 1e12);

        for (unsigned moduli : MODULI_LIST) {
            for (int fast = 0; fast <= 1; fast++) {
                bool fastmode = (fast == 1);
                gemmul8::gemm<double>(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, &alpha, d_A, n, d_B, n,
                                      &beta, d_C, n, moduli, fastmode, d_work);
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipMemcpy(h_test.data(), d_C, bytes, hipMemcpyDeviceToHost));

                double err = frobenius_rel_error(h_ref, h_test);
                csv << "," << std::scientific << std::setprecision(6) << err;
                std::cerr << "  " << moduli << (fastmode ? "f" : "a") << "=" << std::scientific
                          << std::setprecision(2) << err;
            }
        }
        csv << std::fixed << "\n";
        csv.flush();
        std::cerr << std::endl;
    }

    csv.close();
    HIP_CHECK(hipFree(d_work));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    HIPRAND_CHECK(hiprandDestroyGenerator(gen));
    HIPBLAS_CHECK(hipblasDestroy(handle));

    std::cerr << "Wrote " << csv_path << std::endl;
    return 0;
}
