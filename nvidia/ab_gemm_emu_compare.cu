/**
 * A/B GEMM Emulation Comparison — CUDA port of ../ab_gemm_emu_compare.cu
 * -----------------------------------------------------------------------
 * Reads matrix pairs A and B from a folder (condition-number sweep produced by
 * ab_matrix_generator) and computes C = A * B with five methods:
 *
 *   1. Native FP64           (cublasDgemm)
 *   2. INT8 Ozaki-II, 12 moduli   (gemmul8::gemm,   Backend::INT8)
 *   3. INT8 Ozaki-II, 16 moduli   (gemmul8::gemm,   Backend::INT8)
 *   4. FP8  Ozaki-II, 10 moduli   (gemmul8::gemmLt, Backend::FP8)
 *   5. FP8  Ozaki-II, 12 moduli   (gemmul8::gemmLt, Backend::FP8)
 *
 * For each matrix pair it:
 *   - saves all 5 result matrices to <output_folder>/C_<method>_cond1eX.txt
 *   - records the Frobenius-norm difference ||C_native - C_method||_F
 *
 * A CSV summary is written with 6 columns:
 *   cond_number, native_diff, int8_ozaki12_diff, int8_ozaki16_diff,
 *   fp8_ozaki10_diff, fp8_ozaki12_diff
 * (native_diff is 0 by construction.)
 *
 * Build (from this directory):
 *   make ab_gemm_emu_compare
 *
 * Usage:
 *   ./ab_gemm_emu_compare <matrix_folder> [output_folder] [--no-save-matrices]
 */

#include "cuda_helpers.cuh"

#include <cublasLt.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#include "gemmul8.hpp"

#define CUBLASLT_CHECK(call)                                                     \
    do {                                                                         \
        cublasStatus_t status = (call);                                          \
        if (status != CUBLAS_STATUS_SUCCESS) {                                     \
            std::cerr << "cuBLASLt error at " << __FILE__ << ":" << __LINE__     \
                      << " status=" << status << std::endl;                      \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

static const bool FASTMODE = false;

static const int INT8_MODULI_A = 12;
static const int INT8_MODULI_B = 16;
static const int FP8_MODULI_A  = 10;
static const int FP8_MODULI_B  = 12;

bool read_matrix_from_file(const std::string& filename,
                           std::vector<double>& data,
                           size_t& rows, size_t& cols) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    std::vector<std::vector<double>> temp_data;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            try {
                row.push_back(std::stod(value));
            } catch (const std::exception&) {
                std::cerr << "Error parsing value: " << value << std::endl;
                return false;
            }
        }
        if (!row.empty()) temp_data.push_back(row);
    }

    if (temp_data.empty()) {
        std::cerr << "Error: Empty matrix in file " << filename << std::endl;
        return false;
    }

    rows = temp_data.size();
    cols = temp_data[0].size();

    data.resize(rows * cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            data[j * rows + i] = temp_data[i][j];
        }
    }
    return true;
}

void write_matrix_to_file(const double* C, size_t rows, size_t cols,
                          const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        return;
    }
    // Full double precision: scientific notation with 17 significant digits
    // (max needed to round-trip an IEEE-754 double, > 16 digits).
    file << std::scientific << std::setprecision(17);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            if (j > 0) file << ",";
            file << C[j * rows + i];
        }
        file << "\n";
    }
}

bool create_directory(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode);
    }
    return mkdir(path.c_str(), 0755) == 0;
}

std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");
    return ss.str();
}

struct MatrixPair {
    std::string file_A;
    std::string file_B;
    int log10_cond;
    size_t N, K, M;
};

std::vector<MatrixPair> find_matrix_pairs(const std::string& folder) {
    std::vector<MatrixPair> pairs;
    std::map<int, MatrixPair> pair_map;

    DIR* dir = opendir(folder.c_str());
    if (!dir) {
        std::cerr << "Error: Cannot open directory " << folder << std::endl;
        return pairs;
    }

    std::regex pattern_A("A_(\\d+)x(\\d+)_cond1e(\\d+)\\.txt");
    std::regex pattern_B("B_(\\d+)x(\\d+)_cond1e(\\d+)\\.txt");
    std::smatch match;

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        if (std::regex_match(filename, match, pattern_A)) {
            int cond = std::stoi(match[3].str());
            pair_map[cond].file_A = folder + "/" + filename;
            pair_map[cond].N = std::stoull(match[1].str());
            pair_map[cond].K = std::stoull(match[2].str());
            pair_map[cond].log10_cond = cond;
        } else if (std::regex_match(filename, match, pattern_B)) {
            int cond = std::stoi(match[3].str());
            pair_map[cond].file_B = folder + "/" + filename;
            pair_map[cond].M = std::stoull(match[2].str());
            pair_map[cond].log10_cond = cond;
        }
    }
    closedir(dir);

    for (auto& kv : pair_map) {
        if (!kv.second.file_A.empty() && !kv.second.file_B.empty()) {
            pairs.push_back(kv.second);
        }
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const MatrixPair& a, const MatrixPair& b) { return a.log10_cond < b.log10_cond; });
    return pairs;
}

double compute_frobenius_diff(const double* A, const double* B, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = A[i] - B[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

struct PairResult {
    int log10_cond;
    double diff_int8_12;
    double diff_int8_16;
    double diff_fp8_10;
    double diff_fp8_12;
};

bool process_pair(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle,
                  const MatrixPair& pair, const std::string& out_folder,
                  bool save_matrices, PairResult& result) {

    std::cout << "\n========================================" << std::endl;
    std::cout << "Condition number: 10^" << pair.log10_cond << std::endl;

    std::vector<double> h_A, h_B;
    size_t rows_A, cols_A, rows_B, cols_B;
    if (!read_matrix_from_file(pair.file_A, h_A, rows_A, cols_A)) return false;
    if (!read_matrix_from_file(pair.file_B, h_B, rows_B, cols_B)) return false;

    if (cols_A != rows_B) {
        std::cerr << "Error: dimension mismatch A(" << rows_A << "x" << cols_A
                  << ") B(" << rows_B << "x" << cols_B << ")" << std::endl;
        return false;
    }

    size_t N = rows_A, K = cols_A, M = cols_B;
    size_t size_A = N * K, size_B = K * M, size_C = N * M;
    std::cout << "  A: " << N << "x" << K << ", B: " << K << "x" << M
              << ", C: " << N << "x" << M << std::endl;

    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B * sizeof(double), cudaMemcpyHostToDevice));

    size_t ws_int8_12 = gemmul8::workSize<false, gemmul8::Backend::INT8>(N, M, K, INT8_MODULI_A);
    size_t ws_int8_16 = gemmul8::workSize<false, gemmul8::Backend::INT8>(N, M, K, INT8_MODULI_B);
    size_t ws_fp8_10  = gemmul8::workSize<false, gemmul8::Backend::FP8>(N, M, K, FP8_MODULI_A);
    size_t ws_fp8_12  = gemmul8::workSize<false, gemmul8::Backend::FP8>(N, M, K, FP8_MODULI_B);
    size_t ws_max = std::max({ws_int8_12, ws_int8_16, ws_fp8_10, ws_fp8_12});

    void* d_work;
    CUDA_CHECK(cudaMalloc(&d_work, ws_max));

    double alpha = 1.0, beta = 0.0;

    std::vector<double> h_C_native(size_C), h_C(size_C);

    std::cout << "  [1/5] Native FP64 (cublasDgemm)..." << std::flush;
    CUBLAS_CHECK(cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             (int)N, (int)M, (int)K,
                             &alpha, d_A, (int)N, d_B, (int)K, &beta, d_C, (int)N));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C_native.data(), d_C, size_C * sizeof(double), cudaMemcpyDeviceToHost));
    std::cout << " done" << std::endl;

    std::string cond_tag = "cond1e" + std::to_string(pair.log10_cond);
    if (save_matrices) {
        write_matrix_to_file(h_C_native.data(), N, M,
                             out_folder + "/C_native_fp64_" + cond_tag + ".txt");
    }

    auto run_int8 = [&](int moduli, const char* tag) -> double {
        std::cout << "  INT8 Ozaki-II (" << moduli << " moduli)..." << std::flush;
        gemmul8::gemm<double, gemmul8::Backend::INT8>(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
            &alpha, d_A, N, d_B, K, &beta, d_C, N,
            moduli, FASTMODE, d_work);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C * sizeof(double), cudaMemcpyDeviceToHost));
        double diff = compute_frobenius_diff(h_C_native.data(), h_C.data(), size_C);
        if (save_matrices) {
            write_matrix_to_file(h_C.data(), N, M,
                                 out_folder + "/C_" + tag + "_" + cond_tag + ".txt");
        }
        std::cout << " frob_diff=" << std::scientific << std::setprecision(6) << diff
                  << std::fixed << std::endl;
        return diff;
    };

    auto run_fp8 = [&](int moduli, const char* tag) -> double {
        std::cout << "  FP8  Ozaki-II (" << moduli << " moduli)..." << std::flush;
        gemmul8::gemmLt<double, gemmul8::Backend::FP8>(
            cublaslt_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
            &alpha, d_A, N, d_B, K, &beta, d_C, N,
            moduli, FASTMODE, d_work);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C * sizeof(double), cudaMemcpyDeviceToHost));
        double diff = compute_frobenius_diff(h_C_native.data(), h_C.data(), size_C);
        if (save_matrices) {
            write_matrix_to_file(h_C.data(), N, M,
                                 out_folder + "/C_" + tag + "_" + cond_tag + ".txt");
        }
        std::cout << " frob_diff=" << std::scientific << std::setprecision(6) << diff
                  << std::fixed << std::endl;
        return diff;
    };

    result.diff_int8_12 = run_int8(INT8_MODULI_A, "int8_ozaki12");
    result.diff_int8_16 = run_int8(INT8_MODULI_B, "int8_ozaki16");
    result.diff_fp8_10 = run_fp8(FP8_MODULI_A, "fp8_ozaki10");
    result.diff_fp8_12 = run_fp8(FP8_MODULI_B, "fp8_ozaki12");

    result.log10_cond = pair.log10_cond;

    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return true;
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " <matrix_folder> [output_folder] [--no-save-matrices]\n\n";
    std::cout << "Reads A_NxK_cond1eX.txt / B_KxM_cond1eX.txt pairs and computes C = A*B with:\n";
    std::cout << "  native FP64, INT8 Ozaki-II (12 & 16 moduli), FP8 Ozaki-II (10 & 12 moduli).\n\n";
    std::cout << "Outputs (in output_folder, default <matrix_folder>/emu_compare_<timestamp>):\n";
    std::cout << "  C_native_fp64_cond1eX.txt, C_int8_ozaki12_cond1eX.txt, ...\n";
    std::cout << "  frobenius_diff_summary.csv (6 columns)\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
        print_usage(argv[0]);
        return (argc < 2) ? 1 : 0;
    }

    std::string matrix_folder = argv[1];
    if (!matrix_folder.empty() && matrix_folder.back() == '/') matrix_folder.pop_back();

    std::string output_folder;
    bool save_matrices = true;
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--no-save-matrices") {
            save_matrices = false;
        } else if (output_folder.empty()) {
            output_folder = arg;
        }
    }
    if (output_folder.empty()) {
        output_folder = matrix_folder + "/emu_compare_" + get_timestamp();
    }

    std::cout << "============================================" << std::endl;
    std::cout << "A/B GEMM Emulation Comparison (FP64 vs Ozaki-II, CUDA)" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Input folder:  " << matrix_folder << std::endl;
    std::cout << "Output folder: " << output_folder << std::endl;
    std::cout << "Save result matrices: " << (save_matrices ? "yes" : "no") << std::endl;

    std::vector<MatrixPair> pairs = find_matrix_pairs(matrix_folder);
    if (pairs.empty()) {
        std::cerr << "No A_*/B_* cond matrix pairs found in " << matrix_folder << std::endl;
        return 1;
    }
    std::cout << "Found " << pairs.size() << " matrix pair(s)" << std::endl;

    if (!create_directory(output_folder)) {
        std::cerr << "Failed to create output folder " << output_folder << std::endl;
        return 1;
    }

    cudaDeviceProp dprop;
    CUDA_CHECK(cudaGetDeviceProperties(&dprop, 0));
    std::cout << "Device: " << dprop.name << std::endl;

    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUBLASLT_CHECK(cublasLtCreate(&cublaslt_handle));

    std::string csv_path = output_folder + "/frobenius_diff_summary.csv";
    std::ofstream csv(csv_path);
    csv << "cond_number,native_diff,int8_ozaki12_diff,int8_ozaki16_diff,"
        << "fp8_ozaki10_diff,fp8_ozaki12_diff\n";
    csv << std::scientific << std::setprecision(17);

    for (const auto& pair : pairs) {
        PairResult r{};
        if (!process_pair(cublas_handle, cublaslt_handle, pair, output_folder, save_matrices, r)) {
            std::cerr << "Skipping condition 10^" << pair.log10_cond << std::endl;
            continue;
        }
        csv << "1e" << r.log10_cond << ","
            << 0.0 << ","
            << r.diff_int8_12 << ","
            << r.diff_int8_16 << ","
            << r.diff_fp8_10 << ","
            << r.diff_fp8_12 << "\n";
        csv.flush();
    }

    csv.close();
    CUBLASLT_CHECK(cublasLtDestroy(cublaslt_handle));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));

    std::cout << "\n============================================" << std::endl;
    std::cout << "Done. CSV summary: " << csv_path << std::endl;
    if (save_matrices) {
        std::cout << "Result matrices written to: " << output_folder << std::endl;
    }
    std::cout << "============================================" << std::endl;
    return 0;
}
