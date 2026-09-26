/**
 * Simple GEMM Benchmark — CUDA / cuBLAS + GEMMul8 (NVIDIA port of ../simple_gemm_benchmark.cu)
 *
 * Compile:
 *   nvcc -O3 -std=c++17 simple_gemm_benchmark.cu -o simple_gemm_benchmark \
 *        -I${GEMMUL8_PATH}/include -L${GEMMUL8_PATH}/lib -lgemmul8 -lcublas -lcudart
 */

#include "cuda_helpers.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "gemmul8.hpp"

static const int NUM_WARMUP = 2;
static const int NUM_ITERATIONS = 5;
static const std::vector<unsigned> NUM_MODULI_LIST = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

bool read_matrix_from_file(const std::string& filename, std::vector<double>& data, size_t& rows, size_t& cols) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }
    std::vector<std::vector<double>> temp_data;
    std::string line;
    while (std::getline(file,
        line
    )) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss,
            value,
            ','
        )) {
            try {
                row.push_back(std::stod(value));
            } catch (const std::exception&) {
                std::cerr << "Error parsing value: " << value << std::endl;
                return false;
            }
        }
        if (!row.empty()) {
            temp_data.push_back(row);
        }
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
    file.close();
    return true;
}

bool find_matrix_files(const std::string& folder, std::string& file_A, std::string& file_B, size_t& N, size_t& K_A,
                       size_t& K_B, size_t& M) {
    DIR* dir = opendir(folder.c_str());
    if (!dir) {
        std::cerr << "Error: Cannot open directory " << folder << std::endl;
        return false;
    }
    std::regex pattern_A("A_(\\d+)x(\\d+)\\.txt");
    std::regex pattern_B("B_(\\d+)x(\\d+)\\.txt");
    std::smatch match;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        if (filename.find("_cond") != std::string::npos) {
            continue;
        }
        if (std::regex_match(filename,
            match,
            pattern_A
        )) {
            file_A = folder + "/" + filename;
            N = std::stoull(match[1].str());
            K_A = std::stoull(match[2].str());
        } else if (std::regex_match(filename,
            match,
            pattern_B
        )) {
            file_B = folder + "/" + filename;
            K_B = std::stoull(match[1].str());
            M = std::stoull(match[2].str());
        }
    }
    closedir(dir);
    if (file_A.empty() || file_B.empty()) {
        std::cerr << "Error: Could not find A_NxK.txt and B_KxM.txt files in " << folder << std::endl;
        return false;
    }
    if (K_A != K_B) {
        std::cerr << "Error: Inner dimensions don't match! A is " << N << "x" << K_A << ", B is " << K_B << "x" << M
                  << std::endl;
        return false;
    }
    return true;
}

double compute_frobenius_diff(const double* A, const double* B, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = A[i] - B[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

double compute_frobenius_norm(const double* A, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += A[i] * A[i];
    }
    return std::sqrt(sum);
}

double compute_max_abs_diff(const double* A, const double* B, size_t n) {
    double max_diff = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = std::abs(A[i] - B[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

void compute_relative_error_stats(const double* C_ref, const double* C_test, size_t n, double& max_rel_err,
                                  double& avg_rel_err) {
    max_rel_err = 0.0;
    double sum_rel_err = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < n; i++) {
        double ref_val = std::abs(C_ref[i]);
        if (ref_val > 1e-300) {
            double rel_err = std::abs(C_test[i] - C_ref[i]) / ref_val;
            if (rel_err > max_rel_err) {
                max_rel_err = rel_err;
            }
            sum_rel_err += rel_err;
            count++;
        }
    }
    avg_rel_err = (count > 0) ? sum_rel_err / count : 0.0;
}

std::string get_device_name() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop,
        0
    ));
    std::string name = prop.name;
    for (char& c : name) {
        if (c == ' ' || c == '/' || c == '\\') {
            c = '_';
        }
    }
    return name;
}

std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time),
        "%Y-%m-%d_%H-%M-%S"
    );
    return ss.str();
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <matrix_folder>" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
        print_usage(argv[0]);
        return (argc < 2) ? 1 : 0;
    }
    std::string matrix_folder = argv[1];
    if (!matrix_folder.empty() && matrix_folder.back() == '/') {
        matrix_folder.pop_back();
    }
    std::cout << "============================================" << std::endl;
    std::cout << "Simple GEMM Benchmark (CUDA): FP64 vs GEMMul8" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Matrix folder: " << matrix_folder << std::endl;
    std::string file_A, file_B;
    size_t N, K_A, K_B, M;
    if (!find_matrix_files(matrix_folder,
        file_A,
        file_B,
        N,
        K_A,
        K_B,
        M
    )) {
        return 1;
    }
    size_t K = K_A;
    std::cout << "Found matrices:" << std::endl;
    std::cout << "  A: " << file_A << " (" << N << " x " << K << ")" << std::endl;
    std::cout << "  B: " << file_B << " (" << K << " x " << M << ")" << std::endl;
    std::cout << "  C: " << N << " x " << M << " (result)" << std::endl << std::endl;
    std::string device_name = get_device_name();
    std::string timestamp = get_timestamp();
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop,
        0
    ));
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Memory: " << prop.totalGlobalMem / (1024 * 1024 * 1024.0) << " GB" << std::endl;
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem,
        &total_mem
    ));
    std::cout << "Free memory: " << free_mem / (1024 * 1024 * 1024.0) << " GB" << std::endl << std::endl;
    std::vector<double> h_A;
    size_t rows_A, cols_A;
    if (!read_matrix_from_file(file_A,
        h_A,
        rows_A,
        cols_A
    )) {
        return 1;
    }
    std::vector<double> h_B;
    size_t rows_B, cols_B;
    if (!read_matrix_from_file(file_B,
        h_B,
        rows_B,
        cols_B
    )) {
        return 1;
    }
    if (cols_A != rows_B) {
        std::cerr << "Error: Matrix dimensions incompatible for multiplication!" << std::endl;
        return 1;
    }
    N = rows_A;
    K = cols_A;
    M = cols_B;
    size_t size_A = N * K;
    size_t size_B = K * M;
    size_t size_C = N * M;
    size_t mem_A = size_A * sizeof(double);
    size_t mem_B = size_B * sizeof(double);
    size_t mem_C = size_C * sizeof(double);
    size_t mem_fp64_total = mem_A + mem_B + mem_C;
    unsigned max_moduli = *std::max_element(NUM_MODULI_LIST.begin(),
        NUM_MODULI_LIST.end()
    );
    unsigned min_moduli = *std::min_element(NUM_MODULI_LIST.begin(),
        NUM_MODULI_LIST.end()
    );
    size_t worksize_max = gemmul8::workSize<false, gemmul8::Backend::INT8>(N,
        M,
        K,
        max_moduli
    );
    size_t worksize_min = gemmul8::workSize<false, gemmul8::Backend::INT8>(N,
        M,
        K,
        min_moduli
    );
    size_t mem_ozaki_total_max = mem_A + mem_B + mem_C + worksize_max;
    std::cout << std::endl;
    std::cout << "GEMM: C(" << N << "x" << M << ") = A(" << N << "x" << K << ") * B(" << K << "x" << M << ")" << std::endl;
    std::cout << "FLOPs per GEMM: " << 2.0 * N * M * K / 1e9 << " GFLOP" << std::endl;
    std::cout << "\n--- Memory Requirements ---" << std::endl;
    std::cout << "  Native FP64 (A+B+C): " << mem_fp64_total / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Ozaki workspace: " << worksize_min / (1024.0 * 1024.0) << " – "
              << worksize_max / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Ozaki total max: " << mem_ozaki_total_max / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl
              << std::endl;
    double* d_A, *d_B, *d_C_blas, *d_C_ozaki;
    CUDA_CHECK(cudaMalloc(&d_A,
        size_A * sizeof(double)
    ));
    CUDA_CHECK(cudaMalloc(&d_B,
        size_B * sizeof(double)
    ));
    CUDA_CHECK(cudaMalloc(&d_C_blas,
        size_C * sizeof(double)
    ));
    CUDA_CHECK(cudaMalloc(&d_C_ozaki,
        size_C * sizeof(double)
    ));
    CUDA_CHECK(cudaMemcpy(d_A,
        h_A.data(),
        size_A * sizeof(double),
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(d_B,
        h_B.data(),
        size_B * sizeof(double),
        cudaMemcpyHostToDevice
    ));
    std::vector<double> h_C_blas(size_C);
    std::vector<double> h_C_ozaki(size_C);
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    double alpha = 1.0;
    double beta = 0.0;
    std::cout << "\n--- cuBLAS DGEMM (FP64) ---" << std::endl;
    for (int i = 0; i < NUM_WARMUP; i++) {
        CUBLAS_CHECK(cublasDgemm(handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            (int)N,
            (int)M,
            (int)K,
            &alpha,
            d_A,
            (int)N,
            d_B,
            (int)K,
            &beta,
            d_C_blas,
            (int)N
        ));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        CUBLAS_CHECK(cublasDgemm(handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            (int)N,
            (int)M,
            (int)K,
            &alpha,
            d_A,
            (int)N,
            d_B,
            (int)K,
            &beta,
            d_C_blas,
            (int)N
        ));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    double cublas_time = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
    double cublas_tflops = (2.0 * N * M * K) / (cublas_time * 1e12);
    std::cout << "  Time: " << cublas_time * 1000.0 << " ms" << std::endl;
    std::cout << "  Performance: " << cublas_tflops << " TFLOPS" << std::endl;
    CUDA_CHECK(cudaMemcpy(h_C_blas.data(),
        d_C_blas,
        size_C * sizeof(double),
        cudaMemcpyDeviceToHost
    ));
    std::cout << "\n--- Ozaki-II (GEMMul8) ---" << std::endl;
    void* d_work;
    CUDA_CHECK(cudaMalloc(&d_work,
        worksize_max
    ));
    std::string results_filename =
        matrix_folder + "/simple_gemm_benchmark_cuda_" + device_name + "_" + timestamp + ".csv";
    std::ofstream results_file(results_filename);
    results_file << std::fixed << std::setprecision(6);
    results_file << "# Device: " << prop.name << std::endl;
    results_file << "method,num_moduli,fastmode,time_ms,tflops,workspace_mb,frob_rel_err,max_rel_err,avg_rel_err,max_abs_diff"
                 << std::endl;
    results_file << "cuBLAS_FP64,0,N/A," << cublas_time * 1000.0 << "," << cublas_tflops << ",0,0,0,0,0" << std::endl;
    for (unsigned num_moduli : NUM_MODULI_LIST) {
        size_t worksize_this = gemmul8::workSize<false, gemmul8::Backend::INT8>(N,
            M,
            K,
            num_moduli
        );
        for (int fast_mode = 0; fast_mode <= 1; fast_mode++) {
            bool fastmode = (fast_mode == 1);
            std::string mode_str = fastmode ? "fast" : "accurate";
            for (int i = 0; i < NUM_WARMUP; i++) {
                gemmul8::gemm<double>(handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    N,
                    M,
                    K,
                    &alpha,
                    d_A,
                    N,
                    d_B,
                    K,
                    &beta,
                    d_C_ozaki,
                    N,
                    num_moduli,
                    fastmode,
                    d_work
                );
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < NUM_ITERATIONS; i++) {
                gemmul8::gemm<double>(handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    N,
                    M,
                    K,
                    &alpha,
                    d_A,
                    N,
                    d_B,
                    K,
                    &beta,
                    d_C_ozaki,
                    N,
                    num_moduli,
                    fastmode,
                    d_work
                );
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            end = std::chrono::high_resolution_clock::now();
            double ozaki_time = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
            double ozaki_tflops = (2.0 * N * M * K) / (ozaki_time * 1e12);
            CUDA_CHECK(cudaMemcpy(h_C_ozaki.data(),
                d_C_ozaki,
                size_C * sizeof(double),
                cudaMemcpyDeviceToHost
            ));
            double frob_diff = compute_frobenius_diff(h_C_blas.data(),
                h_C_ozaki.data(),
                size_C
            );
            double frob_norm = compute_frobenius_norm(h_C_blas.data(),
                size_C
            );
            double frob_rel_err = (frob_norm > 0) ? frob_diff / frob_norm : 0.0;
            double max_abs_diff = compute_max_abs_diff(h_C_blas.data(),
                h_C_ozaki.data(),
                size_C
            );
            double max_rel_err, avg_rel_err;
            compute_relative_error_stats(h_C_blas.data(),
                h_C_ozaki.data(),
                size_C,
                max_rel_err,
                avg_rel_err
            );
            std::cout << "  Moduli=" << num_moduli << " (" << mode_str << "): " << ozaki_time * 1000.0 << " ms, "
                      << ozaki_tflops << " TFLOPS"
                      << ", rel_err=" << std::scientific << frob_rel_err << std::fixed << std::endl;
            results_file << "Ozaki-II_EMU," << num_moduli << "," << mode_str << "," << ozaki_time * 1000.0 << ","
                         << ozaki_tflops << "," << worksize_this / (1024.0 * 1024.0) << "," << std::scientific
                         << frob_rel_err << "," << max_rel_err << "," << avg_rel_err << "," << max_abs_diff
                         << std::fixed << std::endl;
        }
    }
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_blas));
    CUDA_CHECK(cudaFree(d_C_ozaki));
    CUBLAS_CHECK(cublasDestroy(handle));
    results_file.close();
    std::cout << "\nResults saved to: " << results_filename << std::endl;
    return 0;
}
