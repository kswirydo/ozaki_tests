/**
 * GEMM Benchmark: cuBLAS vs Ozaki-II (GEMMul8) — NVIDIA CUDA port of ../gemm_benchmark.cu
 *
 * Requires GEMMul8 built for CUDA (headers + lib linking against cuBLAS).
 *
 * Compile (adjust GEMMUL8_PATH):
 *   nvcc -O3 -std=c++17 gemm_benchmark.cu -o gemm_benchmark \
 *        -I${GEMMUL8_PATH}/include -L${GEMMUL8_PATH}/lib -lgemmul8 \
 *        -lcublas -lcudart
 */

#include "cuda_helpers.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
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
    while (std::getline(file, line)) {
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

void benchmark_matrix(cublasHandle_t handle, const std::string& filename, int log10_cond, std::ofstream& results_file) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Processing: " << filename << std::endl;
    std::cout << "Condition number: 10^" << log10_cond << std::endl;
    std::cout << "========================================" << std::endl;
    std::vector<double> h_A;
    size_t m, n;
    if (!read_matrix_from_file(filename, h_A, m, n)) {
        std::cerr << "Skipping " << filename << std::endl;
        return;
    }
    std::cout << "Matrix size: " << m << " x " << n << std::endl;
    if (m != n) {
        std::cerr << "Error: Matrix must be square for A*A operation" << std::endl;
        return;
    }
    size_t matrix_size = m * n;
    size_t matrix_bytes = matrix_size * sizeof(double);
    double *d_A, *d_C_ref, *d_C_ozaki;
    CUDA_CHECK(cudaMalloc(&d_A, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_C_ref, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_C_ozaki, matrix_bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), matrix_bytes, cudaMemcpyHostToDevice));
    std::vector<double> h_C_ref(matrix_size);
    std::vector<double> h_C_ozaki(matrix_size);
    double alpha = 1.0;
    double beta = 0.0;
    std::cout << "\n--- cuBLAS DGEMM (FP64 reference) ---" << std::endl;
    for (int i = 0; i < NUM_WARMUP; i++) {
        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)m, (int)n, (int)n, &alpha, d_A, (int)m, d_A,
                                 (int)n, &beta, d_C_ref, (int)m));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)m, (int)n, (int)n, &alpha, d_A, (int)m, d_A,
                                 (int)n, &beta, d_C_ref, (int)m));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    double cublas_time = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
    double cublas_gflops = (2.0 * m * n * n) / (cublas_time * 1e9);
    double cublas_tflops = cublas_gflops / 1000.0;
    std::cout << "  Time: " << cublas_time * 1000.0 << " ms" << std::endl;
    std::cout << "  Performance: " << cublas_tflops << " TFLOPS" << std::endl;
    CUDA_CHECK(cudaMemcpy(h_C_ref.data(), d_C_ref, matrix_bytes, cudaMemcpyDeviceToHost));
    std::cout << "\n--- Ozaki-II GEMM (GEMMul8) ---" << std::endl;
    unsigned max_moduli = *std::max_element(NUM_MODULI_LIST.begin(), NUM_MODULI_LIST.end());
    size_t worksize = gemmul8::workSize(m, n, n, max_moduli);
    void* d_work;
    CUDA_CHECK(cudaMalloc(&d_work, worksize));
    results_file << "\n# Condition number: 10^" << log10_cond << ", Size: " << m << "x" << n << std::endl;
    results_file << "method,num_moduli,fastmode,time_ms,tflops,frob_rel_err,max_rel_err,avg_rel_err" << std::endl;
    results_file << "cuBLAS_FP64,0,N/A," << cublas_time * 1000.0 << "," << cublas_tflops << ",0,0,0" << std::endl;
    for (unsigned num_moduli : NUM_MODULI_LIST) {
        for (int fast_mode = 0; fast_mode <= 1; fast_mode++) {
            bool fastmode = (fast_mode == 1);
            std::string mode_str = fastmode ? "fast" : "accurate";
            for (int i = 0; i < NUM_WARMUP; i++) {
                gemmul8::gemm<double>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, &alpha, d_A, m, d_A, n, &beta,
                                      d_C_ozaki, m, num_moduli, fastmode, d_work);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < NUM_ITERATIONS; i++) {
                gemmul8::gemm<double>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, &alpha, d_A, m, d_A, n, &beta,
                                      d_C_ozaki, m, num_moduli, fastmode, d_work);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            end = std::chrono::high_resolution_clock::now();
            double ozaki_time = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
            double ozaki_gflops = (2.0 * m * n * n) / (ozaki_time * 1e9);
            double ozaki_tflops = ozaki_gflops / 1000.0;
            CUDA_CHECK(cudaMemcpy(h_C_ozaki.data(), d_C_ozaki, matrix_bytes, cudaMemcpyDeviceToHost));
            double frob_diff = compute_frobenius_diff(h_C_ref.data(), h_C_ozaki.data(), matrix_size);
            double frob_norm = compute_frobenius_norm(h_C_ref.data(), matrix_size);
            double frob_rel_err = (frob_norm > 0) ? frob_diff / frob_norm : 0.0;
            double max_rel_err, avg_rel_err;
            compute_relative_error_stats(h_C_ref.data(), h_C_ozaki.data(), matrix_size, max_rel_err, avg_rel_err);
            std::cout << "  Moduli=" << num_moduli << " (" << mode_str << "): " << ozaki_time * 1000.0 << " ms, "
                      << ozaki_tflops << " TFLOPS, "
                      << "rel_err=" << std::scientific << frob_rel_err << std::fixed << std::endl;
            results_file << "Ozaki-II," << num_moduli << "," << mode_str << "," << ozaki_time * 1000.0 << ","
                         << ozaki_tflops << "," << std::scientific << frob_rel_err << "," << max_rel_err << ","
                         << avg_rel_err << std::fixed << std::endl;
        }
    }
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C_ref));
    CUDA_CHECK(cudaFree(d_C_ozaki));
    std::cout << "Completed benchmark for condition number 10^" << log10_cond << std::endl;
}

std::string get_device_name() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
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
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d_%H-%M-%S");
    return ss.str();
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [matrix_folder]" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc >= 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        print_usage(argv[0]);
        return 0;
    }
    std::string matrix_folder = ".";
    if (argc >= 2) {
        matrix_folder = argv[1];
        if (!matrix_folder.empty() && matrix_folder.back() == '/') {
            matrix_folder.pop_back();
        }
    }
    std::cout << "============================================" << std::endl;
    std::cout << "GEMM Benchmark: cuBLAS vs Ozaki-II (GEMMul8)" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Matrix folder: " << matrix_folder << std::endl;
    std::string device_name = get_device_name();
    std::string timestamp = get_timestamp();
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Memory: " << prop.totalGlobalMem / (1024 * 1024 * 1024.0) << " GB" << std::endl;
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    std::cout << "Free memory: " << free_mem / (1024 * 1024 * 1024.0) << " GB" << std::endl << std::endl;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    std::string results_filename = matrix_folder + "/gemm_benchmark_cuda_" + device_name + "_" + timestamp + ".csv";
    std::ofstream results_file(results_filename);
    results_file << std::fixed << std::setprecision(6);
    results_file << "# GEMM Benchmark Results (CUDA)" << std::endl;
    results_file << "# Device: " << prop.name << std::endl;
    results_file << "# Date: " << timestamp << std::endl;
    results_file << "# Matrix folder: " << matrix_folder << std::endl;
    results_file << "# Warmup iterations: " << NUM_WARMUP << std::endl;
    results_file << "# Benchmark iterations: " << NUM_ITERATIONS << std::endl;
    int matrices_processed = 0;
    for (int log10_cond = 2; log10_cond <= 17; log10_cond++) {
        std::string filepath = matrix_folder + "/M_cond_1e" + std::to_string(log10_cond) + ".txt";
        std::ifstream test_file(filepath);
        if (!test_file.good()) {
            std::cout << "File not found: " << filepath << ", skipping..." << std::endl;
            continue;
        }
        test_file.close();
        benchmark_matrix(handle, filepath, log10_cond, results_file);
        matrices_processed++;
    }
    CUBLAS_CHECK(cublasDestroy(handle));
    results_file.close();
    std::cout << "\n============================================" << std::endl;
    std::cout << "Benchmark complete!" << std::endl;
    std::cout << "Matrices processed: " << matrices_processed << std::endl;
    std::cout << "Results saved to: " << results_filename << std::endl;
    std::cout << "============================================" << std::endl;
    return 0;
}
