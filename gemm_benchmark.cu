/**
 * GEMM Benchmark: rocBLAS vs Ozaki-II (GEMMul8)
 * 
 * Reads matrices from files and performs:
 * 1. Standard DGEMM using rocBLAS (hipblasDgemm)
 * 2. Ozaki-II GEMM using GEMMul8 library
 * 
 * Computes C = A * A (matrix multiplied with itself)
 * 
 * Usage:
 *   ./gemm_benchmark [matrix_folder]
 *   
 *   matrix_folder - Folder containing M_cond_1e*.txt files (default: current directory)
 * 
 * Compile with:
 *   hipcc -O3 -std=c++17 gemm_benchmark.cu -o gemm_benchmark \
 *         -I/home/kswirydo/GEMMul8/GEMMul8/include \
 *         -L/home/kswirydo/GEMMul8/GEMMul8/lib \
 *         -lgemmul8 -lhipblas -lamdhip64 \
 *         -I/opt/rocm/include -L/opt/rocm/lib
 */

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include "gemmul8.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <numeric>

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

// Configuration
static const int NUM_WARMUP = 2;
static const int NUM_ITERATIONS = 5;
static const std::vector<unsigned> NUM_MODULI_LIST = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

/**
 * Read a matrix from CSV file
 * Returns dimensions and data in column-major format
 */
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
        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;
        
        while (std::getline(ss, value, ',')) {
            try {
                row.push_back(std::stod(value));
            } catch (const std::exception& e) {
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
    
    // Convert to column-major format
    data.resize(rows * cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            data[j * rows + i] = temp_data[i][j];  // Column-major
        }
    }
    
    file.close();
    return true;
}

/**
 * Compute Frobenius norm of difference between two matrices
 */
double compute_frobenius_diff(const double* A, const double* B, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = A[i] - B[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

/**
 * Compute Frobenius norm of a matrix
 */
double compute_frobenius_norm(const double* A, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += A[i] * A[i];
    }
    return std::sqrt(sum);
}

/**
 * Compute max absolute difference
 */
double compute_max_abs_diff(const double* A, const double* B, size_t n) {
    double max_diff = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = std::abs(A[i] - B[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

/**
 * Compute element-wise relative error statistics
 */
void compute_relative_error_stats(const double* C_ref, const double* C_test, 
                                   size_t n, double& max_rel_err, double& avg_rel_err) {
    max_rel_err = 0.0;
    double sum_rel_err = 0.0;
    size_t count = 0;
    
    for (size_t i = 0; i < n; i++) {
        double ref_val = std::abs(C_ref[i]);
        if (ref_val > 1e-300) {  // Avoid division by very small numbers
            double rel_err = std::abs(C_test[i] - C_ref[i]) / ref_val;
            if (rel_err > max_rel_err) max_rel_err = rel_err;
            sum_rel_err += rel_err;
            count++;
        }
    }
    
    avg_rel_err = (count > 0) ? sum_rel_err / count : 0.0;
}

void benchmark_matrix(hipblasHandle_t handle, 
                      const std::string& filename,
                      int log10_cond,
                      std::ofstream& results_file) {
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Processing: " << filename << std::endl;
    std::cout << "Condition number: 10^" << log10_cond << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Read matrix from file
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
    
    // Allocate device memory
    double *d_A, *d_C_rocblas, *d_C_ozaki;
    HIP_CHECK(hipMalloc(&d_A, matrix_bytes));
    HIP_CHECK(hipMalloc(&d_C_rocblas, matrix_bytes));
    HIP_CHECK(hipMalloc(&d_C_ozaki, matrix_bytes));
    
    // Copy A to device
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), matrix_bytes, hipMemcpyHostToDevice));
    
    // Allocate host memory for results
    std::vector<double> h_C_rocblas(matrix_size);
    std::vector<double> h_C_ozaki(matrix_size);
    
    double alpha = 1.0;
    double beta = 0.0;
    
    // =========================================
    // rocBLAS DGEMM: C = A * A
    // =========================================
    std::cout << "\n--- rocBLAS DGEMM ---" << std::endl;
    
    // Warmup
    for (int i = 0; i < NUM_WARMUP; i++) {
        HIPBLAS_CHECK(hipblasDgemm(handle,
                                   HIPBLAS_OP_N, HIPBLAS_OP_N,
                                   m, n, n,
                                   &alpha,
                                   d_A, m,
                                   d_A, n,
                                   &beta,
                                   d_C_rocblas, m));
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        HIPBLAS_CHECK(hipblasDgemm(handle,
                                   HIPBLAS_OP_N, HIPBLAS_OP_N,
                                   m, n, n,
                                   &alpha,
                                   d_A, m,
                                   d_A, n,
                                   &beta,
                                   d_C_rocblas, m));
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double rocblas_time = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
    double rocblas_gflops = (2.0 * m * n * n) / (rocblas_time * 1e9);
    double rocblas_tflops = rocblas_gflops / 1000.0;
    
    std::cout << "  Time: " << rocblas_time * 1000.0 << " ms" << std::endl;
    std::cout << "  Performance: " << rocblas_tflops << " TFLOPS" << std::endl;
    
    // Copy result to host (reference)
    HIP_CHECK(hipMemcpy(h_C_rocblas.data(), d_C_rocblas, matrix_bytes, hipMemcpyDeviceToHost));
    
    // =========================================
    // Ozaki-II GEMM using GEMMul8
    // =========================================
    std::cout << "\n--- Ozaki-II GEMM (GEMMul8) ---" << std::endl;
    
    // Get maximum workspace size needed
    unsigned max_moduli = *std::max_element(NUM_MODULI_LIST.begin(), NUM_MODULI_LIST.end());
    size_t worksize = gemmul8::workSize<false>(m, n, n, max_moduli);
    
    void* d_work;
    HIP_CHECK(hipMalloc(&d_work, worksize));
    
    // Write CSV header for this matrix
    results_file << "\n# Condition number: 10^" << log10_cond << ", Size: " << m << "x" << n << std::endl;
    results_file << "method,num_moduli,fastmode,time_ms,tflops,frob_rel_err,max_rel_err,avg_rel_err" << std::endl;
    
    // Write rocBLAS result
    results_file << "rocBLAS,0,N/A," 
                 << rocblas_time * 1000.0 << "," 
                 << rocblas_tflops << ",0,0,0" << std::endl;
    
    // Test different number of moduli and both fast/accurate modes
    for (unsigned num_moduli : NUM_MODULI_LIST) {
        for (int fast_mode = 0; fast_mode <= 1; fast_mode++) {
            bool fastmode = (fast_mode == 1);
            std::string mode_str = fastmode ? "fast" : "accurate";
            
            // Warmup
            for (int i = 0; i < NUM_WARMUP; i++) {
                gemmul8::gemm<double>(handle,
                                      HIPBLAS_OP_N, HIPBLAS_OP_N,
                                      m, n, n,
                                      &alpha,
                                      d_A, m,
                                      d_A, n,
                                      &beta,
                                      d_C_ozaki, m,
                                      num_moduli,
                                      fastmode,
                                      d_work);
            }
            HIP_CHECK(hipDeviceSynchronize());
            
            // Benchmark
            start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < NUM_ITERATIONS; i++) {
                gemmul8::gemm<double>(handle,
                                      HIPBLAS_OP_N, HIPBLAS_OP_N,
                                      m, n, n,
                                      &alpha,
                                      d_A, m,
                                      d_A, n,
                                      &beta,
                                      d_C_ozaki, m,
                                      num_moduli,
                                      fastmode,
                                      d_work);
            }
            HIP_CHECK(hipDeviceSynchronize());
            end = std::chrono::high_resolution_clock::now();
            
            double ozaki_time = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
            double ozaki_gflops = (2.0 * m * n * n) / (ozaki_time * 1e9);
            double ozaki_tflops = ozaki_gflops / 1000.0;
            
            // Copy result to host
            HIP_CHECK(hipMemcpy(h_C_ozaki.data(), d_C_ozaki, matrix_bytes, hipMemcpyDeviceToHost));
            
            // Compute error metrics (comparing to rocBLAS as reference)
            double frob_diff = compute_frobenius_diff(h_C_rocblas.data(), h_C_ozaki.data(), matrix_size);
            double frob_norm = compute_frobenius_norm(h_C_rocblas.data(), matrix_size);
            double frob_rel_err = (frob_norm > 0) ? frob_diff / frob_norm : 0.0;
            
            double max_rel_err, avg_rel_err;
            compute_relative_error_stats(h_C_rocblas.data(), h_C_ozaki.data(), 
                                         matrix_size, max_rel_err, avg_rel_err);
            
            std::cout << "  Moduli=" << num_moduli << " (" << mode_str << "): "
                      << ozaki_time * 1000.0 << " ms, "
                      << ozaki_tflops << " TFLOPS, "
                      << "rel_err=" << std::scientific << frob_rel_err << std::fixed
                      << std::endl;
            
            // Write to results file
            results_file << "Ozaki-II," << num_moduli << "," << mode_str << ","
                         << ozaki_time * 1000.0 << ","
                         << ozaki_tflops << ","
                         << std::scientific << frob_rel_err << ","
                         << max_rel_err << ","
                         << avg_rel_err << std::fixed << std::endl;
        }
    }
    
    // Cleanup
    HIP_CHECK(hipFree(d_work));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_C_rocblas));
    HIP_CHECK(hipFree(d_C_ozaki));
    
    std::cout << "Completed benchmark for condition number 10^" << log10_cond << std::endl;
}

std::string get_device_name() {
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::string name = prop.name;
    
    // Replace spaces with underscores
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
    std::cout << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  matrix_folder - Folder containing M_cond_1e*.txt files (default: current directory)" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << "                                    # Read from current directory" << std::endl;
    std::cout << "  " << program_name << " matrices_4096x4096_20260128_143022 # Read from specified folder" << std::endl;
}

int main(int argc, char* argv[]) {
    // Check for help flag
    if (argc >= 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        print_usage(argv[0]);
        return 0;
    }
    
    // Parse matrix folder argument
    std::string matrix_folder = ".";  // Default: current directory
    if (argc >= 2) {
        matrix_folder = argv[1];
        // Remove trailing slash if present
        if (!matrix_folder.empty() && matrix_folder.back() == '/') {
            matrix_folder.pop_back();
        }
    }
    
    std::cout << "============================================" << std::endl;
    std::cout << "GEMM Benchmark: rocBLAS vs Ozaki-II (GEMMul8)" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Matrix folder: " << matrix_folder << std::endl;
    
    // Get device info
    std::string device_name = get_device_name();
    std::string timestamp = get_timestamp();
    
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Memory: " << prop.totalGlobalMem / (1024*1024*1024.0) << " GB" << std::endl;
    
    // Check available memory
    size_t free_mem, total_mem;
    HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));
    std::cout << "Free memory: " << free_mem / (1024*1024*1024.0) << " GB" << std::endl;
    std::cout << std::endl;
    
    // Create hipBLAS handle
    hipblasHandle_t handle;
    HIPBLAS_CHECK(hipblasCreate(&handle));
    
    // Output file for results (save in the matrix folder)
    std::string results_filename = matrix_folder + "/gemm_benchmark_" + device_name + "_" + timestamp + ".csv";
    std::ofstream results_file(results_filename);
    results_file << std::fixed << std::setprecision(6);
    
    results_file << "# GEMM Benchmark Results" << std::endl;
    results_file << "# Device: " << prop.name << std::endl;
    results_file << "# Date: " << timestamp << std::endl;
    results_file << "# Matrix folder: " << matrix_folder << std::endl;
    results_file << "# Warmup iterations: " << NUM_WARMUP << std::endl;
    results_file << "# Benchmark iterations: " << NUM_ITERATIONS << std::endl;
    
    // Count how many matrices were processed
    int matrices_processed = 0;
    
    // Process each matrix file
    // Condition numbers from 10^2 to 10^17 (matching matrix_generator.cpp)
    for (int log10_cond = 2; log10_cond <= 17; log10_cond++) {
        std::string filepath = matrix_folder + "/M_cond_1e" + std::to_string(log10_cond) + ".txt";
        
        // Check if file exists
        std::ifstream test_file(filepath);
        if (!test_file.good()) {
            std::cout << "File not found: " << filepath << ", skipping..." << std::endl;
            continue;
        }
        test_file.close();
        
        benchmark_matrix(handle, filepath, log10_cond, results_file);
        matrices_processed++;
    }
    
    // Cleanup
    HIPBLAS_CHECK(hipblasDestroy(handle));
    results_file.close();
    
    std::cout << "\n============================================" << std::endl;
    std::cout << "Benchmark complete!" << std::endl;
    std::cout << "Matrices processed: " << matrices_processed << std::endl;
    std::cout << "Results saved to: " << results_filename << std::endl;
    std::cout << "============================================" << std::endl;
    
    return 0;
}
