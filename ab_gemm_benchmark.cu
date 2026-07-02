/**
 * A/B GEMM Benchmark: rocBLAS (native FP64) vs Ozaki-II (GEMMul8 emulated)
 * 
 * Reads matrix pairs A and B from files and performs C = A * B using:
 * 1. Standard DGEMM using rocBLAS (hipblasDgemm) - native FP64
 * 2. Ozaki-II GEMM using GEMMul8 library - emulated via INT8 Tensor Cores
 * 
 * Processes all matrix pairs in the folder with different condition numbers.
 * 
 * Usage:
 *   ./ab_gemm_benchmark <matrix_folder>
 *   
 *   matrix_folder - Folder containing matrix pairs:
 *     - Condition number: A_*_cond1e*.txt and B_*_cond1e*.txt
 *     - Aspect ratio:     A_*_ratio1e*.txt and B_*_ratio1e*.txt
 */

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include "gemmul8.hpp"

// Include internal header to access profiling flag
// This allows GEMMUL8_PROFILE=1 to work
namespace oz2 { extern bool g_profiling_enabled; }

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
#include <dirent.h>
#include <regex>
#include <map>
#include <cstdlib>

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
            data[j * rows + i] = temp_data[i][j];
        }
    }
    
    file.close();
    return true;
}

// Structure to hold matrix pair info
struct MatrixPair {
    std::string file_A;
    std::string file_B;
    int log10_cond;
    size_t N, K, M;
};

/**
 * Find all matrix pairs in folder
 */
std::vector<MatrixPair> find_matrix_pairs(const std::string& folder) {
    std::vector<MatrixPair> pairs;
    std::map<int, MatrixPair> pair_map_cond;  // For condition number matrices
    std::map<int, MatrixPair> pair_map_ratio; // For aspect ratio matrices
    
    DIR* dir = opendir(folder.c_str());
    if (!dir) {
        std::cerr << "Error: Cannot open directory " << folder << std::endl;
        return pairs;
    }
    
    // Patterns for condition number matrices
    std::regex pattern_A_cond("A_(\\d+)x(\\d+)_cond1e(\\d+)\\.txt");
    std::regex pattern_B_cond("B_(\\d+)x(\\d+)_cond1e(\\d+)\\.txt");
    // Patterns for aspect ratio matrices
    std::regex pattern_A_ratio("A_(\\d+)x(\\d+)_ratio1e(\\d+)\\.txt");
    std::regex pattern_B_ratio("B_(\\d+)x(\\d+)_ratio1e(\\d+)\\.txt");
    std::smatch match;
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        
        // Check condition number patterns
        if (std::regex_match(filename, match, pattern_A_cond)) {
            int cond = std::stoi(match[3].str());
            pair_map_cond[cond].file_A = folder + "/" + filename;
            pair_map_cond[cond].N = std::stoull(match[1].str());
            pair_map_cond[cond].K = std::stoull(match[2].str());
            pair_map_cond[cond].log10_cond = cond;
        }
        else if (std::regex_match(filename, match, pattern_B_cond)) {
            int cond = std::stoi(match[3].str());
            pair_map_cond[cond].file_B = folder + "/" + filename;
            size_t K_B = std::stoull(match[1].str());
            pair_map_cond[cond].M = std::stoull(match[2].str());
            if (pair_map_cond[cond].K != 0 && pair_map_cond[cond].K != K_B) {
                std::cerr << "Warning: K dimension mismatch for condition 1e" << cond << std::endl;
            }
        }
        // Check aspect ratio patterns
        else if (std::regex_match(filename, match, pattern_A_ratio)) {
            int ratio = std::stoi(match[3].str());
            pair_map_ratio[ratio].file_A = folder + "/" + filename;
            pair_map_ratio[ratio].N = std::stoull(match[1].str());
            pair_map_ratio[ratio].K = std::stoull(match[2].str());
            pair_map_ratio[ratio].log10_cond = ratio;  // Reuse field for aspect ratio
        }
        else if (std::regex_match(filename, match, pattern_B_ratio)) {
            int ratio = std::stoi(match[3].str());
            pair_map_ratio[ratio].file_B = folder + "/" + filename;
            size_t K_B = std::stoull(match[1].str());
            pair_map_ratio[ratio].M = std::stoull(match[2].str());
            if (pair_map_ratio[ratio].K != 0 && pair_map_ratio[ratio].K != K_B) {
                std::cerr << "Warning: K dimension mismatch for ratio 1e" << ratio << std::endl;
            }
        }
    }
    
    closedir(dir);
    
    // Collect condition number pairs
    for (auto& kv : pair_map_cond) {
        if (!kv.second.file_A.empty() && !kv.second.file_B.empty()) {
            pairs.push_back(kv.second);
        }
    }
    
    // Collect aspect ratio pairs
    for (auto& kv : pair_map_ratio) {
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
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

void compute_relative_error_stats(const double* C_ref, const double* C_test, 
                                   size_t n, double& max_rel_err, double& avg_rel_err) {
    max_rel_err = 0.0;
    double sum_rel_err = 0.0;
    size_t count = 0;
    
    for (size_t i = 0; i < n; i++) {
        double ref_val = std::abs(C_ref[i]);
        if (ref_val > 1e-300) {
            double rel_err = std::abs(C_test[i] - C_ref[i]) / ref_val;
            if (rel_err > max_rel_err) max_rel_err = rel_err;
            sum_rel_err += rel_err;
            count++;
        }
    }
    
    avg_rel_err = (count > 0) ? sum_rel_err / count : 0.0;
}

std::string get_device_name() {
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::string name = prop.name;
    for (char& c : name) {
        if (c == ' ' || c == '/' || c == '\\') c = '_';
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

void benchmark_matrix_pair(hipblasHandle_t handle, const MatrixPair& pair, std::ofstream& results_file) {
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Condition number: 10^" << pair.log10_cond << std::endl;
    std::cout << "  A: " << pair.file_A << std::endl;
    std::cout << "  B: " << pair.file_B << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::vector<double> h_A, h_B;
    size_t rows_A, cols_A, rows_B, cols_B;
    
    if (!read_matrix_from_file(pair.file_A, h_A, rows_A, cols_A)) return;
    if (!read_matrix_from_file(pair.file_B, h_B, rows_B, cols_B)) return;
    
    if (cols_A != rows_B) {
        std::cerr << "Error: Dimension mismatch! A: " << rows_A << "x" << cols_A
                  << ", B: " << rows_B << "x" << cols_B << std::endl;
        return;
    }
    
    size_t N = rows_A, K = cols_A, M = cols_B;
    size_t size_A = N * K, size_B = K * M, size_C = N * M;
    
    // Memory requirements
    size_t mem_A = size_A * sizeof(double);
    size_t mem_B = size_B * sizeof(double);
    size_t mem_C = size_C * sizeof(double);
    size_t mem_fp64_total = mem_A + mem_B + mem_C;
    
    unsigned max_moduli = *std::max_element(NUM_MODULI_LIST.begin(), NUM_MODULI_LIST.end());
    unsigned min_moduli = *std::min_element(NUM_MODULI_LIST.begin(), NUM_MODULI_LIST.end());
    size_t worksize_max = gemmul8::workSize<false>(N, M, K, max_moduli);
    size_t worksize_min = gemmul8::workSize<false>(N, M, K, min_moduli);
    
    size_t mem_ozaki_total_min = mem_fp64_total + worksize_min;
    size_t mem_ozaki_total_max = mem_fp64_total + worksize_max;
    
    std::cout << "\n--- Memory Requirements ---" << std::endl;
    std::cout << "  Matrix A (" << N << "x" << K << "): " << std::fixed << std::setprecision(2) 
              << mem_A / (1024.0*1024.0) << " MB" << std::endl;
    std::cout << "  Matrix B (" << K << "x" << M << "): " << mem_B / (1024.0*1024.0) << " MB" << std::endl;
    std::cout << "  Matrix C (" << N << "x" << M << "): " << mem_C / (1024.0*1024.0) << " MB" << std::endl;
    std::cout << std::endl;
    std::cout << "  Native FP64 (rocBLAS):" << std::endl;
    std::cout << "    A + B + C = " << mem_fp64_total / (1024.0*1024.0) << " MB ("
              << mem_fp64_total / (1024.0*1024.0*1024.0) << " GB)" << std::endl;
    std::cout << std::endl;
    std::cout << "  Ozaki-II Emulated (GEMMul8):" << std::endl;
    std::cout << "    Workspace (" << min_moduli << " moduli): " 
              << worksize_min / (1024.0*1024.0) << " MB" << std::endl;
    std::cout << "    Workspace (" << max_moduli << " moduli): " 
              << worksize_max / (1024.0*1024.0) << " MB" << std::endl;
    std::cout << "    Total (" << min_moduli << " moduli): A + B + C + workspace = " 
              << mem_ozaki_total_min / (1024.0*1024.0) << " MB ("
              << mem_ozaki_total_min / (1024.0*1024.0*1024.0) << " GB)" << std::endl;
    std::cout << "    Total (" << max_moduli << " moduli): A + B + C + workspace = " 
              << mem_ozaki_total_max / (1024.0*1024.0) << " MB ("
              << mem_ozaki_total_max / (1024.0*1024.0*1024.0) << " GB)" << std::endl;
    std::cout << std::endl;
    std::cout << "  Memory overhead (Ozaki vs Native):" << std::endl;
    std::cout << "    " << min_moduli << " moduli: " << std::setprecision(1) 
              << (double)mem_ozaki_total_min / mem_fp64_total << "x" << std::endl;
    std::cout << "    " << max_moduli << " moduli: " 
              << (double)mem_ozaki_total_max / mem_fp64_total << "x" << std::endl;
    
    double *d_A, *d_B, *d_C_rocblas, *d_C_ozaki;
    HIP_CHECK(hipMalloc(&d_A, mem_A));
    HIP_CHECK(hipMalloc(&d_B, mem_B));
    HIP_CHECK(hipMalloc(&d_C_rocblas, mem_C));
    HIP_CHECK(hipMalloc(&d_C_ozaki, mem_C));
    
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), mem_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), mem_B, hipMemcpyHostToDevice));
    
    std::vector<double> h_C_rocblas(size_C), h_C_ozaki(size_C);
    double alpha = 1.0, beta = 0.0;
    
    // rocBLAS benchmark
    std::cout << "\n--- rocBLAS DGEMM (Native FP64) ---" << std::endl;
    
    for (int i = 0; i < NUM_WARMUP; i++) {
        HIPBLAS_CHECK(hipblasDgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K,
                                   &alpha, d_A, N, d_B, K, &beta, d_C_rocblas, N));
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        HIPBLAS_CHECK(hipblasDgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K,
                                   &alpha, d_A, N, d_B, K, &beta, d_C_rocblas, N));
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double rocblas_time = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
    double rocblas_tflops = (2.0 * N * M * K) / (rocblas_time * 1e12);
    
    std::cout << "  Time: " << rocblas_time * 1000.0 << " ms, " << rocblas_tflops << " TFLOPS" << std::endl;
    
    HIP_CHECK(hipMemcpy(h_C_rocblas.data(), d_C_rocblas, mem_C, hipMemcpyDeviceToHost));
    
    results_file << pair.log10_cond << ",rocBLAS_FP64,0,N/A," 
                 << rocblas_time * 1000.0 << "," << rocblas_tflops << ",0,0,0,0,0" << std::endl;
    
    // Ozaki-II benchmark
    std::cout << "\n--- Ozaki-II GEMM (Emulated FP64) ---" << std::endl;
    
    void* d_work;
    HIP_CHECK(hipMalloc(&d_work, worksize_max));
    
    for (unsigned num_moduli : NUM_MODULI_LIST) {
        size_t worksize_this = gemmul8::workSize<false>(N, M, K, num_moduli);
        
        for (int fast_mode = 0; fast_mode <= 1; fast_mode++) {
            bool fastmode = (fast_mode == 1);
            std::string mode_str = fastmode ? "fast" : "accurate";
            
            for (int i = 0; i < NUM_WARMUP; i++) {
                gemmul8::gemm<double>(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K,
                                      &alpha, d_A, N, d_B, K, &beta, d_C_ozaki, N,
                                      num_moduli, fastmode, d_work);
            }
            HIP_CHECK(hipDeviceSynchronize());
            
            start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < NUM_ITERATIONS; i++) {
                gemmul8::gemm<double>(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K,
                                      &alpha, d_A, N, d_B, K, &beta, d_C_ozaki, N,
                                      num_moduli, fastmode, d_work);
            }
            HIP_CHECK(hipDeviceSynchronize());
            end = std::chrono::high_resolution_clock::now();
            
            double ozaki_time = std::chrono::duration<double>(end - start).count() / NUM_ITERATIONS;
            double ozaki_tflops = (2.0 * N * M * K) / (ozaki_time * 1e12);
            
            HIP_CHECK(hipMemcpy(h_C_ozaki.data(), d_C_ozaki, mem_C, hipMemcpyDeviceToHost));
            
            double frob_diff = compute_frobenius_diff(h_C_rocblas.data(), h_C_ozaki.data(), size_C);
            double frob_norm = compute_frobenius_norm(h_C_rocblas.data(), size_C);
            double frob_rel_err = (frob_norm > 0) ? frob_diff / frob_norm : 0.0;
            double max_abs_diff = compute_max_abs_diff(h_C_rocblas.data(), h_C_ozaki.data(), size_C);
            double max_rel_err, avg_rel_err;
            compute_relative_error_stats(h_C_rocblas.data(), h_C_ozaki.data(), size_C, max_rel_err, avg_rel_err);
            
            std::cout << "  Moduli=" << std::setw(2) << num_moduli << " (" << std::setw(8) << mode_str << "): "
                      << std::setw(8) << std::fixed << std::setprecision(2) << ozaki_time * 1000.0 << " ms, "
                      << std::setw(6) << std::setprecision(3) << ozaki_tflops << " TFLOPS, "
                      << "ws=" << std::setprecision(0) << worksize_this / (1024.0*1024.0) << "MB, "
                      << "rel_err=" << std::scientific << std::setprecision(2) << frob_rel_err << std::fixed
                      << std::endl;
            
            results_file << pair.log10_cond << ",Ozaki-II_EMU," << num_moduli << "," << mode_str << ","
                         << ozaki_time * 1000.0 << "," << ozaki_tflops << ","
                         << worksize_this / (1024.0*1024.0) << ","
                         << std::scientific << frob_rel_err << "," << max_rel_err << ","
                         << avg_rel_err << "," << max_abs_diff << std::fixed << std::endl;
        }
    }
    
    HIP_CHECK(hipFree(d_work));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C_rocblas));
    HIP_CHECK(hipFree(d_C_ozaki));
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <matrix_folder>" << std::endl;
    std::cout << "\nBenchmarks C = A * B using native FP64 vs emulated (Ozaki-II)" << std::endl;
    std::cout << "\nExpected files:" << std::endl;
    std::cout << "  Condition number: A_NxK_cond1eX.txt and B_KxM_cond1eX.txt" << std::endl;
    std::cout << "  Aspect ratio:     A_NxK_ratio1eX.txt and B_KxM_ratio1eX.txt" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
        print_usage(argv[0]);
        return (argc < 2) ? 1 : 0;
    }
    
    // Check for GEMMUL8_PROFILE environment variable
    const char* prof = getenv("GEMMUL8_PROFILE");
    if (prof && std::string(prof) == "1") {
        oz2::g_profiling_enabled = true;
        std::cerr << "[GEMMUL8] Profiling enabled" << std::endl;
    }
    
    std::string matrix_folder = argv[1];
    if (!matrix_folder.empty() && matrix_folder.back() == '/') matrix_folder.pop_back();
    
    std::cout << "============================================" << std::endl;
    std::cout << "A/B GEMM Benchmark: Native FP64 vs Emulated" << std::endl;
    std::cout << "============================================" << std::endl;
    
    std::vector<MatrixPair> pairs = find_matrix_pairs(matrix_folder);
    if (pairs.empty()) {
        std::cerr << "No matrix pairs found in " << matrix_folder << std::endl;
        return 1;
    }
    
    std::cout << "Found " << pairs.size() << " matrix pair(s)" << std::endl;
    
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << std::endl;
    
    hipblasHandle_t handle;
    HIPBLAS_CHECK(hipblasCreate(&handle));
    
    std::string results_filename = matrix_folder + "/ab_gemm_benchmark_" + get_device_name() + "_" + get_timestamp() + ".csv";
    std::ofstream results_file(results_filename);
    results_file << std::fixed << std::setprecision(6);
    results_file << "log10_cond,method,num_moduli,fastmode,time_ms,tflops,workspace_mb,frob_rel_err,max_rel_err,avg_rel_err,max_abs_diff" << std::endl;
    
    for (const auto& pair : pairs) {
        benchmark_matrix_pair(handle, pair, results_file);
    }
    
    HIPBLAS_CHECK(hipblasDestroy(handle));
    results_file.close();
    
    std::cout << "\nResults saved to: " << results_filename << std::endl;
    
    return 0;
}
