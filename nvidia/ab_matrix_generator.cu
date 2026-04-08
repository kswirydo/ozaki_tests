/**
 * NVIDIA CUDA A/B matrix generator with prescribed condition numbers.
 *
 * Same outputs as ../ab_matrix_generator.cpp (ROCm).
 *
 * Compile:
 *   nvcc -O3 -std=c++17 ab_matrix_generator.cu -o ab_matrix_generator \
 *        -lcublas -lcusolver -lcurand -lcudart
 */

#include "cuda_helpers.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

__global__ void set_diagonal_kernel(double* S, size_t n, size_t ld, double* singular_values) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        S[idx * ld + idx] = singular_values[idx];
    }
}

__global__ void zero_matrix_kernel(double* A, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] = 0.0;
    }
}

__global__ void compute_singular_values_kernel(double* sv, size_t n, double log10_cond) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double exponent = (n > 1) ? ((double)idx / (double)(n - 1) * log10_cond) : 0.0;
        sv[idx] = pow(10.0, exponent);
    }
}

std::string generate_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");
    return ss.str();
}

bool create_directory(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) == 0) {
        if (S_ISDIR(st.st_mode)) {
            return true;
        }
        std::cerr << "Error: " << path << " exists but is not a directory" << std::endl;
        return false;
    }
    if (mkdir(path.c_str(), 0755) == 0) {
        return true;
    }
    std::cerr << "Error: Failed to create directory " << path << std::endl;
    return false;
}

void write_matrix_to_file(const double* h_A, size_t rows, size_t cols, const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        return;
    }
    file.precision(17);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            if (j > 0) {
                file << ",";
            }
            file << h_A[j * rows + i];
        }
        file << "\n";
    }
    file.close();
    std::cout << "    Written to " << filepath << std::endl;
}

void generate_conditioned_matrix(cublasHandle_t cublas_h, cusolverDnHandle_t cusolver_h, curandGenerator_t gen,
                                 size_t rows, size_t cols, int log10_cond, double* d_result,
                                 std::vector<double>& h_result) {
    size_t min_dim = std::min(rows, cols);
    size_t size_U = rows * min_dim;
    size_t size_V = cols * min_dim;
    size_t size_S = min_dim * min_dim;
    size_t size_result = rows * cols;
    double *d_U, *d_V, *d_S, *d_temp;
    double *d_tau_U, *d_tau_V;
    double* d_singular_values;
    CUDA_CHECK(cudaMalloc(&d_U, size_U * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_V, size_V * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_S, size_S * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_temp, rows * min_dim * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_tau_U, min_dim * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_tau_V, min_dim * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_singular_values, min_dim * sizeof(double)));
    CURAND_CHECK(curandGenerateNormalDouble(gen, d_U, size_U, 0.0, 1.0));
    CURAND_CHECK(curandGenerateNormalDouble(gen, d_V, size_V, 0.0, 1.0));
    cusolver_thin_orthogonal_q(cusolver_h, (int)rows, (int)min_dim, d_U, (int)rows, d_tau_U);
    cusolver_thin_orthogonal_q(cusolver_h, (int)cols, (int)min_dim, d_V, (int)cols, d_tau_V);
    int block_size = 256;
    size_t num_blocks_zero = (size_S + block_size - 1) / block_size;
    zero_matrix_kernel<<<num_blocks_zero, block_size>>>(d_S, size_S);
    size_t num_blocks_sv = (min_dim + block_size - 1) / block_size;
    compute_singular_values_kernel<<<num_blocks_sv, block_size>>>(d_singular_values, min_dim, (double)log10_cond);
    set_diagonal_kernel<<<num_blocks_sv, block_size>>>(d_S, min_dim, min_dim, d_singular_values);
    CUDA_CHECK(cudaDeviceSynchronize());
    double alpha = 1.0;
    double beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(cublas_h, CUBLAS_OP_N, CUBLAS_OP_N, (int)rows, (int)min_dim, (int)min_dim, &alpha, d_U,
                             (int)rows, d_S, (int)min_dim, &beta, d_temp, (int)rows));
    CUBLAS_CHECK(cublasDgemm(cublas_h, CUBLAS_OP_N, CUBLAS_OP_T, (int)rows, (int)cols, (int)min_dim, &alpha, d_temp,
                             (int)rows, d_V, (int)cols, &beta, d_result, (int)rows));
    CUDA_CHECK(cudaDeviceSynchronize());
    h_result.resize(size_result);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, size_result * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_tau_U));
    CUDA_CHECK(cudaFree(d_tau_V));
    CUDA_CHECK(cudaFree(d_singular_values));
}

void generate_matrix_pair(cublasHandle_t cublas_h, cusolverDnHandle_t cusolver_h, curandGenerator_t gen, size_t N,
                          size_t M, size_t K, int log10_cond, const std::string& output_folder) {
    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << "Generating matrices with condition number 10^" << log10_cond << std::endl;
    double *d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, N * K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, K * M * sizeof(double)));
    std::vector<double> h_A, h_B;
    std::cout << "  Generating A (" << N << " x " << K << ")..." << std::endl;
    generate_conditioned_matrix(cublas_h, cusolver_h, gen, N, K, log10_cond, d_A, h_A);
    std::cout << "  Generating B (" << K << " x " << M << ")..." << std::endl;
    generate_conditioned_matrix(cublas_h, cusolver_h, gen, K, M, log10_cond, d_B, h_B);
    std::string filename_A = output_folder + "/A_" + std::to_string(N) + "x" + std::to_string(K) + "_cond1e" +
                             std::to_string(log10_cond) + ".txt";
    std::string filename_B = output_folder + "/B_" + std::to_string(K) + "x" + std::to_string(M) + "_cond1e" +
                             std::to_string(log10_cond) + ".txt";
    std::cout << "  Writing matrices to files..." << std::endl;
    write_matrix_to_file(h_A.data(), N, K, filename_A);
    write_matrix_to_file(h_B.data(), K, M, filename_B);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "  Completed in " << duration.count() << " seconds" << std::endl << std::endl;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " N M K [output_folder]" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 4 || (argc >= 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help"))) {
        print_usage(argv[0]);
        return (argc < 4) ? 1 : 0;
    }
    size_t N = std::stoull(argv[1]);
    size_t M = std::stoull(argv[2]);
    size_t K = std::stoull(argv[3]);
    std::string output_folder;
    if (argc >= 5) {
        output_folder = argv[4];
    } else {
        output_folder = "ab_matrices_" + std::to_string(N) + "x" + std::to_string(M) + "x" + std::to_string(K) +
                        "_" + generate_timestamp();
    }
    std::cout << "==================================================" << std::endl;
    std::cout << "NVIDIA CUDA A/B Matrix Generator (condition numbers)" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Matrix A: " << N << " x " << K << std::endl;
    std::cout << "Matrix B: " << K << " x " << M << std::endl;
    std::cout << "Output folder: " << output_folder << std::endl;
    std::cout << "==================================================" << std::endl << std::endl;
    if (!create_directory(output_folder)) {
        std::cerr << "Failed to create output directory. Exiting." << std::endl;
        return 1;
    }
    std::cout << "Output directory created/verified: " << output_folder << std::endl;
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    std::cout << "GPU Memory: " << free_mem / (1024.0 * 1024.0 * 1024.0) << " GB free / "
              << total_mem / (1024.0 * 1024.0 * 1024.0) << " GB total" << std::endl;
    size_t max_dim = std::max({N, M, K});
    size_t required_mem = max_dim * max_dim * sizeof(double) * 5;
    std::cout << "Estimated memory requirement: " << required_mem / (1024.0 * 1024.0 * 1024.0) << " GB"
              << std::endl;
    if (required_mem > free_mem) {
        std::cerr << "Warning: May not have enough GPU memory!" << std::endl;
    }
    std::cout << std::endl;
    cublasHandle_t cublas_h;
    cusolverDnHandle_t cusolver_h;
    CUBLAS_CHECK(cublasCreate(&cublas_h));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_h));
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 12345ULL));
    for (int log10_cond = 2; log10_cond <= 17; log10_cond++) {
        generate_matrix_pair(cublas_h, cusolver_h, gen, N, M, K, log10_cond, output_folder);
    }
    CURAND_CHECK(curandDestroyGenerator(gen));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver_h));
    CUBLAS_CHECK(cublasDestroy(cublas_h));
    std::cout << "==================================================" << std::endl;
    std::cout << "All matrices generated successfully!" << std::endl;
    std::cout << "Output folder: " << output_folder << std::endl;
    std::cout << "==================================================" << std::endl;
    return 0;
}
