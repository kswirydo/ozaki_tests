/**
 * NVIDIA CUDA matrix generator with prescribed condition number (cuBLAS + cuSOLVER + cuRAND).
 *
 * Same outputs as ../matrix_generator.cpp (ROCm).
 *
 * Compile:
 *   nvcc -O3 -std=c++17 matrix_generator.cu -o matrix_generator \
 *        -lcublas -lcusolver -lcurand -lcudart
 */

#include "cuda_helpers.cuh"

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

__global__ void set_diagonal_kernel(double* S, int n, int ld, double* singular_values) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
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

__global__ void compute_singular_values_kernel(double* sv, int n, double log10KA) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double exponent = (n > 1) ? ((double)idx / (double)(n - 1) * log10KA) : 0.0;
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

void write_matrix_to_file(const double* h_A, int m, int n, const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        return;
    }
    file.precision(17);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (j > 0) {
                file << ",";
            }
            file << h_A[j * m + i];
        }
        file << "\n";
    }
    file.close();
    std::cout << "  Written to " << filepath << std::endl;
}

void generate_matrix(cublasHandle_t cublas, cusolverDnHandle_t cusolver, curandGenerator_t gen, int m, int n,
                     int log10KA, const std::string& output_folder) {
    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << "Generating matrix with condition number 10^" << log10KA << std::endl;
    size_t size_U = (size_t)m * n;
    size_t size_V = (size_t)n * n;
    size_t size_S = (size_t)n * n;
    size_t size_A = (size_t)m * n;
    double *d_U, *d_V, *d_S, *d_A, *d_temp;
    double *d_tau_U, *d_tau_V;
    double* d_singular_values;
    CUDA_CHECK(cudaMalloc(&d_U, size_U * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_V, size_V * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_S, size_S * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_temp, size_A * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_tau_U, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_tau_V, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_singular_values, n * sizeof(double)));
    std::cout << "  Generating random matrices..." << std::endl;
    CURAND_CHECK(curandGenerateNormalDouble(gen, d_U, size_U, 0.0, 1.0));
    CURAND_CHECK(curandGenerateNormalDouble(gen, d_V, size_V, 0.0, 1.0));
    std::cout << "  Computing QR decomposition of U (" << m << "x" << n << ")..." << std::endl;
    cusolver_thin_orthogonal_q(cusolver, m, n, d_U, m, d_tau_U);
    std::cout << "  Computing QR decomposition of V (" << n << "x" << n << ")..." << std::endl;
    cusolver_thin_orthogonal_q(cusolver, n, n, d_V, n, d_tau_V);
    int block_size = 256;
    int num_blocks_zero = (int)((size_S + block_size - 1) / block_size);
    zero_matrix_kernel<<<num_blocks_zero, block_size>>>(d_S, size_S);
    std::cout << "  Setting up singular values..." << std::endl;
    int num_blocks_sv = (n + block_size - 1) / block_size;
    compute_singular_values_kernel<<<num_blocks_sv, block_size>>>(d_singular_values, n, (double)log10KA);
    set_diagonal_kernel<<<num_blocks_sv, block_size>>>(d_S, n, n, d_singular_values);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "  Computing A = U * S * V'..." << std::endl;
    double alpha = 1.0;
    double beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, &alpha, d_U, m, d_S, n, &beta, d_temp,
                             m));
    CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, m, n, n, &alpha, d_temp, m, d_V, n, &beta, d_A, m));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "  Copying result to host..." << std::endl;
    std::vector<double> h_A(size_A);
    CUDA_CHECK(cudaMemcpy(h_A.data(), d_A, size_A * sizeof(double), cudaMemcpyDeviceToHost));
    std::string filename = output_folder + "/M_cond_1e" + std::to_string(log10KA) + ".txt";
    std::cout << "  Writing matrix to file..." << std::endl;
    write_matrix_to_file(h_A.data(), m, n, filename);
    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_tau_U));
    CUDA_CHECK(cudaFree(d_tau_V));
    CUDA_CHECK(cudaFree(d_singular_values));
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "  Completed in " << duration.count() << " seconds" << std::endl << std::endl;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [m] [n] [output_folder]" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc >= 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        print_usage(argv[0]);
        return 0;
    }
    int m = 54272;
    int n = 54272;
    std::string output_folder;
    if (argc >= 3) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
    }
    if (argc >= 4) {
        output_folder = argv[3];
    } else {
        output_folder = "matrices_" + std::to_string(m) + "x" + std::to_string(n) + "_" + generate_timestamp();
    }
    std::cout << "==================================================" << std::endl;
    std::cout << "NVIDIA CUDA Matrix Generator (prescribed condition)" << std::endl;
    std::cout << "Matrix size: " << m << " x " << n << std::endl;
    std::cout << "Output folder: " << output_folder << std::endl;
    std::cout << "==================================================" << std::endl;
    if (!create_directory(output_folder)) {
        std::cerr << "Failed to create output directory. Exiting." << std::endl;
        return 1;
    }
    std::cout << "Output directory created/verified: " << output_folder << std::endl;
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    std::cout << "GPU Memory: " << free_mem / (1024 * 1024 * 1024.0) << " GB free / "
              << total_mem / (1024 * 1024 * 1024.0) << " GB total" << std::endl;
    size_t required_mem =
        (size_t)m * n * sizeof(double) * 4 + (size_t)n * n * sizeof(double) * 2;
    std::cout << "Estimated memory requirement: " << required_mem / (1024 * 1024 * 1024.0) << " GB" << std::endl;
    if (required_mem > free_mem) {
        std::cerr << "Warning: May not have enough GPU memory!" << std::endl;
    }
    std::cout << std::endl;
    cublasHandle_t cublas;
    cusolverDnHandle_t cusolver;
    CUBLAS_CHECK(cublasCreate(&cublas));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 12345ULL));
    for (int i = 1; i <= 16; i++) {
        int log10KA = 1 + i;
        generate_matrix(cublas, cusolver, gen, m, n, log10KA, output_folder);
    }
    CURAND_CHECK(curandDestroyGenerator(gen));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver));
    CUBLAS_CHECK(cublasDestroy(cublas));
    std::cout << "==================================================" << std::endl;
    std::cout << "All matrices generated successfully!" << std::endl;
    std::cout << "Output folder: " << output_folder << std::endl;
    std::cout << "==================================================" << std::endl;
    return 0;
}
