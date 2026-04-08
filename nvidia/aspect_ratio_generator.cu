/**
 * NVIDIA CUDA aspect-ratio matrix generator (same logic as ../aspect_ratio_generator.cu).
 *
 * Compile:
 *   nvcc -O3 -std=c++17 aspect_ratio_generator.cu -o aspect_ratio_generator -lcurand -lcudart
 */

#include "cuda_helpers.cuh"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/stat.h>
#include <vector>

__global__ void apply_aspect_ratio_kernel(double* data, const double* uniform, const double* signs, size_t size,
                                          double log10_ratio) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double exponent = log10_ratio * (uniform[idx] - 0.5);
        double magnitude = pow(10.0, exponent);
        double sign = (signs[idx] > 0.5) ? 1.0 : -1.0;
        data[idx] = sign * magnitude;
    }
}

__global__ void ensure_row_extremes_kernel(double* data, size_t rows, size_t cols, double log10_ratio) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        double max_val = pow(10.0, log10_ratio / 2.0);
        double min_val = pow(10.0, -log10_ratio / 2.0);
        size_t max_col = 0;
        size_t min_col = (cols > 1) ? 1 : 0;
        double sign_max = (data[row * cols + max_col] >= 0) ? 1.0 : -1.0;
        double sign_min = (data[row * cols + min_col] >= 0) ? 1.0 : -1.0;
        data[row * cols + max_col] = sign_max * max_val;
        data[row * cols + min_col] = sign_min * min_val;
    }
}

__global__ void ensure_col_extremes_kernel(double* data, size_t rows, size_t cols, double log10_ratio) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        double max_val = pow(10.0, log10_ratio / 2.0);
        double min_val = pow(10.0, -log10_ratio / 2.0);
        size_t max_row = (rows > 2) ? 2 : 0;
        size_t min_row = (rows > 3) ? 3 : ((rows > 1) ? 1 : 0);
        if (max_row >= 2 || rows <= 2) {
            double sign_max = (data[max_row * cols + col] >= 0) ? 1.0 : -1.0;
            data[max_row * cols + col] = sign_max * max_val;
        }
        if (min_row >= 2 || rows <= 2) {
            double sign_min = (data[min_row * cols + col] >= 0) ? 1.0 : -1.0;
            data[min_row * cols + col] = sign_min * min_val;
        }
    }
}

void generate_aspect_ratio_matrix(curandGenerator_t gen, size_t rows, size_t cols, int log10_ratio,
                                  double* d_matrix) {
    size_t size = rows * cols;
    double *d_uniform, *d_signs;
    CUDA_CHECK(cudaMalloc(&d_uniform, size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_signs, size * sizeof(double)));
    CURAND_CHECK(curandGenerateUniformDouble(gen, d_uniform, size));
    CURAND_CHECK(curandGenerateUniformDouble(gen, d_signs, size));
    int block_size = 256;
    size_t num_blocks = (size + block_size - 1) / block_size;
    apply_aspect_ratio_kernel<<<num_blocks, block_size>>>(d_matrix, d_uniform, d_signs, size, (double)log10_ratio);
    num_blocks = (rows + block_size - 1) / block_size;
    ensure_row_extremes_kernel<<<num_blocks, block_size>>>(d_matrix, rows, cols, (double)log10_ratio);
    num_blocks = (cols + block_size - 1) / block_size;
    ensure_col_extremes_kernel<<<num_blocks, block_size>>>(d_matrix, rows, cols, (double)log10_ratio);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_uniform));
    CUDA_CHECK(cudaFree(d_signs));
}

void save_matrix(const std::string& filename, const double* h_matrix, size_t rows, size_t cols) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(1);
    }
    file << std::scientific << std::setprecision(15);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            if (j > 0) {
                file << ",";
            }
            file << h_matrix[i * cols + j];
        }
        file << "\n";
    }
    file.close();
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <M> <N> <K> <output_folder>" << std::endl;
        return 1;
    }
    size_t M = std::stoull(argv[1]);
    size_t N = std::stoull(argv[2]);
    size_t K = std::stoull(argv[3]);
    std::string folder = argv[4];
    mkdir(folder.c_str(), 0755);
    std::cout << "Aspect Ratio Matrix Generator (CUDA)" << std::endl;
    std::cout << "==============================" << std::endl;
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 12345ULL));
    size_t size_A = M * K;
    size_t size_B = K * N;
    double *d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(double)));
    std::vector<double> h_A(size_A);
    std::vector<double> h_B(size_B);
    for (int log10_ratio = 0; log10_ratio <= 31; log10_ratio++) {
        std::cout << "Generating aspect ratio 10^" << log10_ratio << "..." << std::flush;
        generate_aspect_ratio_matrix(gen, M, K, log10_ratio, d_A);
        CUDA_CHECK(cudaMemcpy(h_A.data(), d_A, size_A * sizeof(double), cudaMemcpyDeviceToHost));
        generate_aspect_ratio_matrix(gen, K, N, log10_ratio, d_B);
        CUDA_CHECK(cudaMemcpy(h_B.data(), d_B, size_B * sizeof(double), cudaMemcpyDeviceToHost));
        std::string filename_A = folder + "/A_" + std::to_string(M) + "x" + std::to_string(K) + "_ratio1e" +
                                 std::to_string(log10_ratio) + ".txt";
        std::string filename_B = folder + "/B_" + std::to_string(K) + "x" + std::to_string(N) + "_ratio1e" +
                                 std::to_string(log10_ratio) + ".txt";
        save_matrix(filename_A, h_A.data(), M, K);
        save_matrix(filename_B, h_B.data(), K, N);
        std::cout << " done" << std::endl;
    }
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CURAND_CHECK(curandDestroyGenerator(gen));
    std::cout << "Generated 64 matrices (32 pairs) in " << folder << std::endl;
    return 0;
}
