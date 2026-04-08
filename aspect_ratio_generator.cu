/**
 * Aspect Ratio Matrix Generator
 * 
 * Generates matrices with controlled dynamic range (aspect ratio).
 * Creates 64 matrices total: 32 pairs of A(MxK) and B(KxN) with aspect ratios
 * from 10^0 to 10^31 (powers of 10).
 * 
 * Aspect ratio = max(|values|) / min(|values|) within each row and column.
 * 
 * Usage:
 *   ./aspect_ratio_generator <M> <N> <K> <output_folder>
 */

#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <sys/stat.h>

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

#define HIPRAND_CHECK(call) do { \
    hiprandStatus_t status = call; \
    if (status != HIPRAND_STATUS_SUCCESS) { \
        std::cerr << "hipRAND error: " << status << std::endl; \
        exit(1); \
    } \
} while(0)

/**
 * Kernel to apply aspect ratio transformation.
 * Transforms uniform [0,1] values to have specified dynamic range.
 * 
 * For aspect_ratio R:
 *   - Maps values from [0,1] to [1/sqrt(R), sqrt(R)]
 *   - Result: max/min ratio = R within rows/columns
 */
__global__ void apply_aspect_ratio_kernel(double* data, const double* uniform, 
                                           const double* signs, size_t size,
                                           double log10_ratio) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // uniform[idx] is in [0, 1]
        // Transform to achieve desired ratio
        // v_new = 10^(log10_ratio * (uniform - 0.5))
        // This maps [0,1] -> [10^(-log10_ratio/2), 10^(log10_ratio/2)]
        // Ratio = 10^log10_ratio
        
        double exponent = log10_ratio * (uniform[idx] - 0.5);
        double magnitude = pow(10.0, exponent);
        
        // Apply random sign (signs should be -1 or +1)
        double sign = (signs[idx] > 0.5) ? 1.0 : -1.0;
        
        data[idx] = sign * magnitude;
    }
}

/**
 * Kernel to ensure each row has at least one min and one max value.
 * Places extreme values at specific positions.
 */
__global__ void ensure_row_extremes_kernel(double* data, size_t rows, size_t cols,
                                            double log10_ratio) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        double max_val = pow(10.0, log10_ratio / 2.0);
        double min_val = pow(10.0, -log10_ratio / 2.0);
        
        // Place max at position 0, min at position 1 (or cols-1 if only 1 col)
        size_t max_col = 0;
        size_t min_col = (cols > 1) ? 1 : 0;
        
        // Preserve sign but set magnitude
        double sign_max = (data[row * cols + max_col] >= 0) ? 1.0 : -1.0;
        double sign_min = (data[row * cols + min_col] >= 0) ? 1.0 : -1.0;
        
        data[row * cols + max_col] = sign_max * max_val;
        data[row * cols + min_col] = sign_min * min_val;
    }
}

/**
 * Kernel to ensure each column has at least one min and one max value.
 */
__global__ void ensure_col_extremes_kernel(double* data, size_t rows, size_t cols,
                                            double log10_ratio) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        double max_val = pow(10.0, log10_ratio / 2.0);
        double min_val = pow(10.0, -log10_ratio / 2.0);
        
        // Place max at row 2, min at row 3 (avoid overwriting row extremes)
        size_t max_row = (rows > 2) ? 2 : 0;
        size_t min_row = (rows > 3) ? 3 : ((rows > 1) ? 1 : 0);
        
        // Only set if not already set by row extremes
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

void generate_aspect_ratio_matrix(hiprandGenerator_t gen, size_t rows, size_t cols,
                                   int log10_ratio, double* d_matrix) {
    size_t size = rows * cols;
    
    // Allocate temporary buffers
    double *d_uniform, *d_signs;
    HIP_CHECK(hipMalloc(&d_uniform, size * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_signs, size * sizeof(double)));
    
    // Generate uniform random values in [0, 1]
    HIPRAND_CHECK(hiprandGenerateUniformDouble(gen, d_uniform, size));
    HIPRAND_CHECK(hiprandGenerateUniformDouble(gen, d_signs, size));
    
    // Apply aspect ratio transformation
    int block_size = 256;
    size_t num_blocks = (size + block_size - 1) / block_size;
    
    hipLaunchKernelGGL(apply_aspect_ratio_kernel, dim3(num_blocks), dim3(block_size),
                       0, 0, d_matrix, d_uniform, d_signs, size, (double)log10_ratio);
    
    // Ensure row extremes
    num_blocks = (rows + block_size - 1) / block_size;
    hipLaunchKernelGGL(ensure_row_extremes_kernel, dim3(num_blocks), dim3(block_size),
                       0, 0, d_matrix, rows, cols, (double)log10_ratio);
    
    // Ensure column extremes
    num_blocks = (cols + block_size - 1) / block_size;
    hipLaunchKernelGGL(ensure_col_extremes_kernel, dim3(num_blocks), dim3(block_size),
                       0, 0, d_matrix, rows, cols, (double)log10_ratio);
    
    HIP_CHECK(hipDeviceSynchronize());
    
    HIP_CHECK(hipFree(d_uniform));
    HIP_CHECK(hipFree(d_signs));
}

void save_matrix(const std::string& filename, const double* h_matrix, 
                 size_t rows, size_t cols) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(1);
    }
    
    file << std::scientific << std::setprecision(15);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            if (j > 0) file << ",";
            file << h_matrix[i * cols + j];
        }
        file << "\n";
    }
    file.close();
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <M> <N> <K> <output_folder>" << std::endl;
        std::cerr << "  Generates A(MxK) and B(KxN) matrix pairs with aspect ratios 10^0 to 10^31" << std::endl;
        std::cerr << "  Total: 64 matrices (32 pairs)" << std::endl;
        return 1;
    }
    
    size_t M = std::stoull(argv[1]);
    size_t N = std::stoull(argv[2]);
    size_t K = std::stoull(argv[3]);
    std::string folder = argv[4];
    
    // Create output folder
    mkdir(folder.c_str(), 0755);
    
    std::cout << "Aspect Ratio Matrix Generator" << std::endl;
    std::cout << "==============================" << std::endl;
    std::cout << "A: " << M << " x " << K << std::endl;
    std::cout << "B: " << K << " x " << N << std::endl;
    std::cout << "C = A*B: " << M << " x " << N << std::endl;
    std::cout << "Output folder: " << folder << std::endl;
    std::cout << "Generating 32 pairs (aspect ratios 10^0 to 10^31)..." << std::endl;
    std::cout << std::endl;
    
    // Initialize random generator
    hiprandGenerator_t gen;
    HIPRAND_CHECK(hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT));
    HIPRAND_CHECK(hiprandSetPseudoRandomGeneratorSeed(gen, 12345ULL));
    
    // Allocate device memory
    size_t size_A = M * K;
    size_t size_B = K * N;
    
    double *d_A, *d_B;
    HIP_CHECK(hipMalloc(&d_A, size_A * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_B, size_B * sizeof(double)));
    
    // Host buffers
    std::vector<double> h_A(size_A);
    std::vector<double> h_B(size_B);
    
    // Generate matrices for each aspect ratio
    for (int log10_ratio = 0; log10_ratio <= 31; log10_ratio++) {
        std::cout << "Generating aspect ratio 10^" << log10_ratio << "..." << std::flush;
        
        // Generate A
        generate_aspect_ratio_matrix(gen, M, K, log10_ratio, d_A);
        HIP_CHECK(hipMemcpy(h_A.data(), d_A, size_A * sizeof(double), hipMemcpyDeviceToHost));
        
        // Generate B
        generate_aspect_ratio_matrix(gen, K, N, log10_ratio, d_B);
        HIP_CHECK(hipMemcpy(h_B.data(), d_B, size_B * sizeof(double), hipMemcpyDeviceToHost));
        
        // Save matrices
        std::string filename_A = folder + "/A_" + std::to_string(M) + "x" + std::to_string(K) 
                                + "_ratio1e" + std::to_string(log10_ratio) + ".txt";
        std::string filename_B = folder + "/B_" + std::to_string(K) + "x" + std::to_string(N) 
                                + "_ratio1e" + std::to_string(log10_ratio) + ".txt";
        
        save_matrix(filename_A, h_A.data(), M, K);
        save_matrix(filename_B, h_B.data(), K, N);
        
        std::cout << " done" << std::endl;
    }
    
    // Cleanup
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIPRAND_CHECK(hiprandDestroyGenerator(gen));
    
    std::cout << std::endl;
    std::cout << "Generated 64 matrices (32 pairs) in " << folder << std::endl;
    
    return 0;
}
