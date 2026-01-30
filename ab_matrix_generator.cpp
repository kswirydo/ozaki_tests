/**
 * ROCm A/B Matrix Generator with Prescribed Condition Numbers
 * 
 * Generates pairs of random matrices A (N x K) and B (K x M) with prescribed
 * condition numbers for GEMM benchmarking.
 * 
 * Matrices are generated as: A = U * S * V' where U, V are orthogonal and
 * S is diagonal with singular values from 1 to 10^log10_cond.
 * 
 * Compile with:
 *   hipcc -O3 ab_matrix_generator.cpp -o ab_matrix_generator \
 *         -lrocblas -lrocsolver -lhiprand -I/opt/rocm/include -L/opt/rocm/lib
 * 
 * Usage:
 *   ./ab_matrix_generator N M K [output_folder]
 *   
 *   N             - Number of rows of A and C
 *   M             - Number of columns of B and C
 *   K             - Inner dimension (columns of A, rows of B)
 *   output_folder - Folder to save matrices (optional, default: auto-generated)
 */

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <hiprand/hiprand.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>

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

#define ROCBLAS_CHECK(call)                                                     \
    do {                                                                        \
        rocblas_status status = call;                                           \
        if (status != rocblas_status_success) {                                 \
            std::cerr << "rocBLAS error at " << __FILE__ << ":" << __LINE__     \
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

// Kernel to set diagonal elements of S matrix (column-major)
__global__ void set_diagonal_kernel(double* S, size_t n, size_t ld, double* singular_values) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        S[idx * ld + idx] = singular_values[idx];
    }
}

// Kernel to zero out a matrix
__global__ void zero_matrix_kernel(double* A, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] = 0.0;
    }
}

// Kernel to compute singular values: 10^(linspace(0, log10_cond, n))
__global__ void compute_singular_values_kernel(double* sv, size_t n, double log10_cond) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double exponent = (n > 1) ? ((double)idx / (double)(n - 1) * log10_cond) : 0.0;
        sv[idx] = pow(10.0, exponent);
    }
}

// Generate timestamp string for folder name
std::string generate_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");
    return ss.str();
}

// Create directory (returns true on success or if already exists)
bool create_directory(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) == 0) {
        if (S_ISDIR(st.st_mode)) {
            return true;  // Already exists
        } else {
            std::cerr << "Error: " << path << " exists but is not a directory" << std::endl;
            return false;
        }
    }
    
    // Create directory with permissions rwxr-xr-x
    if (mkdir(path.c_str(), 0755) == 0) {
        return true;
    } else {
        std::cerr << "Error: Failed to create directory " << path << std::endl;
        return false;
    }
}

void write_matrix_to_file(const double* h_A, size_t rows, size_t cols, const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        return;
    }
    
    file.precision(17);  // Full double precision
    
    // Write in row-major format (converting from column-major storage)
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            if (j > 0) file << ",";
            file << h_A[j * rows + i];  // Column-major access
        }
        file << "\n";
    }
    
    file.close();
    std::cout << "    Written to " << filepath << std::endl;
}

/**
 * Generate a matrix with prescribed condition number using SVD construction:
 * M = U * S * V' where U, V are orthogonal and S has singular values from 1 to 10^log10_cond
 */
void generate_conditioned_matrix(rocblas_handle handle, hiprandGenerator_t gen,
                                  size_t rows, size_t cols, int log10_cond,
                                  double* d_result, std::vector<double>& h_result) {
    
    size_t min_dim = std::min(rows, cols);
    size_t size_U = rows * min_dim;
    size_t size_V = cols * min_dim;
    size_t size_S = min_dim * min_dim;
    size_t size_result = rows * cols;
    
    // Device memory allocation
    double *d_U, *d_V, *d_S, *d_temp;
    double *d_tau_U, *d_tau_V;
    double *d_singular_values;
    
    HIP_CHECK(hipMalloc(&d_U, size_U * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_V, size_V * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_S, size_S * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_temp, rows * min_dim * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_tau_U, min_dim * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_tau_V, min_dim * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_singular_values, min_dim * sizeof(double)));
    
    // Generate random matrices U and V using normal distribution
    HIPRAND_CHECK(hiprandGenerateNormalDouble(gen, d_U, size_U, 0.0, 1.0));
    HIPRAND_CHECK(hiprandGenerateNormalDouble(gen, d_V, size_V, 0.0, 1.0));
    
    // QR decomposition of U: U = Q_U * R_U, we keep Q_U (rows x min_dim)
    rocsolver_dgeqrf(handle, rows, min_dim, d_U, rows, d_tau_U);
    rocsolver_dorgqr(handle, rows, min_dim, min_dim, d_U, rows, d_tau_U);
    
    // QR decomposition of V: V = Q_V * R_V, we keep Q_V (cols x min_dim)
    rocsolver_dgeqrf(handle, cols, min_dim, d_V, cols, d_tau_V);
    rocsolver_dorgqr(handle, cols, min_dim, min_dim, d_V, cols, d_tau_V);
    
    // Initialize S to zero
    int block_size = 256;
    size_t num_blocks_zero = (size_S + block_size - 1) / block_size;
    hipLaunchKernelGGL(zero_matrix_kernel, dim3(num_blocks_zero), dim3(block_size), 
                       0, 0, d_S, size_S);
    
    // Compute singular values: 10^(linspace(0, log10_cond, min_dim))
    size_t num_blocks_sv = (min_dim + block_size - 1) / block_size;
    hipLaunchKernelGGL(compute_singular_values_kernel, dim3(num_blocks_sv), dim3(block_size),
                       0, 0, d_singular_values, min_dim, (double)log10_cond);
    
    // Set diagonal of S
    hipLaunchKernelGGL(set_diagonal_kernel, dim3(num_blocks_sv), dim3(block_size),
                       0, 0, d_S, min_dim, min_dim, d_singular_values);
    
    HIP_CHECK(hipDeviceSynchronize());
    
    // Compute result = U * S * V'
    // First: temp = U * S (rows x min_dim) * (min_dim x min_dim) = (rows x min_dim)
    // Then:  result = temp * V' (rows x min_dim) * (min_dim x cols) = (rows x cols)
    
    double alpha = 1.0;
    double beta = 0.0;
    
    // temp = U * S
    ROCBLAS_CHECK(rocblas_dgemm(handle,
                                rocblas_operation_none,    // U
                                rocblas_operation_none,    // S
                                rows, min_dim, min_dim,    // m, n, k
                                &alpha,
                                d_U, rows,                 // A, lda
                                d_S, min_dim,              // B, ldb
                                &beta,
                                d_temp, rows));            // C, ldc
    
    // result = temp * V' = temp * V^T
    ROCBLAS_CHECK(rocblas_dgemm(handle,
                                rocblas_operation_none,       // temp
                                rocblas_operation_transpose,  // V'
                                rows, cols, min_dim,          // m, n, k
                                &alpha,
                                d_temp, rows,                 // A, lda
                                d_V, cols,                    // B, ldb
                                &beta,
                                d_result, rows));             // C, ldc
    
    HIP_CHECK(hipDeviceSynchronize());
    
    // Copy result to host
    h_result.resize(size_result);
    HIP_CHECK(hipMemcpy(h_result.data(), d_result, size_result * sizeof(double), hipMemcpyDeviceToHost));
    
    // Free device memory
    HIP_CHECK(hipFree(d_U));
    HIP_CHECK(hipFree(d_V));
    HIP_CHECK(hipFree(d_S));
    HIP_CHECK(hipFree(d_temp));
    HIP_CHECK(hipFree(d_tau_U));
    HIP_CHECK(hipFree(d_tau_V));
    HIP_CHECK(hipFree(d_singular_values));
}

void generate_matrix_pair(rocblas_handle handle, hiprandGenerator_t gen,
                          size_t N, size_t M, size_t K, int log10_cond,
                          const std::string& output_folder) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "Generating matrices with condition number 10^" << log10_cond << std::endl;
    
    // Allocate device memory for results
    double *d_A, *d_B;
    HIP_CHECK(hipMalloc(&d_A, N * K * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_B, K * M * sizeof(double)));
    
    std::vector<double> h_A, h_B;
    
    // Generate matrix A (N x K) with prescribed condition number
    std::cout << "  Generating A (" << N << " x " << K << ")..." << std::endl;
    generate_conditioned_matrix(handle, gen, N, K, log10_cond, d_A, h_A);
    
    // Generate matrix B (K x M) with prescribed condition number
    std::cout << "  Generating B (" << K << " x " << M << ")..." << std::endl;
    generate_conditioned_matrix(handle, gen, K, M, log10_cond, d_B, h_B);
    
    // Build filenames: A_NxK_cond1eX.txt and B_KxM_cond1eX.txt
    std::string filename_A = output_folder + "/A_" + std::to_string(N) + "x" + std::to_string(K) + 
                             "_cond1e" + std::to_string(log10_cond) + ".txt";
    std::string filename_B = output_folder + "/B_" + std::to_string(K) + "x" + std::to_string(M) + 
                             "_cond1e" + std::to_string(log10_cond) + ".txt";
    
    // Write matrices to files
    std::cout << "  Writing matrices to files..." << std::endl;
    write_matrix_to_file(h_A.data(), N, K, filename_A);
    write_matrix_to_file(h_B.data(), K, M, filename_B);
    
    // Free device memory
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "  Completed in " << duration.count() << " seconds" << std::endl;
    std::cout << std::endl;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " N M K [output_folder]" << std::endl;
    std::cout << std::endl;
    std::cout << "Generates pairs of matrices A and B with prescribed condition numbers for GEMM: C = A * B" << std::endl;
    std::cout << "  A: N x K matrix" << std::endl;
    std::cout << "  B: K x M matrix" << std::endl;
    std::cout << "  C: N x M matrix (result)" << std::endl;
    std::cout << std::endl;
    std::cout << "Generates matrix pairs for condition numbers: 10^2, 10^3, ..., 10^17" << std::endl;
    std::cout << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  N             - Number of rows of A (and C)" << std::endl;
    std::cout << "  M             - Number of columns of B (and C)" << std::endl;
    std::cout << "  K             - Inner dimension (columns of A, rows of B)" << std::endl;
    std::cout << "  output_folder - Folder to save matrices (optional, default: ab_matrices_NxMxK_timestamp)" << std::endl;
    std::cout << std::endl;
    std::cout << "Output files (for each condition number X from 2 to 17):" << std::endl;
    std::cout << "  A_NxK_cond1eX.txt  - Matrix A in CSV format" << std::endl;
    std::cout << "  B_KxM_cond1eX.txt  - Matrix B in CSV format" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << " 1024 1024 1024           # Square 1024 matrices, auto folder" << std::endl;
    std::cout << "  " << program_name << " 4096 4096 512            # Rectangular with K=512" << std::endl;
    std::cout << "  " << program_name << " 2048 2048 2048 my_folder # Save to 'my_folder'" << std::endl;
}

int main(int argc, char* argv[]) {
    // Check for help flag or insufficient arguments
    if (argc < 4 || (argc >= 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help"))) {
        print_usage(argv[0]);
        return (argc < 4) ? 1 : 0;
    }
    
    // Parse dimensions
    size_t N = std::stoull(argv[1]);
    size_t M = std::stoull(argv[2]);
    size_t K = std::stoull(argv[3]);
    
    // Parse optional output folder
    std::string output_folder;
    if (argc >= 5) {
        output_folder = argv[4];
    } else {
        // Generate folder name based on dimensions and timestamp
        output_folder = "ab_matrices_" + std::to_string(N) + "x" + std::to_string(M) + "x" + 
                        std::to_string(K) + "_" + generate_timestamp();
    }
    
    std::cout << "==================================================" << std::endl;
    std::cout << "ROCm A/B Matrix Generator with Condition Numbers" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Matrix A: " << N << " x " << K << std::endl;
    std::cout << "Matrix B: " << K << " x " << M << std::endl;
    std::cout << "Result C: " << N << " x " << M << " (for reference)" << std::endl;
    std::cout << "Condition numbers: 10^2 to 10^17" << std::endl;
    std::cout << "Output folder: " << output_folder << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << std::endl;
    
    // Create output directory
    if (!create_directory(output_folder)) {
        std::cerr << "Failed to create output directory. Exiting." << std::endl;
        return 1;
    }
    std::cout << "Output directory created/verified: " << output_folder << std::endl;
    
    // Check available GPU memory
    size_t free_mem, total_mem;
    HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));
    std::cout << "GPU Memory: " << free_mem / (1024.0*1024.0*1024.0) << " GB free / "
              << total_mem / (1024.0*1024.0*1024.0) << " GB total" << std::endl;
    
    // Estimate memory requirements (for one matrix generation at a time)
    size_t max_dim = std::max({N, M, K});
    size_t required_mem = max_dim * max_dim * sizeof(double) * 5;  // U, V, S, temp, result
    std::cout << "Estimated memory requirement: " << required_mem / (1024.0*1024.0*1024.0) 
              << " GB" << std::endl;
    
    if (required_mem > free_mem) {
        std::cerr << "Warning: May not have enough GPU memory!" << std::endl;
    }
    std::cout << std::endl;
    
    // Initialize rocBLAS
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
    
    // Initialize hipRAND
    hiprandGenerator_t gen;
    HIPRAND_CHECK(hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT));
    HIPRAND_CHECK(hiprandSetPseudoRandomGeneratorSeed(gen, 12345ULL));
    
    // Generate matrix pairs for condition numbers 10^2 to 10^17
    for (int log10_cond = 2; log10_cond <= 17; log10_cond++) {
        generate_matrix_pair(handle, gen, N, M, K, log10_cond, output_folder);
    }
    
    // Cleanup
    HIPRAND_CHECK(hiprandDestroyGenerator(gen));
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    
    std::cout << "==================================================" << std::endl;
    std::cout << "All matrices generated successfully!" << std::endl;
    std::cout << "Output folder: " << output_folder << std::endl;
    std::cout << "Generated " << 16 << " pairs of matrices (condition 10^2 to 10^17)" << std::endl;
    std::cout << "==================================================" << std::endl;
    
    return 0;
}
