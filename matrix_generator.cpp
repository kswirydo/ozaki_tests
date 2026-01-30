/**
 * ROCm Matrix Generator with Prescribed Condition Number
 * 
 * Generates dense matrices A = U * S * V' where:
 * - U, V are orthogonal (from QR decomposition of random matrices)
 * - S is diagonal with singular values from 1 to 10^log10KA
 * 
 * Compile with:
 *   hipcc -O3 matrix_generator.cpp -o matrix_generator \
 *         -lrocblas -lrocsolver -lhiprand -I/opt/rocm/include -L/opt/rocm/lib
 * 
 * Usage:
 *   ./matrix_generator [m] [n] [output_folder]
 *   
 *   m, n          - Matrix dimensions (default: 54272 x 54272)
 *   output_folder - Folder to save matrices (default: auto-generated from timestamp)
 */

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <hiprand/hiprand.h>

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
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
__global__ void set_diagonal_kernel(double* S, int n, int ld, double* singular_values) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
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

// Kernel to compute singular values: 10^(linspace(0, log10KA, n))
__global__ void compute_singular_values_kernel(double* sv, int n, double log10KA) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double exponent = (double)idx / (double)(n - 1) * log10KA;
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

// Extract Q from the result of geqrf (applies orgqr)
void extract_Q_from_QR(rocblas_handle handle, 
                       double* d_A, int m, int n, int lda,
                       double* d_tau, double* d_work, int lwork) {
    // Apply orgqr to get Q matrix from the QR factorization
    rocsolver_dorgqr(handle, m, n, n, d_A, lda, d_tau);
}

void write_matrix_to_file(const double* h_A, int m, int n, const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        return;
    }
    
    file.precision(17);  // Full double precision
    
    // Write in row-major format (converting from column-major storage)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (j > 0) file << ",";
            file << h_A[j * m + i];  // Column-major access
        }
        file << "\n";
    }
    
    file.close();
    std::cout << "  Written to " << filepath << std::endl;
}

void generate_matrix(rocblas_handle handle, hiprandGenerator_t gen,
                     int m, int n, int log10KA, const std::string& output_folder) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "Generating matrix with condition number 10^" << log10KA << std::endl;
    
    size_t size_U = (size_t)m * n;
    size_t size_V = (size_t)n * n;
    size_t size_S = (size_t)n * n;
    size_t size_A = (size_t)m * n;
    
    // Device memory allocation
    double *d_U, *d_V, *d_S, *d_A, *d_temp;
    double *d_tau_U, *d_tau_V;
    double *d_singular_values;
    
    HIP_CHECK(hipMalloc(&d_U, size_U * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_V, size_V * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_S, size_S * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_A, size_A * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_temp, size_A * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_tau_U, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_tau_V, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_singular_values, n * sizeof(double)));
    
    // Generate random matrices U and V using normal distribution
    std::cout << "  Generating random matrices..." << std::endl;
    HIPRAND_CHECK(hiprandGenerateNormalDouble(gen, d_U, size_U, 0.0, 1.0));
    HIPRAND_CHECK(hiprandGenerateNormalDouble(gen, d_V, size_V, 0.0, 1.0));
    
    // QR decomposition of U: U = Q_U * R_U, we keep Q_U
    std::cout << "  Computing QR decomposition of U (" << m << "x" << n << ")..." << std::endl;
    rocsolver_dgeqrf(handle, m, n, d_U, m, d_tau_U);
    rocsolver_dorgqr(handle, m, n, n, d_U, m, d_tau_U);
    
    // QR decomposition of V: V = Q_V * R_V, we keep Q_V
    std::cout << "  Computing QR decomposition of V (" << n << "x" << n << ")..." << std::endl;
    rocsolver_dgeqrf(handle, n, n, d_V, n, d_tau_V);
    rocsolver_dorgqr(handle, n, n, n, d_V, n, d_tau_V);
    
    // Initialize S to zero
    int block_size = 256;
    int num_blocks_zero = (size_S + block_size - 1) / block_size;
    hipLaunchKernelGGL(zero_matrix_kernel, dim3(num_blocks_zero), dim3(block_size), 
                       0, 0, d_S, size_S);
    
    // Compute singular values: 10^(linspace(0, log10KA, n))
    std::cout << "  Setting up singular values..." << std::endl;
    int num_blocks_sv = (n + block_size - 1) / block_size;
    hipLaunchKernelGGL(compute_singular_values_kernel, dim3(num_blocks_sv), dim3(block_size),
                       0, 0, d_singular_values, n, (double)log10KA);
    
    // Set diagonal of S
    hipLaunchKernelGGL(set_diagonal_kernel, dim3(num_blocks_sv), dim3(block_size),
                       0, 0, d_S, n, n, d_singular_values);
    
    HIP_CHECK(hipDeviceSynchronize());
    
    // Compute A = U * S * V'
    // First: temp = U * S (m x n) * (n x n) = (m x n)
    // Then:  A = temp * V' (m x n) * (n x n) = (m x n)
    
    std::cout << "  Computing A = U * S * V'..." << std::endl;
    
    double alpha = 1.0;
    double beta = 0.0;
    
    // temp = U * S
    ROCBLAS_CHECK(rocblas_dgemm(handle,
                                rocblas_operation_none,    // U
                                rocblas_operation_none,    // S
                                m, n, n,                   // m, n, k
                                &alpha,
                                d_U, m,                    // A, lda
                                d_S, n,                    // B, ldb
                                &beta,
                                d_temp, m));               // C, ldc
    
    // A = temp * V' = temp * V^T
    ROCBLAS_CHECK(rocblas_dgemm(handle,
                                rocblas_operation_none,       // temp
                                rocblas_operation_transpose,  // V'
                                m, n, n,                      // m, n, k
                                &alpha,
                                d_temp, m,                    // A, lda
                                d_V, n,                       // B, ldb
                                &beta,
                                d_A, m));                     // C, ldc
    
    HIP_CHECK(hipDeviceSynchronize());
    
    // Copy result to host and write to file
    std::cout << "  Copying result to host..." << std::endl;
    std::vector<double> h_A(size_A);
    HIP_CHECK(hipMemcpy(h_A.data(), d_A, size_A * sizeof(double), hipMemcpyDeviceToHost));
    
    // Build full filepath with output folder
    std::string filename = output_folder + "/M_cond_1e" + std::to_string(log10KA) + ".txt";
    
    std::cout << "  Writing matrix to file..." << std::endl;
    write_matrix_to_file(h_A.data(), m, n, filename);
    
    // Free device memory
    HIP_CHECK(hipFree(d_U));
    HIP_CHECK(hipFree(d_V));
    HIP_CHECK(hipFree(d_S));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_temp));
    HIP_CHECK(hipFree(d_tau_U));
    HIP_CHECK(hipFree(d_tau_V));
    HIP_CHECK(hipFree(d_singular_values));
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "  Completed in " << duration.count() << " seconds" << std::endl;
    std::cout << std::endl;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [m] [n] [output_folder]" << std::endl;
    std::cout << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  m             - Number of rows (default: 54272)" << std::endl;
    std::cout << "  n             - Number of columns (default: 54272)" << std::endl;
    std::cout << "  output_folder - Folder to save matrices (default: matrices_MxN_YYYYMMDD_HHMMSS)" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << "                      # 54272x54272, auto folder" << std::endl;
    std::cout << "  " << program_name << " 4096 4096             # 4096x4096, auto folder" << std::endl;
    std::cout << "  " << program_name << " 4096 4096 my_matrices # 4096x4096, folder 'my_matrices'" << std::endl;
}

int main(int argc, char* argv[]) {
    // Check for help flag
    if (argc >= 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        print_usage(argv[0]);
        return 0;
    }
    
    // Default matrix dimensions (same as MATLAB code)
    int m = 54272;
    int n = 54272;
    std::string output_folder;
    
    // Parse command line arguments
    if (argc >= 3) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
    }
    
    if (argc >= 4) {
        output_folder = argv[3];
    } else {
        // Generate folder name based on matrix size and timestamp
        output_folder = "matrices_" + std::to_string(m) + "x" + std::to_string(n) + "_" + generate_timestamp();
    }
    
    std::cout << "==================================================" << std::endl;
    std::cout << "ROCm Matrix Generator with Prescribed Condition Number" << std::endl;
    std::cout << "Matrix size: " << m << " x " << n << std::endl;
    std::cout << "Output folder: " << output_folder << std::endl;
    std::cout << "==================================================" << std::endl;
    
    // Create output directory
    if (!create_directory(output_folder)) {
        std::cerr << "Failed to create output directory. Exiting." << std::endl;
        return 1;
    }
    std::cout << "Output directory created/verified: " << output_folder << std::endl;
    
    // Check available GPU memory
    size_t free_mem, total_mem;
    HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));
    std::cout << "GPU Memory: " << free_mem / (1024*1024*1024.0) << " GB free / "
              << total_mem / (1024*1024*1024.0) << " GB total" << std::endl;
    
    // Estimate memory requirements
    size_t required_mem = (size_t)m * n * sizeof(double) * 4 +  // U, temp, A, and intermediate
                          (size_t)n * n * sizeof(double) * 2;   // V, S
    std::cout << "Estimated memory requirement: " << required_mem / (1024*1024*1024.0) 
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
    
    // Generate matrices for condition numbers 10^2 to 10^17
    for (int i = 1; i <= 16; i++) {
        int log10KA = 1 + i;  // Matches MATLAB: 2, 3, ..., 17
        generate_matrix(handle, gen, m, n, log10KA, output_folder);
    }
    
    // Cleanup
    HIPRAND_CHECK(hiprandDestroyGenerator(gen));
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    
    std::cout << "==================================================" << std::endl;
    std::cout << "All matrices generated successfully!" << std::endl;
    std::cout << "Output folder: " << output_folder << std::endl;
    std::cout << "==================================================" << std::endl;
    
    return 0;
}
