/**
 * Simple ROCm A/B Matrix Generator
 * 
 * Generates two random matrices A (N x K) and B (K x M) for GEMM benchmarking.
 * Matrices are filled with random values from a normal distribution.
 * 
 * Compile with:
 *   hipcc -O3 simple_matrix_generator.cpp -o simple_matrix_generator \
 *         -lhiprand -I/opt/rocm/include -L/opt/rocm/lib
 * 
 * Usage:
 *   ./simple_matrix_generator N M K [output_folder]
 *   
 *   N             - Number of rows of A and C
 *   M             - Number of columns of B and C
 *   K             - Inner dimension (columns of A, rows of B)
 *   output_folder - Folder to save matrices (optional, default: current directory)
 */

#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>

#include <iostream>
#include <fstream>
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

#define HIPRAND_CHECK(call)                                                     \
    do {                                                                        \
        hiprandStatus_t status = call;                                          \
        if (status != HIPRAND_STATUS_SUCCESS) {                                 \
            std::cerr << "hipRAND error at " << __FILE__ << ":" << __LINE__     \
                      << " status=" << status << std::endl;                     \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

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
    std::cout << "  Written to " << filepath << std::endl;
}

void generate_and_save_matrix(hiprandGenerator_t gen,
                               size_t rows, size_t cols,
                               const std::string& name,
                               const std::string& output_folder) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "Generating matrix " << name << " (" << rows << " x " << cols << ")..." << std::endl;
    
    size_t size = rows * cols;
    size_t size_bytes = size * sizeof(double);
    
    // Device memory allocation
    double* d_matrix;
    HIP_CHECK(hipMalloc(&d_matrix, size_bytes));
    
    // Generate random matrix using normal distribution (mean=0, stddev=1)
    std::cout << "  Generating random values..." << std::endl;
    HIPRAND_CHECK(hiprandGenerateNormalDouble(gen, d_matrix, size, 0.0, 1.0));
    HIP_CHECK(hipDeviceSynchronize());
    
    // Copy result to host
    std::cout << "  Copying to host..." << std::endl;
    std::vector<double> h_matrix(size);
    HIP_CHECK(hipMemcpy(h_matrix.data(), d_matrix, size_bytes, hipMemcpyDeviceToHost));
    
    // Build filename: A_NxK.txt or B_KxM.txt
    std::string filename = output_folder + "/" + name + "_" + 
                           std::to_string(rows) + "x" + std::to_string(cols) + ".txt";
    
    std::cout << "  Writing to file..." << std::endl;
    write_matrix_to_file(h_matrix.data(), rows, cols, filename);
    
    // Free device memory
    HIP_CHECK(hipFree(d_matrix));
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "  Completed in " << duration.count() << " ms" << std::endl;
    std::cout << std::endl;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " N M K [output_folder]" << std::endl;
    std::cout << std::endl;
    std::cout << "Generates two random matrices for GEMM: C = A * B" << std::endl;
    std::cout << "  A: N x K matrix" << std::endl;
    std::cout << "  B: K x M matrix" << std::endl;
    std::cout << "  C: N x M matrix (result)" << std::endl;
    std::cout << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  N             - Number of rows of A (and C)" << std::endl;
    std::cout << "  M             - Number of columns of B (and C)" << std::endl;
    std::cout << "  K             - Inner dimension (columns of A, rows of B)" << std::endl;
    std::cout << "  output_folder - Folder to save matrices (optional, default: current directory)" << std::endl;
    std::cout << std::endl;
    std::cout << "Output files:" << std::endl;
    std::cout << "  A_NxK.txt     - Matrix A in CSV format" << std::endl;
    std::cout << "  B_KxM.txt     - Matrix B in CSV format" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << " 1024 1024 1024           # Square 1024x1024 matrices" << std::endl;
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
    std::string output_folder = ".";
    if (argc >= 5) {
        output_folder = argv[4];
    }
    
    std::cout << "==================================================" << std::endl;
    std::cout << "Simple ROCm A/B Matrix Generator" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Matrix A: " << N << " x " << K << std::endl;
    std::cout << "Matrix B: " << K << " x " << M << std::endl;
    std::cout << "Result C: " << N << " x " << M << " (for reference)" << std::endl;
    std::cout << "Output folder: " << output_folder << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << std::endl;
    
    // Create output directory if needed
    if (output_folder != ".") {
        if (!create_directory(output_folder)) {
            std::cerr << "Failed to create output directory. Exiting." << std::endl;
            return 1;
        }
        std::cout << "Output directory created/verified: " << output_folder << std::endl;
    }
    
    // Check available GPU memory
    size_t free_mem, total_mem;
    HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));
    std::cout << "GPU Memory: " << free_mem / (1024.0*1024.0*1024.0) << " GB free / "
              << total_mem / (1024.0*1024.0*1024.0) << " GB total" << std::endl;
    
    // Estimate memory requirements (we generate one matrix at a time)
    size_t max_matrix_size = std::max(N * K, K * M);
    size_t required_mem = max_matrix_size * sizeof(double);
    std::cout << "Max matrix memory requirement: " << required_mem / (1024.0*1024.0*1024.0) 
              << " GB" << std::endl;
    
    if (required_mem > free_mem) {
        std::cerr << "Warning: May not have enough GPU memory!" << std::endl;
    }
    std::cout << std::endl;
    
    // Initialize hipRAND
    hiprandGenerator_t gen;
    HIPRAND_CHECK(hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT));
    HIPRAND_CHECK(hiprandSetPseudoRandomGeneratorSeed(gen, 42ULL));
    
    // Generate matrix A (N x K)
    generate_and_save_matrix(gen, N, K, "A", output_folder);
    
    // Generate matrix B (K x M)
    generate_and_save_matrix(gen, K, M, "B", output_folder);
    
    // Cleanup
    HIPRAND_CHECK(hiprandDestroyGenerator(gen));
    
    std::cout << "==================================================" << std::endl;
    std::cout << "Matrices generated successfully!" << std::endl;
    std::cout << "  A_" << N << "x" << K << ".txt" << std::endl;
    std::cout << "  B_" << K << "x" << M << ".txt" << std::endl;
    std::cout << "Output folder: " << output_folder << std::endl;
    std::cout << "==================================================" << std::endl;
    
    return 0;
}
