#include "../include/gpu_helper.h"
#include "../include/matrix_multiply.h"
#include "../include/merge_sort.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>

void print_help() {
    std::cout << "GPU Performance Test\n"
              << "Usage: ./gpu_test [options]\n"
              << "Options:\n"
              << "  --task TASK        Task to run: matmul, sort, all (default: all)\n"
              << "  --size N           Size for sorting (default: 100000)\n"
              << "  --rows R           Rows for matrix A (default: 512)\n"
              << "  --cols C           Columns for matrix A / rows for B (default: 512)\n"
              << "  --k K              Columns for matrix B (default: 512)\n"
              << "  --threads T        Comma-separated local sizes to test (default: 1,2,4,8)\n"
              << "  --verify           Verify results against CPU\n"
              << "  --help             Show this help\n";
}

int main(int argc, char* argv[]) {
    std::string task = "all";
    size_t sort_size = 100000;
    size_t rows = 512;
    size_t cols = 512;
    size_t k = 512;
    std::vector<size_t> thread_tests = {1, 2, 4, 8};
    bool verify = false;
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--task" && i+1 < argc) {
            task = argv[++i];
        } else if (arg == "--size" && i+1 < argc) {
            sort_size = std::stoul(argv[++i]);
        } else if (arg == "--rows" && i+1 < argc) {
            rows = std::stoul(argv[++i]);
        } else if (arg == "--cols" && i+1 < argc) {
            cols = std::stoul(argv[++i]);
        } else if (arg == "--k" && i+1 < argc) {
            k = std::stoul(argv[++i]);
        } else if (arg == "--threads" && i+1 < argc) {
            thread_tests.clear();
            std::string threads_str = argv[++i];
            size_t pos = 0;
            while ((pos = threads_str.find(',')) != std::string::npos) {
                thread_tests.push_back(std::stoul(threads_str.substr(0, pos)));
                threads_str.erase(0, pos + 1);
            }
            thread_tests.push_back(std::stoul(threads_str));
        } else if (arg == "--verify") {
            verify = true;
        } else if (arg == "--help") {
            print_help();
            return 0;
        }
    }
    
    try {
        // Task 4.1: Matrix Multiplication
        if (task == "matmul" || task == "all") {
            std::cout << "\n=== Matrix Multiplication Benchmark ===\n";
            std::cout << "Matrix A: " << rows << "x" << cols << ", Matrix B: " << cols << "x" << k << "\n";
            
            // Generate random matrices
            std::vector<float> A = random_vector<float>(rows * cols);
            std::vector<float> B = random_vector<float>(cols * k);
            
            MatrixMultiplier multiplier;
            
            // CPU baseline
            Timer timer;
            timer.start();
            std::vector<float> C_cpu = MatrixMultiplier::multiply_cpu_blocked(A, B, rows, cols, k, 64);
            timer.stop();
            double cpu_time = timer.elapsed();
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "CPU blocked time: " << cpu_time << " s\n";
            
            // Test different local sizes
            std::cout << "\nGPU performance:\n";
            std::cout << "Local Size\tTime (s)\tSpeedup\n";
            std::cout << std::string(40, '-') << "\n";
            
            for (size_t local_size : thread_tests) {
                try {
                    timer.start();
                    std::vector<float> C_gpu = multiplier.multiply(A, B, rows, cols, k, local_size);
                    timer.stop();
                    double gpu_time = timer.elapsed();
                    double speedup = cpu_time / gpu_time;
                    
                    std::cout << std::setw(10) << local_size << "\t"
                            << std::fixed << std::setprecision(6) << gpu_time << "\t"
                            << std::setprecision(2) << speedup << "x\n";
                    
                    if (verify) {
                        if (MatrixMultiplier::verify(C_cpu, C_gpu)) {
                            std::cout << "  ✓ Results match\n";
                        } else {
                            std::cout << "  ✗ Results differ!\n";
                        }
                    }
                } catch (const std::exception& e) {
                    std::cout << "Local size " << local_size << " failed: " << e.what() << "\n";
                }
            }
        }
        
        // Task 4.2: Merge Sort
        if (task == "sort" || task == "all") {
            std::cout << "\n=== Merge Sort Benchmark ===\n";
            std::cout << "Data size: " << sort_size << " elements\n";
            
            std::vector<float> data = random_vector<float>(sort_size);
            
            MergeSorter sorter;
            
            // CPU baseline
            Timer timer;
            timer.start();
            std::vector<float> sorted_cpu = MergeSorter::sort_cpu(data);
            timer.stop();
            double cpu_time = timer.elapsed();
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "CPU std::sort time: " << cpu_time << " s\n";
            
            // Test different local sizes
            std::cout << "\nGPU performance (Bitonic sort):\n";
            std::cout << "Local Size\tTime (s)\tSpeedup\n";
            std::cout << std::string(40, '-') << "\n";
            
            for (size_t local_size : thread_tests) {
                try {
                    timer.start();
                    std::vector<float> sorted_gpu = sorter.sort(data, local_size);
                    timer.stop();
                    double gpu_time = timer.elapsed();
                    double speedup = cpu_time / gpu_time;
                    
                    std::cout << std::setw(10) << local_size << "\t"
                            << std::fixed << std::setprecision(6) << gpu_time << "\t"
                            << std::setprecision(2) << speedup << "x\n";
                    
                    if (verify) {
                        if (MergeSorter::verify(sorted_gpu)) {
                            std::cout << "  ✓ Array is sorted\n";
                        } else {
                            std::cout << "  ✗ Array not sorted!\n";
                        }
                    }
                } catch (const std::exception& e) {
                    std::cout << "Local size " << local_size << " failed: " << e.what() << "\n";
                }
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
