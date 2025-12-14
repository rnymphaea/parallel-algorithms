#include "../include/matrix_ops/cpu.hpp"
#include "../include/matrix_ops/gpu.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <random>
#include <iomanip>

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_;
    
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - start_).count();
    }
};

std::vector<float> generateRandomMatrix(int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    std::vector<float> matrix(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = dis(gen);
    }
    return matrix;
}

struct BenchmarkConfig {
    std::vector<int> matrix_sizes;
    std::vector<int> thread_counts;
    std::vector<size_t> workgroup_sizes;
    std::string filename;
};

BenchmarkConfig getBenchmarkConfig() {
    return BenchmarkConfig{
        .matrix_sizes = {64, 128, 256, 512, 1024, 2000, 2500, 3000},
        .thread_counts = {4, 8},
        .workgroup_sizes = {16},
        .filename = "matrix_benchmarks.csv"
    };
}

void runAllBenchmarks(const BenchmarkConfig& config) {
    std::ofstream file(config.filename);
    file << "TestType,MatrixSize,Threads,WorkgroupSize,Time,CPUTime,GPUTime\n";
    
    std::cout << "Running CPU performance tests..." << std::endl;
    for (int size : config.matrix_sizes) {
        int M = size, N = size, K = size;
        auto A = generateRandomMatrix(M, K);
        auto B = generateRandomMatrix(K, N);
        
        for (int threads : config.thread_counts) {
            Timer timer;
            auto result = MatrixCPU::multiplyBlocked(A, B, M, N, K, 32, threads);
            double time = timer.elapsed();
            
            file << "CPU," << size << "," << threads << ",,"
                 << time << ",,\n";
            
            std::cout << "  Size: " << size << " Threads: " << threads 
                      << " Time: " << time << "s" << std::endl;
        }
    }
    
    std::cout << "\nRunning GPU performance tests..." << std::endl;
    MatrixGPU gpu_ops;
    if (!gpu_ops.initialize()) {
        std::cout << "GPU initialization failed - skipping GPU benchmarks" << std::endl;
        file.close();
        return;
    }
    
    for (int size : config.matrix_sizes) {
        int M = size, N = size, K = size;
        auto A = generateRandomMatrix(M, K);
        auto B = generateRandomMatrix(K, N);
        
        for (size_t workgroup : config.workgroup_sizes) {
            try {
                Timer timer;
                auto result = gpu_ops.multiplyBlocked(A, B, M, N, K, workgroup);
                double time = timer.elapsed();
                
                file << "GPU," << size << ",," << workgroup << ","
                     << time << ",,\n";
                
                std::cout << "  Size: " << size << " Workgroup: " << workgroup 
                          << " Time: " << time << "s" << std::endl;
                
            } catch (const std::exception& e) {
                std::cout << "  Error with workgroup " << workgroup << ": " << e.what() << std::endl;
            }
        }
    }
    
    std::cout << "\nRunning CPU-GPU comparison..." << std::endl;
    int cpu_threads = 8;
    size_t gpu_workgroup = 16;
    
    for (int size : config.matrix_sizes) {
        int M = size, N = size, K = size;
        auto A = generateRandomMatrix(M, K);
        auto B = generateRandomMatrix(K, N);
        
        Timer timer;
        auto cpu_result = MatrixCPU::multiplyBlocked(A, B, M, N, K, 32, cpu_threads);
        double cpu_time = timer.elapsed();
        
        timer = Timer();
        auto gpu_result = gpu_ops.multiplyBlocked(A, B, M, N, K, gpu_workgroup);
        double gpu_time = timer.elapsed();
        
        file << "CPU_GPU_Comparison," << size << "," << cpu_threads << "," << gpu_workgroup << ","
             << "," << cpu_time << "," << gpu_time << "\n";
        
        std::cout << "  Size: " << size << " CPU: " << cpu_time << "s GPU: " << gpu_time 
                  << "s Speedup: " << (cpu_time / gpu_time) << std::endl;
    }
    
    file.close();
}

int main() {
    BenchmarkConfig config = getBenchmarkConfig();
    runAllBenchmarks(config);
    
    return 0;
}
