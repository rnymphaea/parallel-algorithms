#include "MatrixCpu.hpp"
#include "MatrixGpu.hpp"
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
        .matrix_sizes = {64, 128, 256, 512, 1024, 2048},
        .thread_counts = {1, 2, 4, 8},
        .workgroup_sizes = {16},
        .filename = "matrix_benchmarks.csv"
    };
}

void printHeader(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << title << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

void printTableHeader() {
    std::cout << std::string(60, '-') << std::endl;
    std::cout << std::left 
              << std::setw(12) << "Size" 
              << std::setw(12) << "Threads/WG" 
              << std::setw(15) << "Time (s)" 
              << std::setw(10) << "Status" 
              << std::endl;
    std::cout << std::string(60, '-') << std::endl;
}

void printResult(int size, const std::string& config, double time, bool success = true) {
    std::cout << std::left 
              << std::setw(12) << std::to_string(size) + "x" + std::to_string(size)
              << std::setw(12) << config 
              << std::setw(15) << std::fixed << std::setprecision(6) << time
              << std::setw(10) << (success ? "OK" : "FAIL") 
              << std::endl;
}

void printComparisonTableHeader() {
    std::cout << std::string(70, '-') << std::endl;
    std::cout << std::left 
              << std::setw(12) << "Size" 
              << std::setw(12) << "CPU 8 thr" 
              << std::setw(12) << "GPU WG 16" 
              << std::setw(12) << "Speedup" 
              << std::setw(15) << "Status" 
              << std::endl;
    std::cout << std::string(70, '-') << std::endl;
}

void printComparisonResult(int size, double cpu_time, double gpu_time, double speedup) {
    std::string status = speedup > 1.0 ? "GPU FASTER" : "CPU FASTER";
    std::string speedup_str = std::to_string(speedup);
    size_t dot_pos = speedup_str.find('.');
    if (dot_pos != std::string::npos && speedup_str.length() > dot_pos + 3) {
        speedup_str = speedup_str.substr(0, dot_pos + 3);
    }
    
    std::cout << std::left 
              << std::setw(12) << std::to_string(size) + "x" + std::to_string(size)
              << std::setw(12) << std::fixed << std::setprecision(6) << cpu_time
              << std::setw(12) << std::fixed << std::setprecision(6) << gpu_time
              << std::setw(12) << speedup_str + "x"
              << std::setw(15) << status 
              << std::endl;
}

void runAllBenchmarks(const BenchmarkConfig& config) {
    std::ofstream file(config.filename);
    file << "TestType,MatrixSize,Threads,WorkgroupSize,Time,CPUTime,GPUTime\n";
    
    printHeader("MATRIX MULTIPLICATION BENCHMARKS");
    
    printHeader("CPU PERFORMANCE TESTS");
    printTableHeader();
    
    for (int size : config.matrix_sizes) {
        int M = size, N = size, K = size;
        auto A = generateRandomMatrix(M, K);
        auto B = generateRandomMatrix(K, N);
        
        for (int threads : config.thread_counts) {
            Timer timer;
            auto result = MatrixCpu::multiplyBlocked(A, B, M, N, K, 32, threads);
            double time = timer.elapsed();
            
            file << "CPU," << size << "," << threads << ",,"
                 << time << ",,\n";
            
            printResult(size, std::to_string(threads) + " threads", time);
        }
        std::cout << std::endl;
    }
    
    printHeader("GPU PERFORMANCE TESTS");
    printTableHeader();
    
    MatrixGpu gpu_ops;
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
                
                printResult(size, "WG " + std::to_string(workgroup), time);
                
            } catch (const std::exception& e) {
                printResult(size, "WG " + std::to_string(workgroup), 0.0, false);
                std::cout << "        Error: " << e.what() << std::endl;
            }
        }
        std::cout << std::endl;
    }
    
    printHeader("CPU vs GPU COMPARISON");
    printComparisonTableHeader();
    
    for (int size : config.matrix_sizes) {
        int M = size, N = size, K = size;
        auto A = generateRandomMatrix(M, K);
        auto B = generateRandomMatrix(K, N);
        
        Timer timer;
        auto cpu_result = MatrixCpu::multiplyBlocked(A, B, M, N, K, 32, 8);
        double cpu_time = timer.elapsed();
        
        timer = Timer();
        auto gpu_result = gpu_ops.multiplyBlocked(A, B, M, N, K, 16);
        double gpu_time = timer.elapsed();
        
        double speedup = cpu_time / gpu_time;
        
        file << "CPU_GPU_Comparison," << size << ",8,16,,"
             << cpu_time << "," << gpu_time << "\n";
        
        printComparisonResult(size, cpu_time, gpu_time, speedup);
    }
    
    std::cout << std::string(70, '-') << std::endl;
    
    file.close();
    
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Benchmark results saved to " << config.filename << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

int main() {
    std::cout << "\n";
    std::cout << "HIGH PERFORMANCE COMPUTING BENCHMARK SUITE" << std::endl;
    std::cout << "Matrix Multiplication: CPU vs GPU" << std::endl;
    std::cout << "\n";
    
    BenchmarkConfig config = getBenchmarkConfig();
    runAllBenchmarks(config);
    
    return 0;
}
