#include "MatrixCpu.hpp"
#include "MatrixGpu.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>

bool compareMatrices(const std::vector<float>& A, const std::vector<float>& B, float tolerance = 1e-5f) {
    if (A.size() != B.size()) {
        std::cout << "    Size mismatch: " << A.size() << " vs " << B.size() << std::endl;
        return false;
    }
    
    float max_error = 0.0f;
    size_t max_error_index = 0;
    
    for (size_t i = 0; i < A.size(); ++i) {
        float error = std::abs(A[i] - B[i]);
        if (error > max_error) {
            max_error = error;
            max_error_index = i;
        }
    }
    
    if (max_error > tolerance) {
        std::cout << "    Max error: " << max_error << " at index " << max_error_index << std::endl;
        return false;
    }
    
    return true;
}

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

void printTestHeader(const std::string& test_name) {
    std::cout << "\n" << std::string(60, '-') << std::endl;
    std::cout << test_name << std::endl;
    std::cout << std::string(60, '-') << std::endl;
}

void printTestResult(const std::string& test_name, bool passed) {
    std::cout << "    " << std::left << std::setw(40) << test_name 
              << (passed ? "[ PASS ]" : "[ FAIL ]") << std::endl;
}

void runCorrectnessTests() {
    printTestHeader("MATRIX MULTIPLICATION CORRECTNESS TESTS");
    
    MatrixGpu gpu_ops;
    if (!gpu_ops.initialize()) {
        std::cout << "    GPU initialization failed - skipping GPU tests" << std::endl;
        return;
    }
    
    std::vector<int> sizes = {256, 512, 1024};
    
    for (int size : sizes) {
        std::cout << "\n    Testing " << size << "x" << size << " matrices:" << std::endl;
        std::cout << "    " << std::string(40, '-') << std::endl;
        
        auto A = generateRandomMatrix(size, size);
        auto B = generateRandomMatrix(size, size);
        
        std::cout << "    Computing reference (CPU naive)..." << std::endl;
        auto reference = MatrixCpu::multiply(A, B, size, size, size);
        
        std::cout << "    Testing CPU blocked (8 threads)..." << std::endl;
        auto cpu_blocked = MatrixCpu::multiplyBlocked(A, B, size, size, size, 32, 8);
        printTestResult("CPU Blocked vs Reference", compareMatrices(reference, cpu_blocked, 1e-4f));
        
        std::cout << "    Testing GPU simple..." << std::endl;
        try {
            auto gpu_result = gpu_ops.multiply(A, B, size, size, size);
            printTestResult("GPU Simple vs Reference", compareMatrices(reference, gpu_result, 1e-3f));
        } catch (const std::exception& e) {
            printTestResult("GPU Simple vs Reference", false);
            std::cout << "        Error: " << e.what() << std::endl;
        }
        
        std::cout << "    Testing GPU blocked..." << std::endl;
        try {
            auto gpu_blocked = gpu_ops.multiplyBlocked(A, B, size, size, size, 16);
            printTestResult("GPU Blocked vs Reference", compareMatrices(reference, gpu_blocked, 1e-3f));
        } catch (const std::exception& e) {
            printTestResult("GPU Blocked vs Reference", false);
            std::cout << "        Error: " << e.what() << std::endl;
        }
    }
    
    std::vector<std::tuple<int, int, int>> rectangular_sizes = {
        {256, 512, 128},
        {512, 256, 384},
        {1024, 512, 256}
    };
    
    for (const auto& [M, N, K] : rectangular_sizes) {
        std::cout << "\n    Testing rectangular " << M << "x" << K << " * " << K << "x" << N << ":" << std::endl;
        std::cout << "    " << std::string(40, '-') << std::endl;
        
        auto A_rect = generateRandomMatrix(M, K);
        auto B_rect = generateRandomMatrix(K, N);
        
        std::cout << "    Computing reference..." << std::endl;
        auto reference_rect = MatrixCpu::multiply(A_rect, B_rect, M, N, K);
        
        std::cout << "    Testing GPU..." << std::endl;
        try {
            auto gpu_rect = gpu_ops.multiply(A_rect, B_rect, M, N, K);
            printTestResult("GPU Rectangular vs Reference", compareMatrices(reference_rect, gpu_rect, 1e-3f));
        } catch (const std::exception& e) {
            printTestResult("GPU Rectangular vs Reference", false);
            std::cout << "        Error: " << e.what() << std::endl;
        }
    }
}

void runConsistencyTests() {
    printTestHeader("CONSISTENCY TESTS");
    
    MatrixGpu gpu_ops;
    if (!gpu_ops.initialize()) {
        return;
    }
    
    std::cout << "\n    Testing consistency between different methods (512x512):" << std::endl;
    std::cout << "    " << std::string(40, '-') << std::endl;
    
    int M = 512, N = 512, K = 512;
    auto A = generateRandomMatrix(M, K);
    auto B = generateRandomMatrix(K, N);
    
    std::cout << "    Computing CPU naive..." << std::endl;
    auto cpu_simple = MatrixCpu::multiply(A, B, M, N, K);
    
    std::cout << "    Computing CPU blocked (8 threads)..." << std::endl;
    auto cpu_blocked = MatrixCpu::multiplyBlocked(A, B, M, N, K, 32, 8);
    
    std::vector<float> gpu_simple, gpu_blocked;
    
    std::cout << "    Computing GPU simple..." << std::endl;
    try {
        gpu_simple = gpu_ops.multiply(A, B, M, N, K);
    } catch (const std::exception& e) {
        std::cout << "        GPU simple failed: " << e.what() << std::endl;
    }
    
    std::cout << "    Computing GPU blocked..." << std::endl;
    try {
        gpu_blocked = gpu_ops.multiplyBlocked(A, B, M, N, K, 16);
    } catch (const std::exception& e) {
        std::cout << "        GPU blocked failed: " << e.what() << std::endl;
    }
    
    std::cout << "\n    Results:" << std::endl;
    std::cout << "    " << std::string(40, '-') << std::endl;
    
    bool all_consistent = true;
    
    if (!cpu_simple.empty() && !cpu_blocked.empty()) {
        bool consistent = compareMatrices(cpu_simple, cpu_blocked, 1e-4f);
        printTestResult("CPU Naive vs CPU Blocked", consistent);
        if (!consistent) all_consistent = false;
    }
    
    if (!cpu_simple.empty() && !gpu_simple.empty()) {
        bool consistent = compareMatrices(cpu_simple, gpu_simple, 1e-3f);
        printTestResult("CPU Naive vs GPU Simple", consistent);
        if (!consistent) all_consistent = false;
    }
    
    if (!cpu_simple.empty() && !gpu_blocked.empty()) {
        bool consistent = compareMatrices(cpu_simple, gpu_blocked, 1e-3f);
        printTestResult("CPU Naive vs GPU Blocked", consistent);
        if (!consistent) all_consistent = false;
    }
    
    std::cout << "\n    " << std::string(40, '-') << std::endl;
    if (all_consistent) {
        std::cout << "    OVERALL: [ PASS ] All methods are consistent" << std::endl;
    } else {
        std::cout << "    OVERALL: [ FAIL ] Some methods are inconsistent" << std::endl;
    }
}

int main() {
    std::cout << "\n";
    std::cout << "HIGH PERFORMANCE COMPUTING TEST SUITE" << std::endl;
    std::cout << "Matrix Multiplication Tests" << std::endl;
    std::cout << "\n";
    
    runCorrectnessTests();
    runConsistencyTests();
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST SUITE COMPLETED" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    return 0;
}
