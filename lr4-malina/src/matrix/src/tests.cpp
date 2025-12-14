#include "../include/cpu_matrix_operations.hpp"
#include "../include/gpu_matrix_operations.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

bool compareMatrices(const std::vector<float>& A, const std::vector<float>& B, float tolerance = 1e-5f) {
    if (A.size() != B.size()) {
        std::cout << "Size mismatch: " << A.size() << " vs " << B.size() << std::endl;
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
        std::cout << "Max error: " << max_error << " at index " << max_error_index << std::endl;
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

void runCorrectnessTests() {
    std::cout << "=== Running Correctness Tests on Large Matrices ===" << std::endl;
    
    GPUMatrixOperations gpu_ops;
    if (!gpu_ops.initialize()) {
        std::cout << "GPU initialization failed - skipping GPU tests" << std::endl;
        return;
    }
    
    // Тест 1: Большие квадратные матрицы
    std::vector<int> sizes = {256, 512, 1024};
    
    for (int size : sizes) {
        std::cout << "\nTesting size " << size << "x" << size << ":" << std::endl;
        
        // Генерируем случайные матрицы
        auto A = generateRandomMatrix(size, size);
        auto B = generateRandomMatrix(size, size);
        
        // Вычисляем эталонный результат обычным умножением
        std::cout << "  Computing reference with simple CPU multiplication..." << std::endl;
        auto reference = CPUMatrixOperations::multiplyMatrices(A, B, size, size, size);
        
        // Тестируем блочное умножение на CPU
        std::cout << "  Testing CPU blocked multiplication..." << std::endl;
        auto cpu_blocked = CPUMatrixOperations::multiplyMatricesBlocked(A, B, size, size, size, 32, 8);
        
        if (compareMatrices(reference, cpu_blocked, 1e-4f)) {
            std::cout << "  ✓ CPU blocked multiplication PASSED" << std::endl;
        } else {
            std::cout << "  ✗ CPU blocked multiplication FAILED" << std::endl;
        }
        
        // Тестируем GPU умножение
        std::cout << "  Testing GPU multiplication..." << std::endl;
        try {
            auto gpu_result = gpu_ops.multiplyMatrices(A, B, size, size, size);
            if (compareMatrices(reference, gpu_result, 1e-3f)) {
                std::cout << "  ✓ GPU simple multiplication PASSED" << std::endl;
            } else {
                std::cout << "  ✗ GPU simple multiplication FAILED" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "  ✗ GPU simple multiplication FAILED: " << e.what() << std::endl;
        }
        
        // Тестируем GPU блочное умножение
        std::cout << "  Testing GPU blocked multiplication..." << std::endl;
        try {
            auto gpu_blocked = gpu_ops.multiplyMatricesBlocked(A, B, size, size, size, 16);
            if (compareMatrices(reference, gpu_blocked, 1e-3f)) {
                std::cout << "  ✓ GPU blocked multiplication PASSED" << std::endl;
            } else {
                std::cout << "  ✗ GPU blocked multiplication FAILED" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "  ✗ GPU blocked multiplication FAILED: " << e.what() << std::endl;
        }
    }
    
    // Тест 2: Прямоугольные матрицы
    std::cout << "\n=== Testing Rectangular Matrices ===" << std::endl;
    std::vector<std::tuple<int, int, int>> rectangular_sizes = {
        {256, 512, 128},  // M=256, N=512, K=128
        {512, 256, 384},  // M=512, N=256, K=384
        {1024, 512, 256}  // M=1024, N=512, K=256
    };
    
    for (const auto& [M, N, K] : rectangular_sizes) {
        std::cout << "\nTesting rectangular " << M << "x" << K << " * " << K << "x" << N << ":" << std::endl;
        
        auto A_rect = generateRandomMatrix(M, K);
        auto B_rect = generateRandomMatrix(K, N);
        
        // Эталонное вычисление
        std::cout << "  Computing reference..." << std::endl;
        auto reference_rect = CPUMatrixOperations::multiplyMatrices(A_rect, B_rect, M, N, K);
        
        // GPU вычисление
        std::cout << "  Testing GPU..." << std::endl;
        try {
            auto gpu_rect = gpu_ops.multiplyMatrices(A_rect, B_rect, M, N, K);
            if (compareMatrices(reference_rect, gpu_rect, 1e-3f)) {
                std::cout << "  ✓ GPU rectangular multiplication PASSED" << std::endl;
            } else {
                std::cout << "  ✗ GPU rectangular multiplication FAILED" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "  ✗ GPU rectangular multiplication FAILED: " << e.what() << std::endl;
        }
    }
}

void runConsistencyTests() {
    std::cout << "\n=== Running Consistency Tests ===" << std::endl;
    
    GPUMatrixOperations gpu_ops;
    if (!gpu_ops.initialize()) {
        return;
    }
    
    // Тест 3: Сравнение разных методов между собой
    std::cout << "Testing consistency between different methods..." << std::endl;
    
    int M = 512, N = 512, K = 512;
    auto A = generateRandomMatrix(M, K);
    auto B = generateRandomMatrix(K, N);
    
    // Получаем результаты всеми методами
    auto cpu_simple = CPUMatrixOperations::multiplyMatrices(A, B, M, N, K);
    auto cpu_blocked = CPUMatrixOperations::multiplyMatricesBlocked(A, B, M, N, K, 32, 8);
    
    std::vector<float> gpu_simple, gpu_blocked;
    
    try {
        gpu_simple = gpu_ops.multiplyMatrices(A, B, M, N, K);
    } catch (const std::exception& e) {
        std::cout << "GPU simple failed: " << e.what() << std::endl;
    }
    
    try {
        gpu_blocked = gpu_ops.multiplyMatricesBlocked(A, B, M, N, K, 16);
    } catch (const std::exception& e) {
        std::cout << "GPU blocked failed: " << e.what() << std::endl;
    }
    
    // Сравниваем все между собой
    bool all_consistent = true;
    
    if (!cpu_simple.empty() && !cpu_blocked.empty()) {
        if (!compareMatrices(cpu_simple, cpu_blocked, 1e-4f)) {
            std::cout << "✗ CPU simple vs CPU blocked: INCONSISTENT" << std::endl;
            all_consistent = false;
        } else {
            std::cout << "✓ CPU simple vs CPU blocked: CONSISTENT" << std::endl;
        }
    }
    
    if (!cpu_simple.empty() && !gpu_simple.empty()) {
        if (!compareMatrices(cpu_simple, gpu_simple, 1e-3f)) {
            std::cout << "✗ CPU simple vs GPU simple: INCONSISTENT" << std::endl;
            all_consistent = false;
        } else {
            std::cout << "✓ CPU simple vs GPU simple: CONSISTENT" << std::endl;
        }
    }
    
    if (!cpu_simple.empty() && !gpu_blocked.empty()) {
        if (!compareMatrices(cpu_simple, gpu_blocked, 1e-3f)) {
            std::cout << "✗ CPU simple vs GPU blocked: INCONSISTENT" << std::endl;
            all_consistent = false;
        } else {
            std::cout << "✓ CPU simple vs GPU blocked: CONSISTENT" << std::endl;
        }
    }
    
    if (all_consistent) {
        std::cout << "✓ ALL METHODS ARE CONSISTENT" << std::endl;
    }
}

int main() {
    std::cout << "Starting Matrix Multiplication Correctness Tests\n" << std::endl;
    
    runCorrectnessTests();
    runConsistencyTests();
    
    std::cout << "\n=== Correctness Tests Completed ===" << std::endl;
    return 0;
}