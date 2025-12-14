#include "../include/cpu_matrix_operations.hpp"
#include "../include/gpu_matrix_operations.hpp"
#include <iostream>
#include <vector>

int main() {
    std::cout << "Matrix Multiplication Demo" << std::endl;
    
    // Example usage
    std::vector<float> A = {1, 2, 3, 4, 5, 6}; // 2x3
    std::vector<float> B = {7, 8, 9, 10, 11, 12}; // 3x2
    
    std::cout << "CPU Multiplication:" << std::endl;
    auto cpu_result = CPUMatrixOperations::multiplyMatrices(A, B, 2, 2, 3);
    
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            std::cout << cpu_result[i * 2 + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Try GPU if available
    GPUMatrixOperations gpu_ops;
    if (gpu_ops.initialize()) {
        std::cout << "\nGPU Multiplication:" << std::endl;
        auto gpu_result = gpu_ops.multiplyMatrices(A, B, 2, 2, 3);
        
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                std::cout << gpu_result[i * 2 + j] << " ";
            }
            std::cout << std::endl;
        }
    } else {
        std::cout << "\nGPU not available" << std::endl;
    }
    
    return 0;
}