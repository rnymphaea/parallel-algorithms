#include "../include/matrix_ops/cpu.hpp"
#include "../include/matrix_ops/gpu.hpp"
#include <iostream>
#include <vector>

int main() {
    std::cout << "Matrix Multiplication Demo" << std::endl;
    
    std::vector<float> A = {1, 2, 3, 4, 5, 6};
    std::vector<float> B = {7, 8, 9, 10, 11, 12};
    
    std::cout << "CPU Multiplication:" << std::endl;
    auto cpu_result = MatrixCPU::multiply(A, B, 2, 2, 3);
    
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            std::cout << cpu_result[i * 2 + j] << " ";
        }
        std::cout << std::endl;
    }
    
    MatrixGPU gpu_ops;
    if (gpu_ops.initialize()) {
        std::cout << "\nGPU Multiplication:" << std::endl;
        auto gpu_result = gpu_ops.multiply(A, B, 2, 2, 3);
        
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
