#include "MatrixCpu.hpp"
#include "MatrixGpu.hpp"
#include <iostream>
#include <vector>

int main() {
    std::cout << "Matrix Multiplication Demo" << std::endl;
    std::cout << "==========================" << std::endl;
    
    std::vector<float> A = {1, 2, 3, 4, 5, 6};
    std::vector<float> B = {7, 8, 9, 10, 11, 12};
    
    std::cout << std::endl;
    std::cout << "CPU Multiplication (2x3 * 3x2):" << std::endl;
    auto cpu_result = MatrixCpu::multiply(A, B, 2, 2, 3);
    
    std::cout << "Result:" << std::endl;
    for (int i = 0; i < 2; ++i) {
        std::cout << "  ";
        for (int j = 0; j < 2; ++j) {
            std::cout << cpu_result[i * 2 + j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << std::endl;
    MatrixGpu gpu_ops;
    if (gpu_ops.initialize()) {
        std::cout << "GPU Multiplication (2x3 * 3x2):" << std::endl;
        auto gpu_result = gpu_ops.multiply(A, B, 2, 2, 3);
        
        std::cout << "Result:" << std::endl;
        for (int i = 0; i < 2; ++i) {
            std::cout << "  ";
            for (int j = 0; j < 2; ++j) {
                std::cout << gpu_result[i * 2 + j] << " ";
            }
            std::cout << std::endl;
        }
    } else {
        std::cout << "GPU not available" << std::endl;
    }
    
    return 0;
}
