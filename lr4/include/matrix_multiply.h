#ifndef MATRIX_MULTIPLY_H
#define MATRIX_MULTIPLY_H

#include "gpu_helper.h"
#include <vector>

class MatrixMultiplier {
public:
    MatrixMultiplier();
    ~MatrixMultiplier();
    
    std::vector<float> multiply(const std::vector<float>& A, const std::vector<float>& B,
                               size_t rowsA, size_t colsA, size_t colsB,
                               size_t local_size = 0);
    
    // CPU implementations for comparison
    static std::vector<float> multiply_cpu_simple(const std::vector<float>& A, const std::vector<float>& B,
                                                 size_t rowsA, size_t colsA, size_t colsB);
    static std::vector<float> multiply_cpu_blocked(const std::vector<float>& A, const std::vector<float>& B,
                                                  size_t rowsA, size_t colsA, size_t colsB,
                                                  size_t block_size = 64);
    
    static bool verify(const std::vector<float>& C1, const std::vector<float>& C2, float eps = 1e-4);
    
private:
    cl_context context;
    cl_device_id device;
    cl_command_queue queue;
    cl_kernel kernel;
    cl_program program;
};

#endif
