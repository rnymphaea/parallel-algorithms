#ifndef GPU_MATRIX_OPERATIONS_HPP
#define GPU_MATRIX_OPERATIONS_HPP

#include <vector>
#include <string>
#include <CL/opencl.h>

class GPUMatrixOperations {
public:
    GPUMatrixOperations();
    ~GPUMatrixOperations();
    
    bool initialize();
    
    std::vector<float> multiplyMatrices(const std::vector<float>& A,
                                       const std::vector<float>& B,
                                       int M, int N, int K,
                                       size_t workgroup_size = 16);
    
    std::vector<float> multiplyMatricesBlocked(const std::vector<float>& A,
                                              const std::vector<float>& B,
                                              int M, int N, int K,
                                              size_t workgroup_size = 16);

private:
    cl::Context context_;
    cl::CommandQueue queue_;
    cl::Program program_;
    cl::Kernel kernel_simple_;
    cl::Kernel kernel_blocked_;
    bool initialized_;
    
    std::string readKernelFile(const std::string& filename);
    cl::Device getDefaultDevice();
};

#endif
