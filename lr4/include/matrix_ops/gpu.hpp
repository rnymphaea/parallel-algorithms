#ifndef MATRIX_GPU_HPP
#define MATRIX_GPU_HPP

#include <vector>
#include <string>
#include <CL/opencl.h>
#include <memory>

class MatrixGPU {
public:
    MatrixGPU();
    ~MatrixGPU();
    
    bool initialize();
    
    std::vector<float> multiply(const std::vector<float>& A,
                                const std::vector<float>& B,
                                int M, int N, int K,
                                size_t workgroup_size = 16);
    
    std::vector<float> multiplyBlocked(const std::vector<float>& A,
                                       const std::vector<float>& B,
                                       int M, int N, int K,
                                       size_t workgroup_size = 16);

private:
    cl_context context_;
    cl_command_queue queue_;
    cl_program program_;
    cl_kernel kernel_simple_;
    cl_kernel kernel_blocked_;
    cl_device_id device_;
    bool initialized_;
    
    std::string readKernelFile(const std::string& filename);
    cl_device_id getDefaultDevice();
    void checkError(cl_int err, const std::string& operation);
};

#endif
