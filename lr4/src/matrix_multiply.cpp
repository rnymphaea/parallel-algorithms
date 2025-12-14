#include "../include/matrix_multiply.h"
#include "../include/gpu_helper.h"
#include <fstream>
#include <sstream>
#include <cstring>

MatrixMultiplier::MatrixMultiplier() : 
    context(nullptr), device(nullptr), queue(nullptr), 
    kernel(nullptr), program(nullptr) {
    
    cl_int err;
    
    // Get platform
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_CL_ERROR(err, "Failed to get platform");
    
    // Get device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
    CHECK_CL_ERROR(err, "Failed to get device");
    
    // Create context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL_ERROR(err, "Failed to create context");
    
    // Create command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_CL_ERROR(err, "Failed to create command queue");
    
    // Load kernel source
    std::ifstream file("src/kernels/matrix_mul.cl");
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open matrix_mul.cl");
    }
    
    std::stringstream source_stream;
    source_stream << file.rdbuf();
    std::string source_str = source_stream.str();
    const char* source = source_str.c_str();
    size_t source_len = source_str.length();
    
    // Create program
    program = clCreateProgramWithSource(context, 1, &source, &source_len, &err);
    CHECK_CL_ERROR(err, "Failed to create program");
    
    // Build program
    err = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        std::cerr << "Build error:\n" << log.data() << std::endl;
        CHECK_CL_ERROR(err, "Failed to build program");
    }
    
    // Create kernel
    kernel = clCreateKernel(program, "matrix_mul", &err);
    CHECK_CL_ERROR(err, "Failed to create kernel");
}

MatrixMultiplier::~MatrixMultiplier() {
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
}

std::vector<float> MatrixMultiplier::multiply(const std::vector<float>& A, const std::vector<float>& B,
                                            size_t rowsA, size_t colsA, size_t colsB,
                                            size_t local_size) {
    size_t M = rowsA;
    size_t N = colsA;
    size_t K = colsB;
    
    std::vector<float> C(M * K, 0.0f);
    cl_int err;
    
    // Create buffers
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                    A.size() * sizeof(float), (void*)A.data(), &err);
    CHECK_CL_ERROR(err, "Failed to create buffer A");
    
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    B.size() * sizeof(float), (void*)B.data(), &err);
    CHECK_CL_ERROR(err, "Failed to create buffer B");
    
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    C.size() * sizeof(float), NULL, &err);
    CHECK_CL_ERROR(err, "Failed to create buffer C");
    
    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    CHECK_CL_ERROR(err, "Failed to set kernel arg 0");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    CHECK_CL_ERROR(err, "Failed to set kernel arg 1");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    CHECK_CL_ERROR(err, "Failed to set kernel arg 2");
    err = clSetKernelArg(kernel, 3, sizeof(int), &M);
    CHECK_CL_ERROR(err, "Failed to set kernel arg 3");
    err = clSetKernelArg(kernel, 4, sizeof(int), &N);
    CHECK_CL_ERROR(err, "Failed to set kernel arg 4");
    err = clSetKernelArg(kernel, 5, sizeof(int), &K);
    CHECK_CL_ERROR(err, "Failed to set kernel arg 5");
    
    int block_size = 16;
    err = clSetKernelArg(kernel, 6, sizeof(int), &block_size);
    CHECK_CL_ERROR(err, "Failed to set kernel arg 6");
    
    // Execute kernel
    size_t global_work_size[2] = {M, K};
    size_t local_work_size[2] = {local_size > 0 ? local_size : 16, 
                                 local_size > 0 ? local_size : 16};
    
    if (local_size > 0 && M % local_size == 0 && K % local_size == 0) {
        err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    } else {
        err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    }
    CHECK_CL_ERROR(err, "Failed to execute kernel");
    
    // Read result
    err = clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, 
                             C.size() * sizeof(float), C.data(), 0, NULL, NULL);
    CHECK_CL_ERROR(err, "Failed to read buffer");
    
    // Cleanup
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    
    return C;
}

std::vector<float> MatrixMultiplier::multiply_cpu_simple(const std::vector<float>& A, const std::vector<float>& B,
                                                       size_t rowsA, size_t colsA, size_t colsB) {
    std::vector<float> C(rowsA * colsB, 0.0f);
    
    for (size_t i = 0; i < rowsA; ++i) {
        for (size_t j = 0; j < colsB; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < colsA; ++k) {
                sum += A[i * colsA + k] * B[k * colsB + j];
            }
            C[i * colsB + j] = sum;
        }
    }
    
    return C;
}

std::vector<float> MatrixMultiplier::multiply_cpu_blocked(const std::vector<float>& A, const std::vector<float>& B,
                                                        size_t rowsA, size_t colsA, size_t colsB,
                                                        size_t block_size) {
    std::vector<float> C(rowsA * colsB, 0.0f);
    
    for (size_t i = 0; i < rowsA; i += block_size) {
        for (size_t j = 0; j < colsB; j += block_size) {
            for (size_t k = 0; k < colsA; k += block_size) {
                size_t i_end = std::min(i + block_size, rowsA);
                size_t j_end = std::min(j + block_size, colsB);
                size_t k_end = std::min(k + block_size, colsA);
                
                for (size_t ii = i; ii < i_end; ++ii) {
                    for (size_t kk = k; kk < k_end; ++kk) {
                        float a = A[ii * colsA + kk];
                        for (size_t jj = j; jj < j_end; ++jj) {
                            C[ii * colsB + jj] += a * B[kk * colsB + jj];
                        }
                    }
                }
            }
        }
    }
    
    return C;
}

bool MatrixMultiplier::verify(const std::vector<float>& C1, const std::vector<float>& C2, float eps) {
    if (C1.size() != C2.size()) {
        std::cout << "Size mismatch: " << C1.size() << " vs " << C2.size() << std::endl;
        return false;
    }
    for (size_t i = 0; i < C1.size(); ++i) {
        if (std::abs(C1[i] - C2[i]) > eps) {
            std::cout << "Mismatch at " << i << ": " << C1[i] << " vs " << C2[i] << std::endl;
            return false;
        }
    }
    return true;
}
