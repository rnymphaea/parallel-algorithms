#include "MatrixGpu.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <memory>
#include <chrono>

MatrixGpu::MatrixGpu() : 
    context_(nullptr), 
    queue_(nullptr), 
    program_(nullptr), 
    kernel_simple_(nullptr),
    kernel_blocked_(nullptr),
    device_(nullptr),
    initialized_(false) {
}

MatrixGpu::~MatrixGpu() {
    if (kernel_simple_) clReleaseKernel(kernel_simple_);
    if (kernel_blocked_) clReleaseKernel(kernel_blocked_);
    if (program_) clReleaseProgram(program_);
    if (queue_) clReleaseCommandQueue(queue_);
    if (context_) clReleaseContext(context_);
}

void MatrixGpu::checkError(cl_int err, const std::string& operation) {
    if (err != CL_SUCCESS) {
        throw std::runtime_error(operation + " failed with error: " + std::to_string(err));
    }
}

bool MatrixGpu::initialize() {
    try {
        device_ = getDefaultDevice();
        
        cl_int err;
        context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
        checkError(err, "clCreateContext");
        
        cl_command_queue_properties props = 0;
        queue_ = clCreateCommandQueue(context_, device_, props, &err);
        checkError(err, "clCreateCommandQueue");
        
        std::string kernelSource = readKernelFile("kernels/matrix_multiply.cl");
        const char* source = kernelSource.c_str();
        size_t sourceSize = kernelSource.size();
        
        program_ = clCreateProgramWithSource(context_, 1, &source, &sourceSize, &err);
        checkError(err, "clCreateProgramWithSource");
        
        err = clBuildProgram(program_, 1, &device_, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t logSize;
            clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
            std::vector<char> log(logSize);
            clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
            std::cerr << "OpenCL build failed:" << std::endl;
            std::cerr << log.data() << std::endl;
            return false;
        }
        
        kernel_simple_ = clCreateKernel(program_, "matrix_multiply_simple", &err);
        checkError(err, "clCreateKernel (simple)");
        
        kernel_blocked_ = clCreateKernel(program_, "matrix_04_multiply_via_local_memory", &err);
        checkError(err, "clCreateKernel (blocked)");
        
        initialized_ = true;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Initialization failed: " << e.what() << std::endl;
        return false;
    }
}

std::vector<float> MatrixGpu::multiply(
    const std::vector<float>& A,
    const std::vector<float>& B,
    int M, int N, int K,
    size_t workgroup_size) {
    
    if (!initialized_) {
        throw std::runtime_error("GPU operations not initialized");
    }
    
    if (A.size() != static_cast<size_t>(M * K) || 
        B.size() != static_cast<size_t>(K * N)) {
        throw std::invalid_argument("Invalid matrix dimensions");
    }
    
    try {
        cl_int err;
        
        cl_mem bufferA = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                       sizeof(float) * A.size(), const_cast<float*>(A.data()), &err);
        checkError(err, "clCreateBuffer A");
        
        cl_mem bufferB = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                       sizeof(float) * B.size(), const_cast<float*>(B.data()), &err);
        checkError(err, "clCreateBuffer B");
        
        cl_mem bufferC = clCreateBuffer(context_, CL_MEM_WRITE_ONLY, sizeof(float) * M * N, nullptr, &err);
        checkError(err, "clCreateBuffer C");
        
        err = clSetKernelArg(kernel_simple_, 0, sizeof(cl_mem), &bufferA);
        checkError(err, "clSetKernelArg 0");
        err = clSetKernelArg(kernel_simple_, 1, sizeof(cl_mem), &bufferB);
        checkError(err, "clSetKernelArg 1");
        err = clSetKernelArg(kernel_simple_, 2, sizeof(cl_mem), &bufferC);
        checkError(err, "clSetKernelArg 2");
        err = clSetKernelArg(kernel_simple_, 3, sizeof(cl_uint), &N);
        checkError(err, "clSetKernelArg 3");
        err = clSetKernelArg(kernel_simple_, 4, sizeof(cl_uint), &M);
        checkError(err, "clSetKernelArg 4");
        err = clSetKernelArg(kernel_simple_, 5, sizeof(cl_uint), &K);
        checkError(err, "clSetKernelArg 5");
        
        size_t global[2] = {static_cast<size_t>(N), static_cast<size_t>(M)};
        size_t local[2] = {1, 1};
        
        err = clEnqueueNDRangeKernel(queue_, kernel_simple_, 2, nullptr, global, local, 0, nullptr, nullptr);
        checkError(err, "clEnqueueNDRangeKernel");
        
        std::vector<float> C(M * N);
        err = clEnqueueReadBuffer(queue_, bufferC, CL_TRUE, 0, 
                                 sizeof(float) * C.size(), C.data(), 0, nullptr, nullptr);
        checkError(err, "clEnqueueReadBuffer");
        
        clReleaseMemObject(bufferA);
        clReleaseMemObject(bufferB);
        clReleaseMemObject(bufferC);
        
        return C;
        
    } catch (const std::exception& e) {
        std::cerr << "Matrix multiplication failed: " << e.what() << std::endl;
        throw;
    }
}

std::vector<float> MatrixGpu::multiplyBlocked(
    const std::vector<float>& A,
    const std::vector<float>& B,
    int M, int N, int K,
    size_t workgroup_size) {
    
    if (!initialized_) {
        throw std::runtime_error("GPU operations not initialized");
    }
    
    if (A.size() != static_cast<size_t>(M * K) || 
        B.size() != static_cast<size_t>(K * N)) {
        throw std::invalid_argument("Invalid matrix dimensions");
    }
    
    try {
        cl_int err;
        
        cl_mem bufferA = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                       sizeof(float) * A.size(), const_cast<float*>(A.data()), &err);
        checkError(err, "clCreateBuffer A");
        
        cl_mem bufferB = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                       sizeof(float) * B.size(), const_cast<float*>(B.data()), &err);
        checkError(err, "clCreateBuffer B");
        
        cl_mem bufferC = clCreateBuffer(context_, CL_MEM_WRITE_ONLY, sizeof(float) * M * N, nullptr, &err);
        checkError(err, "clCreateBuffer C");
        
        err = clSetKernelArg(kernel_blocked_, 0, sizeof(cl_mem), &bufferA);
        checkError(err, "clSetKernelArg 0");
        err = clSetKernelArg(kernel_blocked_, 1, sizeof(cl_mem), &bufferB);
        checkError(err, "clSetKernelArg 1");
        err = clSetKernelArg(kernel_blocked_, 2, sizeof(cl_mem), &bufferC);
        checkError(err, "clSetKernelArg 2");
        err = clSetKernelArg(kernel_blocked_, 3, sizeof(cl_uint), &N);
        checkError(err, "clSetKernelArg 3");
        err = clSetKernelArg(kernel_blocked_, 4, sizeof(cl_uint), &M);
        checkError(err, "clSetKernelArg 4");
        err = clSetKernelArg(kernel_blocked_, 5, sizeof(cl_uint), &K);
        checkError(err, "clSetKernelArg 5");
        
        size_t globalX = (N + workgroup_size - 1) / workgroup_size * workgroup_size;
        size_t globalY = (M + workgroup_size - 1) / workgroup_size * workgroup_size;
        
        size_t global[2] = {globalX, globalY};
        size_t local[2] = {workgroup_size, workgroup_size};
        
        size_t max_work_group_size;
        err = clGetKernelWorkGroupInfo(kernel_blocked_, device_, CL_KERNEL_WORK_GROUP_SIZE, 
                                      sizeof(size_t), &max_work_group_size, nullptr);
        if (err == CL_SUCCESS) {
            size_t required_work_group_size = workgroup_size * workgroup_size;
            if (required_work_group_size > max_work_group_size) {
                throw std::runtime_error("Work group size " + std::to_string(required_work_group_size) + 
                                       " exceeds maximum " + std::to_string(max_work_group_size));
            }
        }
        
        err = clEnqueueNDRangeKernel(queue_, kernel_blocked_, 2, nullptr, global, local, 0, nullptr, nullptr);
        checkError(err, "clEnqueueNDRangeKernel (blocked)");
        
        std::vector<float> C(M * N);
        err = clEnqueueReadBuffer(queue_, bufferC, CL_TRUE, 0, 
                                 sizeof(float) * C.size(), C.data(), 0, nullptr, nullptr);
        checkError(err, "clEnqueueReadBuffer");
        
        clReleaseMemObject(bufferA);
        clReleaseMemObject(bufferB);
        clReleaseMemObject(bufferC);
        
        return C;
        
    } catch (const std::exception& e) {
        std::cerr << "Blocked matrix multiplication failed: " << e.what() << std::endl;
        throw;
    }
}

std::string MatrixGpu::readKernelFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open kernel file: " + filename);
    }
    
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    return content;
}

cl_device_id MatrixGpu::getDefaultDevice() {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_int err;
    
    err = clGetPlatformIDs(1, &platform, nullptr);
    checkError(err, "clGetPlatformIDs");
    
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "GPU not found, trying CPU..." << std::endl;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
        checkError(err, "clGetDeviceIDs (CPU)");
    }
    
    return device;
}
