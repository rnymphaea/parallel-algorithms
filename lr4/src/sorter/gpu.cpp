#include "../include/sorter/gpu.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <chrono>  // Добавлен этот заголовок

SorterGPU::SorterGPU() : 
    context_(nullptr), 
    queue_(nullptr), 
    program_(nullptr), 
    merge_kernel_(nullptr),
    device_(nullptr) {
    
    cl_int err;
    cl_platform_id platform = nullptr;
    
    err = clGetPlatformIDs(1, &platform, nullptr);
    checkError(err, "clGetPlatformIDs");
    
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_, nullptr);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device_, nullptr);
        checkError(err, "clGetDeviceIDs (ALL)");
    }
    
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    checkError(err, "clCreateContext");
    
    // Используем совместимую функцию создания очереди
    cl_command_queue_properties props = 0;
    queue_ = clCreateCommandQueue(context_, device_, props, &err);
    checkError(err, "clCreateCommandQueue");
    
    std::string kernel_path = "kernels/merge_sort.cl";
    std::ifstream source_file(kernel_path);
    if (!source_file.is_open()) {
        kernel_path = "../kernels/merge_sort.cl";
        source_file.open(kernel_path);
        if (!source_file.is_open()) {
            throw std::runtime_error("Cannot open merge_sort.cl");
        }
    }
    
    std::string source_code(
        std::istreambuf_iterator<char>(source_file),
        (std::istreambuf_iterator<char>())
    );
    
    const char* source_str = source_code.c_str();
    size_t source_size = source_code.length();
    program_ = clCreateProgramWithSource(context_, 1, &source_str, &source_size, &err);
    checkError(err, "clCreateProgramWithSource");
    
    err = clBuildProgram(program_, 1, &device_, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Build error:\n" << log.data() << std::endl;
        throw std::runtime_error("OpenCL program build failed");
    }
    
    merge_kernel_ = clCreateKernel(program_, "merge_sort", &err);
    checkError(err, "clCreateKernel");
    
    std::cout << "GPU Sorter initialized successfully" << std::endl;
}

SorterGPU::~SorterGPU() {
    if (merge_kernel_) clReleaseKernel(merge_kernel_);
    if (program_) clReleaseProgram(program_);
    if (queue_) clReleaseCommandQueue(queue_);
    if (context_) clReleaseContext(context_);
}

void SorterGPU::checkError(cl_int err, const std::string& operation) {
    if (err != CL_SUCCESS) {
        throw std::runtime_error(operation + " failed with error: " + std::to_string(err));
    }
}

void SorterGPU::sort(std::vector<int>& array, const GPUConfig& config) {
    int size = static_cast<int>(array.size());
    if (size <= 1) return;
    
    int max_pow = 0;
    int temp = 1;
    while (temp < size) {
        temp <<= 1;
        max_pow++;
    }
    
    cl_int err;
    cl_mem buffer_a = clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeof(int) * size, nullptr, &err);
    checkError(err, "clCreateBuffer A");
    
    cl_mem buffer_b = clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeof(int) * size, nullptr, &err);
    checkError(err, "clCreateBuffer B");
    
    err = clEnqueueWriteBuffer(queue_, buffer_a, CL_TRUE, 0, sizeof(int) * size, array.data(), 0, nullptr, nullptr);
    checkError(err, "clEnqueueWriteBuffer");
    
    cl_mem src = buffer_a;
    cl_mem dst = buffer_b;
    
    size_t global_size = ((size + config.work_group_size - 1) / config.work_group_size) * config.work_group_size;
    size_t local_size = config.work_group_size;
    
    for (int pow_val = 0; pow_val < max_pow; pow_val++) {
        cl_uint pow_arg = static_cast<cl_uint>(pow_val);
        cl_uint size_arg = static_cast<cl_uint>(size);
        
        err = clSetKernelArg(merge_kernel_, 0, sizeof(cl_mem), &src);
        checkError(err, "clSetKernelArg 0");
        err = clSetKernelArg(merge_kernel_, 1, sizeof(cl_mem), &dst);
        checkError(err, "clSetKernelArg 1");
        err = clSetKernelArg(merge_kernel_, 2, sizeof(cl_uint), &size_arg);
        checkError(err, "clSetKernelArg 2");
        err = clSetKernelArg(merge_kernel_, 3, sizeof(cl_uint), &pow_arg);
        checkError(err, "clSetKernelArg 3");
        
        err = clEnqueueNDRangeKernel(queue_, merge_kernel_, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);
        checkError(err, "clEnqueueNDRangeKernel");
        
        err = clFinish(queue_);
        checkError(err, "clFinish");
        
        cl_mem tmp = src;
        src = dst;
        dst = tmp;
    }
    
    err = clEnqueueReadBuffer(queue_, src, CL_TRUE, 0, sizeof(int) * size, array.data(), 0, nullptr, nullptr);
    checkError(err, "clEnqueueReadBuffer");
    
    err = clFinish(queue_);
    checkError(err, "clFinish");
    
    clReleaseMemObject(buffer_a);
    clReleaseMemObject(buffer_b);
}

double SorterGPU::sortWithProfiling(std::vector<int>& array, const GPUConfig& config) {
    auto start_time = std::chrono::high_resolution_clock::now();
    sort(array, config);
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end_time - start_time).count();
}

size_t SorterGPU::getMaxWorkGroupSize() const {
    size_t max_size;
    cl_int err = clGetKernelWorkGroupInfo(merge_kernel_, device_, CL_KERNEL_WORK_GROUP_SIZE, 
                                         sizeof(size_t), &max_size, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error getting max work group size: " << err << std::endl;
        return 256;
    }
    return max_size;
}

size_t SorterGPU::getPreferredWorkGroupSize() const {
    size_t preferred_size;
    cl_int err = clGetKernelWorkGroupInfo(merge_kernel_, device_, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                                         sizeof(size_t), &preferred_size, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error getting preferred work group size: " << err << std::endl;
        return 64;
    }
    return preferred_size;
}

std::string SorterGPU::getDeviceInfo() const {
    std::string info;
    try {
        char device_name[256];
        cl_int err = clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
        if (err == CL_SUCCESS) {
            info += "Device: " + std::string(device_name) + "\n";
        }
        
        info += "Max work-group size: " + std::to_string(getMaxWorkGroupSize()) + "\n";
        info += "Preferred work-group multiple: " + std::to_string(getPreferredWorkGroupSize()) + "\n";
        
        cl_uint compute_units;
        err = clGetDeviceInfo(device_, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, nullptr);
        if (err == CL_SUCCESS) {
            info += "Compute units: " + std::to_string(compute_units) + "\n";
        }
        
        cl_ulong global_mem;
        err = clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, nullptr);
        if (err == CL_SUCCESS) {
            info += "Global memory: " + std::to_string(global_mem / (1024*1024)) + " MB";
        }
    } catch (...) {
        info = "Error getting device info";
    }
    return info;
}
