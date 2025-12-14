#include "../include/merge_sort.h"
#include "../include/gpu_helper.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

MergeSorter::MergeSorter() : 
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
    std::ifstream file("src/kernels/bitonic_sort.cl");
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open bitonic_sort.cl");
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
    kernel = clCreateKernel(program, "bitonic_sort", &err);
    CHECK_CL_ERROR(err, "Failed to create kernel");
}

MergeSorter::~MergeSorter() {
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
}

std::vector<float> MergeSorter::sort(const std::vector<float>& data, size_t local_size) {
    size_t n = data.size();
    
    // Round up to power of 2
    size_t size = 1;
    while (size < n) size <<= 1;
    
    std::vector<float> padded = data;
    padded.resize(size, std::numeric_limits<float>::max());
    
    cl_int err;
    
    // Create buffer
    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  padded.size() * sizeof(float), padded.data(), &err);
    CHECK_CL_ERROR(err, "Failed to create buffer");
    
    // Bitonic sort
    for (unsigned int stage = 2; stage <= size; stage <<= 1) {
        for (unsigned int pass = stage >> 1; pass > 0; pass >>= 1) {
            err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
            CHECK_CL_ERROR(err, "Failed to set kernel arg 0");
            err = clSetKernelArg(kernel, 1, sizeof(unsigned int), &size);
            CHECK_CL_ERROR(err, "Failed to set kernel arg 1");
            err = clSetKernelArg(kernel, 2, sizeof(unsigned int), &stage);
            CHECK_CL_ERROR(err, "Failed to set kernel arg 2");
            err = clSetKernelArg(kernel, 3, sizeof(unsigned int), &pass);
            CHECK_CL_ERROR(err, "Failed to set kernel arg 3");
            
            size_t global_size = size / 2;
            size_t local = (local_size > 0 && global_size >= local_size) ? local_size : 256;
            
            if (global_size % local != 0) {
                global_size = ((global_size + local - 1) / local) * local;
            }
            
            err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local, 0, NULL, NULL);
            CHECK_CL_ERROR(err, "Failed to execute kernel");
            clFinish(queue);
        }
    }
    
    // Read result
    std::vector<float> result(size);
    err = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, 
                             size * sizeof(float), result.data(), 0, NULL, NULL);
    CHECK_CL_ERROR(err, "Failed to read buffer");
    
    clReleaseMemObject(buffer);
    
    result.resize(n);
    return result;
}

std::vector<float> MergeSorter::sort_cpu(const std::vector<float>& data) {
    std::vector<float> result = data;
    std::sort(result.begin(), result.end());
    return result;
}

bool MergeSorter::verify(const std::vector<float>& data) {
    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i] < data[i-1]) {
            std::cout << "Sort error at " << i << ": " << data[i-1] << " > " << data[i] << std::endl;
            return false;
        }
    }
    return true;
}
