#include "opencl_utils.hpp"
#include <fstream>
#include <sstream>
#include <cmath>

OpenCLContext::OpenCLContext() {
    cl_int err;
    
    // Получаем платформу
    err = clGetPlatformIDs(1, &platform_, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to get platform ID");
    }
    
    // Получаем устройство (GPU, если доступно, иначе CPU)
    err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 1, &device_, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Warning: GPU not found, trying CPU..." << std::endl;
        err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_CPU, 1, &device_, nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to get device ID");
        }
    }
    
    // Создаем контекст
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    checkError(err, "clCreateContext");
    
    // Создаем очередь команд
    queue_ = clCreateCommandQueue(context_, device_, 0, &err);
    checkError(err, "clCreateCommandQueue");
}

OpenCLContext::~OpenCLContext() {
    if (queue_) clReleaseCommandQueue(queue_);
    if (context_) clReleaseContext(context_);
}

cl_program OpenCLContext::createProgramFromSource(const std::string& source) {
    cl_int err;
    const char* sourceStr = source.c_str();
    size_t sourceSize = source.length();
    
    cl_program program = clCreateProgramWithSource(context_, 1, &sourceStr, &sourceSize, &err);
    checkError(err, "clCreateProgramWithSource");
    
    err = clBuildProgram(program, 1, &device_, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "Build log:\n" << log.data() << std::endl;
        clReleaseProgram(program);
        checkError(err, "clBuildProgram");
    }
    
    return program;
}

void OpenCLContext::checkError(cl_int err, const std::string& operation) {
    if (err != CL_SUCCESS) {
        throw std::runtime_error(operation + " failed with error: " + std::to_string(err));
    }
}

void OpenCLContext::printDeviceInfo() const {
    char deviceName[256];
    char deviceVendor[256];
    cl_device_type deviceType;
    cl_uint computeUnits;
    size_t maxWorkGroupSize;
    cl_uint maxWorkItemDimensions;
    size_t maxWorkItemSizes[3];
    cl_ulong globalMemSize;
    cl_ulong localMemSize;
    
    clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_VENDOR, sizeof(deviceVendor), deviceVendor, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(maxWorkItemDimensions), &maxWorkItemDimensions, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxWorkItemSizes), maxWorkItemSizes, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMemSize), &globalMemSize, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemSize), &localMemSize, nullptr);
    
    std::cout << "\n╔═══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║              ИНФОРМАЦИЯ ОБ УСТРОЙСТВЕ GPU                 ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << "Устройство: " << deviceName << std::endl;
    std::cout << "Производитель: " << deviceVendor << std::endl;
    std::cout << "Тип: " << (deviceType == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU") << std::endl;
    std::cout << "Compute Units: " << computeUnits << std::endl;
    std::cout << "Max Work Group Size: " << maxWorkGroupSize << std::endl;
    std::cout << "Max Work Item Sizes: [" << maxWorkItemSizes[0] << ", " 
              << maxWorkItemSizes[1] << ", " << maxWorkItemSizes[2] << "]" << std::endl;
    std::cout << "Global Memory: " << (globalMemSize / (1024*1024)) << " MB" << std::endl;
    std::cout << "Local Memory: " << (localMemSize / 1024) << " KB" << std::endl;
}

std::string readKernelFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open kernel file: " + filename);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}