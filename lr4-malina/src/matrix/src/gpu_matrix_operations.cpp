#include "../include/gpu_matrix_operations.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>

GPUMatrixOperations::GPUMatrixOperations() : initialized_(false) {
}

GPUMatrixOperations::~GPUMatrixOperations() {
}

bool GPUMatrixOperations::initialize() {
    try {
        // Получаем устройство по умолчанию
        cl::Device device = getDefaultDevice();
        
        // Создаем контекст и очередь команд
        context_ = cl::Context(device);
        queue_ = cl::CommandQueue(context_, device);
        
        // Загружаем и компилируем программу
        std::string kernelSource = readKernelFile("kernels/matrix_multiply.cl");
        cl::Program::Sources sources;
        sources.push_back({kernelSource.c_str(), kernelSource.length()});
        
        program_ = cl::Program(context_, sources);
        
        // Пытаемся скомпилировать программу
        cl_int buildResult = program_.build({device});
        if (buildResult != CL_SUCCESS) {
            std::string buildLog;
            program_.getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &buildLog);
            std::cerr << "OpenCL build failed: " << buildLog << std::endl;
            return false;
        }
        
        // Создаем кернелы
        kernel_simple_ = cl::Kernel(program_, "matrix_multiply_simple");
        kernel_blocked_ = cl::Kernel(program_, "matrix_04_multiply_via_local_memory");
        
        initialized_ = true;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Initialization failed: " << e.what() << std::endl;
        return false;
    }
}

std::vector<float> GPUMatrixOperations::multiplyMatrices(
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
        // Создаем буферы в GPU памяти
        cl::Buffer bufferA(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                          sizeof(float) * A.size(), const_cast<float*>(A.data()));
        cl::Buffer bufferB(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                          sizeof(float) * B.size(), const_cast<float*>(B.data()));
        cl::Buffer bufferC(context_, CL_MEM_WRITE_ONLY, sizeof(float) * M * N);
        
        // Устанавливаем аргументы кернела
        kernel_simple_.setArg(0, bufferA);
        kernel_simple_.setArg(1, bufferB);
        kernel_simple_.setArg(2, bufferC);
        kernel_simple_.setArg(3, static_cast<cl_uint>(N));
        kernel_simple_.setArg(4, static_cast<cl_uint>(M));
        kernel_simple_.setArg(5, static_cast<cl_uint>(K));
        
        // Запускаем кернел
        cl::NDRange global(N, M); // cols, rows
        cl::NDRange local(1, 1);
        
        cl_int enqueueResult = queue_.enqueueNDRangeKernel(
            kernel_simple_, cl::NullRange, global, local);
        
        if (enqueueResult != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue kernel");
        }
        
        // Читаем результат
        std::vector<float> C(M * N);
        cl_int readResult = queue_.enqueueReadBuffer(bufferC, CL_TRUE, 0, 
                                                    sizeof(float) * C.size(), C.data());
        
        if (readResult != CL_SUCCESS) {
            throw std::runtime_error("Failed to read result buffer");
        }
        
        return C;
        
    } catch (const std::exception& e) {
        std::cerr << "Matrix multiplication failed: " << e.what() << std::endl;
        throw;
    }
}

std::vector<float> GPUMatrixOperations::multiplyMatricesBlocked(
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
    
    // Для блочного кернела требуются квадратные рабочие группы
    size_t group_size_x = workgroup_size;
    size_t group_size_y = workgroup_size;
    
    try {
        // Создаем буферы в GPU памяти
        cl::Buffer bufferA(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                          sizeof(float) * A.size(), const_cast<float*>(A.data()));
        cl::Buffer bufferB(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                          sizeof(float) * B.size(), const_cast<float*>(B.data()));
        cl::Buffer bufferC(context_, CL_MEM_WRITE_ONLY, sizeof(float) * M * N);
        
        // Устанавливаем аргументы кернела
        kernel_blocked_.setArg(0, bufferA);
        kernel_blocked_.setArg(1, bufferB);
        kernel_blocked_.setArg(2, bufferC);
        kernel_blocked_.setArg(3, static_cast<cl_uint>(N));  // w - ширина результата
        kernel_blocked_.setArg(4, static_cast<cl_uint>(M));  // h - высота результата  
        kernel_blocked_.setArg(5, static_cast<cl_uint>(K));  // k - внутренняя размерность
        // Вычисляем размеры для NDRange - выравниваем по границам рабочей группы
        size_t globalX = (N + group_size_x - 1) / group_size_x * group_size_x;
        size_t globalY = (M + group_size_y - 1) / group_size_y * group_size_y;

        //globalX - общее количество потоков по оси X, округленное вверх до кратного group_size_x
        //globalY - общее количество потоков по оси Y, округленное вверх до кратного group_size_y
        
        cl::NDRange global(globalX, globalY);
        cl::NDRange local(group_size_x, group_size_y);
        
        // Проверяем, поддерживается ли такой размер рабочей группы
        size_t max_work_group_size;
        cl_int infoResult = kernel_blocked_.getWorkGroupInfo<size_t>(
            cl::Device::getDefault(), CL_KERNEL_WORK_GROUP_SIZE, &max_work_group_size);
        
        if (infoResult == CL_SUCCESS) {
            size_t required_work_group_size = group_size_x * group_size_y;
            if (required_work_group_size > max_work_group_size) {
                throw std::runtime_error("Work group size " + std::to_string(required_work_group_size) + 
                                       " exceeds maximum " + std::to_string(max_work_group_size));
            }
        }
        
        // Запускаем кернел
        //kernel_blocked_ - кернел для выполнения
        //cl::NullRange - смещение (0,0)
        //global - общее количество потоков
        //local - размер рабочей группы
        cl_int enqueueResult = queue_.enqueueNDRangeKernel(
            kernel_blocked_, cl::NullRange, global, local);
        
        if (enqueueResult != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue blocked kernel (error: " + 
                                   std::to_string(enqueueResult) + ")");
        }
        
        // Читаем результат
        std::vector<float> C(M * N);
        //Копируем данные из буфера GPU bufferC в вектор C, CL_TRUE - блокирующее чтение (ждем завершения операции)
        cl_int readResult = queue_.enqueueReadBuffer(bufferC, CL_TRUE, 0, 
                                                    sizeof(float) * C.size(), C.data());
        
        if (readResult != CL_SUCCESS) {
            throw std::runtime_error("Failed to read result buffer");
        }
        
        return C;
        
    } catch (const std::exception& e) {
        std::cerr << "Blocked matrix multiplication failed: " << e.what() << std::endl;
        throw;
    }
}

std::string GPUMatrixOperations::readKernelFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open kernel file: " + filename);
    }
    
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    return content;
}

cl::Device GPUMatrixOperations::getDefaultDevice() {
    // Получаем все доступные платформы
    std::vector<cl::Platform> platforms;
    cl_int platformResult = cl::Platform::get(&platforms);
    
    if (platformResult != CL_SUCCESS || platforms.empty()) {
        throw std::runtime_error("No OpenCL platforms found");
    }
    
    // Ищем GPU устройство
    for (auto& platform : platforms) {
        std::vector<cl::Device> devices;
        cl_int deviceResult = platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        
        if (deviceResult == CL_SUCCESS && !devices.empty()) {
            return devices[0];
        }
    }
    
    // Если GPU не найден, ищем CPU
    for (auto& platform : platforms) {
        std::vector<cl::Device> devices;
        cl_int deviceResult = platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
        
        if (deviceResult == CL_SUCCESS && !devices.empty()) {
            std::cout << "Using CPU device (GPU not found)" << std::endl;
            return devices[0];
        }
    }
    
    throw std::runtime_error("No OpenCL devices found");
}