#include "gpu_sorter.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>

GPUMergeSorter::GPUMergeSorter() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    
    if (platforms.empty()) {
        throw std::runtime_error("No OpenCL platforms found");
    }
    
    cl::Platform platform = platforms[0];
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    
    if (devices.empty()) {
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        if (devices.empty()) {
            throw std::runtime_error("No OpenCL devices found");
        }
    }
    
    device = devices[0];
    context = cl::Context(device);
    queue = cl::CommandQueue(context, device);
    
    // Загрузка и компиляция программы
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
    
    cl::Program::Sources sources;
    sources.push_back({source_code.c_str(), source_code.length()});
    program = cl::Program(context, sources);
    
    if (program.build({device})!= CL_SUCCESS) {
        std::string build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        std::cerr << "Build error:\n" << build_log << std::endl;
        throw std::runtime_error("OpenCL program build failed");
    }
    
    merge_kernel = cl::Kernel(program, "merge_sort");
    
    std::cout << "GPU Sorter initialized successfully" << std::endl;
}

void GPUMergeSorter::sort(std::vector<int>& array, const GPUConfig& config) {
    int size = array.size();
    if (size <= 1) return;
    
    // Вычисляем необходимое количество итераций
    int max_pow = 0;
    int temp = 1;
    while (temp < size) {
        temp <<= 1; // Удваиваем temp: 1→2→4→8→16...
        max_pow++; // Считаем итерации: 0,1,2,3...
    }
    
    //Создаем два буфера для алгоритма - читаем из одного, пишем в другой.
    cl::Buffer buffer_a(context, CL_MEM_READ_WRITE, sizeof(int) * size);
    cl::Buffer buffer_b(context, CL_MEM_READ_WRITE, sizeof(int) * size);
    
    //Передаем исходный массив из CPU памяти в GPU память (буфер A).
    queue.enqueueWriteBuffer(buffer_a, CL_TRUE, 0, sizeof(int) * size, array.data());
    
    //src - откуда читаем, dst - куда пишем. Будем менять местами каждую итерацию. 
    cl::Buffer* src = &buffer_a;
    cl::Buffer* dst = &buffer_b;
    
    // Оптимизированная версия: один work-item на элемент
    size_t global_size = ((size + config.work_group_size - 1) / config.work_group_size) * config.work_group_size;
    size_t local_size = config.work_group_size;
    
    //Каждая итерация соответствует определенному размеру блоков: (1,2,4...)
    for (int pow_val = 0; pow_val < max_pow; pow_val++) {
        merge_kernel.setArg(0, *src); // Исходный массив
        merge_kernel.setArg(1, *dst); // Целевой массив  
        merge_kernel.setArg(2, size); // Размер массива
        merge_kernel.setArg(3, pow_val); // Текущая "степень" (размер блока = 2^pow_val)
        
        queue.enqueueNDRangeKernel(
            merge_kernel,
            cl::NullRange, // Начальный offset = 0
            cl::NDRange(global_size),
            cl::NDRange(local_size)
        );
        
        queue.finish(); // Ждем завершения всех потоков
        std::swap(src, dst); // Меняем буферы местами
    }
    
    //Копируем отсортированный массив из GPU памяти обратно в CPU память.
    queue.enqueueReadBuffer(*src, CL_TRUE, 0, sizeof(int) * size, array.data());
    queue.finish();
}

double GPUMergeSorter::sort_with_profiling(std::vector<int>& array, const GPUConfig& config) {
    auto start_time = std::chrono::high_resolution_clock::now();
    sort(array, config);
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end_time - start_time).count();
}

size_t GPUMergeSorter::get_max_work_group_size() const {
    try {
        return merge_kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
    } catch (...) {
        std::cerr << "Error getting max work group size" << std::endl;
        return 256;
    }
}

size_t GPUMergeSorter::get_preferred_work_group_size() const {
    try {
        return merge_kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);
    } catch (...) {
        std::cerr << "Error getting preferred work group size" << std::endl;
        return 64;
    }
}

std::string GPUMergeSorter::get_device_info() const {
    std::string info;
    try {
        info = "Device: " + device.getInfo<CL_DEVICE_NAME>() + "\n";
        info += "Max work-group size: " + std::to_string(get_max_work_group_size()) + "\n";
        info += "Preferred work-group multiple: " + std::to_string(get_preferred_work_group_size()) + "\n";
        info += "Compute units: " + std::to_string(device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()) + "\n";
        info += "Global memory: " + std::to_string(device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024*1024)) + " MB";
    } catch (...) {
        info = "Error getting device info";
    }
    return info;
}