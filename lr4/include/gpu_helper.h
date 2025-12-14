#ifndef GPU_HELPER_H
#define GPU_HELPER_H

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <string>

// Определяем типы OpenCL, если заголовок не включен
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define CHECK_CL_ERROR(err, msg) \
    if (err != CL_SUCCESS) { \
        std::cerr << msg << " (Error: " << err << ")" << std::endl; \
        exit(1); \
    }

class Timer {
public:
    Timer() : start_time(), end_time() {}
    void start() { start_time = std::chrono::high_resolution_clock::now(); }
    void stop() { end_time = std::chrono::high_resolution_clock::now(); }
    double elapsed() const {
        return std::chrono::duration<double>(end_time - start_time).count();
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time, end_time;
};

template<typename T>
std::vector<T> random_vector(size_t n, T min = 0, T max = 100) {
    std::vector<T> v(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(min, max);
    for (size_t i = 0; i < n; ++i) v[i] = dist(gen);
    return v;
}

#endif
