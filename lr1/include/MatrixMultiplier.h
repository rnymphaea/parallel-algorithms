#ifndef MATRIX_MULTIPLER_H 
#define MATRIX_MULTIPLER_H

#include "Matrix.h"
#include <thread>
#include <future>
#include <stdexcept>
#include <chrono>

class MatrixMultiplier {
public:
    static Matrix multiplySingleThread(const Matrix& A, const Matrix& B);
    static Matrix multiplyMultiThread(const Matrix& A, const Matrix& B, size_t numThreads);
    static Matrix multiplyAsync(const Matrix& A, const Matrix& B, size_t numTasks); 
    static bool areEqual(const Matrix& A, const Matrix& B, double eps = 1e-6);

    template <typename Func>
    static double measureTime(Func f) {
        auto start = std::chrono::high_resolution_clock::now();
        f();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        return elapsed.count();
    }
};

#endif //MATRIX_MULTIPLIER_H
