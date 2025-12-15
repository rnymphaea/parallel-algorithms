#ifndef STRASSEN_MULTIPLIER_HPP
#define STRASSEN_MULTIPLIER_HPP

#include "Matrix.hpp"
#include <thread>
#include <vector>
#include <future>

class StrassenMultiplier {
private:
    static const size_t MIN_SIZE = 32;
    size_t maxThreads;
    size_t currentDepth;
    
    Matrix multiplyBasic(const Matrix& A, const Matrix& B) const;
    Matrix strassenRecursive(const Matrix& A, const Matrix& B, size_t depth);
    
public:
    StrassenMultiplier(size_t threads = std::thread::hardware_concurrency());
    
    Matrix multiply(const Matrix& A, const Matrix& B);
    Matrix multiplySingleThread(const Matrix& A, const Matrix& B);
    
    static Matrix naiveMultiply(const Matrix& A, const Matrix& B);
};

#endif
