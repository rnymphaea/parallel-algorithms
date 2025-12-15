#ifndef BLOCK_MULTIPLIER_HPP
#define BLOCK_MULTIPLIER_HPP

#include "Matrix.hpp"
#include <thread>
#include <vector>
#include <future>

class BlockMultiplier {
private:
    static const size_t BLOCK_SIZE = 64;
    size_t maxThreads;
    
    Matrix multiplyBasic(const Matrix& A, const Matrix& B) const;
    Matrix multiplyParallel(const Matrix& A, const Matrix& B) const;
    
public:
    BlockMultiplier(size_t threads = std::thread::hardware_concurrency());
    
    Matrix multiply(const Matrix& A, const Matrix& B);
    Matrix multiplySingleThread(const Matrix& A, const Matrix& B);
    
    static Matrix naiveMultiply(const Matrix& A, const Matrix& B);
};

#endif
