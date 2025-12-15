#include "../include/BlockMultiplier.hpp"
#include <iostream>
#include <future>

BlockMultiplier::BlockMultiplier(size_t threads) 
    : maxThreads(threads) {}

Matrix BlockMultiplier::multiplyBasic(const Matrix& A, const Matrix& B) const {
    size_t n = A.getRows();
    size_t m = A.getCols();
    size_t p = B.getCols();
    
    Matrix C(n, p);
    
    for (size_t i = 0; i < n; i += BLOCK_SIZE) {
        size_t i_end = std::min(i + BLOCK_SIZE, n);
        for (size_t j = 0; j < p; j += BLOCK_SIZE) {
            size_t j_end = std::min(j + BLOCK_SIZE, p);
            for (size_t k = 0; k < m; k += BLOCK_SIZE) {
                size_t k_end = std::min(k + BLOCK_SIZE, m);
                
                // Умножение блоков
                for (size_t ii = i; ii < i_end; ++ii) {
                    for (size_t kk = k; kk < k_end; ++kk) {
                        double aik = A(ii, kk);
                        for (size_t jj = j; jj < j_end; ++jj) {
                            C(ii, jj) += aik * B(kk, jj);
                        }
                    }
                }
            }
        }
    }
    
    return C;
}

Matrix BlockMultiplier::multiplyParallel(const Matrix& A, const Matrix& B) const {
    size_t n = A.getRows();
    size_t m = A.getCols();
    size_t p = B.getCols();
    
    Matrix C(n, p);
    
    auto multiplyBlock = [&](size_t i_start, size_t i_end) {
        for (size_t i = i_start; i < i_end; i += BLOCK_SIZE) {
            size_t i_end_block = std::min(i + BLOCK_SIZE, n);
            for (size_t j = 0; j < p; j += BLOCK_SIZE) {
                size_t j_end = std::min(j + BLOCK_SIZE, p);
                for (size_t k = 0; k < m; k += BLOCK_SIZE) {
                    size_t k_end = std::min(k + BLOCK_SIZE, m);
                    
                    for (size_t ii = i; ii < i_end_block; ++ii) {
                        for (size_t kk = k; kk < k_end; ++kk) {
                            double aik = A(ii, kk);
                            for (size_t jj = j; jj < j_end; ++jj) {
                                C(ii, jj) += aik * B(kk, jj);
                            }
                        }
                    }
                }
            }
        }
    };
    
    size_t numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t blocksPerThread = (numBlocks + maxThreads - 1) / maxThreads;
    
    std::vector<std::future<void>> futures;
    for (size_t t = 0; t < maxThreads; ++t) {
        size_t startBlock = t * blocksPerThread;
        size_t endBlock = std::min(startBlock + blocksPerThread, numBlocks);
        if (startBlock >= endBlock) break;
        
        size_t i_start = startBlock * BLOCK_SIZE;
        size_t i_end = std::min(endBlock * BLOCK_SIZE, n);
        
        futures.push_back(std::async(std::launch::async, multiplyBlock, i_start, i_end));
    }
    
    for (auto& fut : futures) {
        fut.get();
    }
    
    return C;
}

Matrix BlockMultiplier::multiply(const Matrix& A, const Matrix& B) {
    if (A.getCols() != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions don't match");
    }
    
    if (maxThreads > 1) {
        return multiplyParallel(A, B);
    } else {
        return multiplyBasic(A, B);
    }
}

Matrix BlockMultiplier::multiplySingleThread(const Matrix& A, const Matrix& B) {
    BlockMultiplier single(1);
    return single.multiply(A, B);
}

Matrix BlockMultiplier::naiveMultiply(const Matrix& A, const Matrix& B) {
    size_t n = A.getRows();
    size_t m = A.getCols();
    size_t p = B.getCols();
    
    if (m != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions don't match");
    }
    
    Matrix C(n, p);
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < p; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < m; ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
    
    return C;
}
