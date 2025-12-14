#ifndef CPU_MATRIX_OPERATIONS_HPP
#define CPU_MATRIX_OPERATIONS_HPP

#include <vector>
#include <thread>
#include <functional>
#include <algorithm>

class CPUMatrixOperations {
public:
    static std::vector<float> multiplyMatrices(const std::vector<float>& A,
                                              const std::vector<float>& B,
                                              int M, int N, int K,
                                              int num_threads = 1);
    
    static std::vector<float> multiplyMatricesBlocked(const std::vector<float>& A,
                                                     const std::vector<float>& B,
                                                     int M, int N, int K,
                                                     int block_size = 32,
                                                     int num_threads = 1);

private:
    static void multiplyBlock(const std::vector<float>& A,
                             const std::vector<float>& B,
                             std::vector<float>& C,
                             int M, int N, int K,
                             int start_row, int end_row,
                             int start_col, int end_col,
                             int block_size);
    
    static void threadedMultiply(const std::vector<float>& A,
                                const std::vector<float>& B,
                                std::vector<float>& C,
                                int M, int N, int K,
                                int start_row, int end_row);
    
    static void threadedMultiplyBlocked(const std::vector<float>& A,
                                       const std::vector<float>& B,
                                       std::vector<float>& C,
                                       int M, int N, int K,
                                       int block_size,
                                       int start_row, int end_row);
};

#endif