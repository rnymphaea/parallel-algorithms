#ifndef MATRIX_CPU_HPP
#define MATRIX_CPU_HPP

#include <vector>
#include <thread>
#include <functional>
#include <algorithm>

class MatrixCPU {
public:
    static std::vector<float> multiply(const std::vector<float>& A,
                                       const std::vector<float>& B,
                                       int M, int N, int K,
                                       int num_threads = 1);
    
    static std::vector<float> multiplyBlocked(const std::vector<float>& A,
                                              const std::vector<float>& B,
                                              int M, int N, int K,
                                              int block_size = 32,
                                              int num_threads = 1);

private:
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
