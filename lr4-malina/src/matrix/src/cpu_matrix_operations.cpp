#include "../include/cpu_matrix_operations.hpp"
#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>

std::vector<float> CPUMatrixOperations::multiplyMatrices(
    const std::vector<float>& A,
    const std::vector<float>& B,
    int M, int N, int K,
    int num_threads) {
    
    std::vector<float> C(M * N, 0.0f);
    
    if (num_threads == 1) {
        // Sequential version
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    } else {
        // Threaded version
        std::vector<std::thread> threads;
        int rows_per_thread = M / num_threads;
        
        for (int t = 0; t < num_threads; ++t) {
            int start_row = t * rows_per_thread;
            int end_row = (t == num_threads - 1) ? M : (t + 1) * rows_per_thread;
            
            threads.emplace_back(threadedMultiply, 
                                std::cref(A), std::cref(B), std::ref(C),
                                M, N, K, start_row, end_row);
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    }
    
    return C;
}

void CPUMatrixOperations::threadedMultiply(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M, int N, int K,
    int start_row, int end_row) {
    
    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

std::vector<float> CPUMatrixOperations::multiplyMatricesBlocked(
    const std::vector<float>& A,
    const std::vector<float>& B,
    int M, int N, int K,
    int block_size,
    int num_threads) {
    
    std::vector<float> C(M * N, 0.0f);
    
    if (num_threads == 1) {
        // Single-threaded blocked version
        for (int i = 0; i < M; i += block_size) {
            for (int j = 0; j < N; j += block_size) {
                for (int p = 0; p < K; p += block_size) {
                    // Process block
                    int i_end = std::min(i + block_size, M);
                    int j_end = std::min(j + block_size, N);
                    int p_end = std::min(p + block_size, K);
                    
                    for (int ii = i; ii < i_end; ++ii) {
                        for (int jj = j; jj < j_end; ++jj) {
                            float sum = C[ii * N + jj];
                            for (int pp = p; pp < p_end; ++pp) {
                                sum += A[ii * K + pp] * B[pp * N + jj];
                            }
                            C[ii * N + jj] = sum;
                        }
                    }
                }
            }
        }
    } else {
        // Multi-threaded blocked version
        std::vector<std::thread> threads;
        int rows_per_thread = (M + num_threads - 1) / num_threads;
        
        for (int t = 0; t < num_threads; ++t) {
            int start_row = t * rows_per_thread;
            int end_row = std::min((t + 1) * rows_per_thread, M);
            
            if (start_row < M) {
                threads.emplace_back(threadedMultiplyBlocked,
                                    std::cref(A), std::cref(B), std::ref(C),
                                    M, N, K, block_size, start_row, end_row);
            }
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    }
    
    return C;
}

void CPUMatrixOperations::threadedMultiplyBlocked(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M, int N, int K,
    int block_size,
    int start_row, int end_row) {
    
    for (int i = start_row; i < end_row; i += block_size) {
        for (int j = 0; j < N; j += block_size) {
            for (int p = 0; p < K; p += block_size) {
                // Process block
                int i_end = std::min(i + block_size, end_row);
                int j_end = std::min(j + block_size, N);
                int p_end = std::min(p + block_size, K);
                
                for (int ii = i; ii < i_end; ++ii) {
                    for (int jj = j; jj < j_end; ++jj) {
                        float sum = C[ii * N + jj];
                        for (int pp = p; pp < p_end; ++pp) {
                            sum += A[ii * K + pp] * B[pp * N + jj];
                        }
                        C[ii * N + jj] = sum;
                    }
                }
            }
        }
    }
}

void CPUMatrixOperations::multiplyBlock(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M, int N, int K,
    int start_row, int end_row,
    int start_col, int end_col,
    int block_size) {
    
    for (int i = start_row; i < end_row; i += block_size) {
        for (int j = start_col; j < end_col; j += block_size) {
            for (int p = 0; p < K; p += block_size) {
                // Process block
                int i_end = std::min(i + block_size, end_row);
                int j_end = std::min(j + block_size, end_col);
                int p_end = std::min(p + block_size, K);
                
                for (int ii = i; ii < i_end; ++ii) {
                    for (int jj = j; jj < j_end; ++jj) {
                        float sum = C[ii * N + jj];
                        for (int pp = p; pp < p_end; ++pp) {
                            sum += A[ii * K + pp] * B[pp * N + jj];
                        }
                        C[ii * N + jj] = sum;
                    }
                }
            }
        }
    }
}