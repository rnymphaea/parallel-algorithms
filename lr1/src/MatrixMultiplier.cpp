#include "MatrixMultiplier.h"

Matrix MatrixMultiplier::multiplySingleThread(const Matrix& A, const Matrix& B) {
    if (A.numCols() != B.numRows()) {
        throw std::invalid_argument("error: invalid size of the matrices");
    }

    Matrix C(A.numRows(), B.numCols());

    for (size_t i = 0; i < A.numRows(); i++) {
        for (size_t j = 0; j < B.numCols(); j++) {
            double sum = 0.0;
            for (size_t k = 0; k < A.numCols(); k++) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
    return C;
}

Matrix MatrixMultiplier::multiplyMultiThread(const Matrix& A, const Matrix& B, size_t numThreads) {
    if (A.numCols() != B.numRows()) {
        throw std::invalid_argument("error: invalid size of the matrices");
    }

    Matrix C(A.numRows(), B.numCols());
    size_t block = this->blockSize; 

    auto worker = [&](size_t iStart, size_t iEnd) {
        size_t m = B.numCols();
        size_t kdim = A.numCols();

        for (size_t i0 = iStart; i0 < iEnd; i0 += block) {
            for (size_t j0 = 0; j0 < m; j0 += block) {
                for (size_t k0 = 0; k0 < kdim; k0 += block) {
                    size_t iMax = std::min(i0 + block, iEnd);
                    size_t jMax = std::min(j0 + block, m);
                    size_t kMax = std::min(k0 + block, kdim);

                    for (size_t i = i0; i < iMax; i++) {
                        for (size_t k = k0; k < kMax; k++) {
                            for (size_t j = j0; j < jMax; j++) {
                                C(i, j) += A(i, k) * B(k, j);
                            }
                        }
                    }
                }
            }
        }
    };

    std::vector<std::thread> threads;
    size_t rowsPerThread = A.numRows() / numThreads;
    size_t extra = A.numRows() % numThreads;

    size_t rowStart = 0;
    for (size_t t = 0; t < numThreads; t++) {
        size_t rowEnd = rowStart + rowsPerThread + (t < extra ? 1 : 0);
        threads.emplace_back(worker, rowStart, rowEnd);
        rowStart = rowEnd;
    }

    for (auto& th : threads) {
        th.join();
    }

    return C;
}

Matrix MatrixMultiplier::multiplyAsync(const Matrix& A, const Matrix& B, size_t numTasks) {
    if (A.numCols() != B.numRows()) {
        throw std::invalid_argument("error: invalid size of the matrices");
    }

    Matrix C(A.numRows(), B.numCols());
    size_t block = this->blockSize;

    auto worker = [&](size_t iStart, size_t iEnd) -> Matrix {
        Matrix partial(iEnd - iStart, B.numCols());

        size_t n = iEnd - iStart;
        size_t m = B.numCols();
        size_t kdim = A.numCols();

        for (size_t i0 = 0; i0 < n; i0 += block) {
            for (size_t j0 = 0; j0 < m; j0 += block) {
                for (size_t k0 = 0; k0 < kdim; k0 += block) {
                    size_t iMax = std::min(i0 + block, n);
                    size_t jMax = std::min(j0 + block, m);
                    size_t kMax = std::min(k0 + block, kdim);

                    for (size_t i = i0; i < iMax; i++) {
                        for (size_t k = k0; k < kMax; k++) {
                            for (size_t j = j0; j < jMax; j++) {
                                partial(i, j) += A(iStart + i, k) * B(k, j);
                            }
                        }
                    }
                }
            }
        }

        return partial;
    };

    size_t rowsPerTask = A.numRows() / numTasks;
    size_t extra = A.numRows() % numTasks;

    std::vector<std::future<Matrix>> futures;
    size_t rowStart = 0;

    for (size_t t = 0; t < numTasks; ++t) {
        size_t rowEnd = rowStart + rowsPerTask + (t < extra ? 1 : 0);
        futures.push_back(std::async(std::launch::async, worker, rowStart, rowEnd));
        rowStart = rowEnd;
    }

    rowStart = 0;
    for (auto& fut : futures) {
        Matrix partial = fut.get();
        for (size_t i = 0; i < partial.numRows(); ++i) {
            for (size_t j = 0; j < partial.numCols(); ++j) {
                C(rowStart + i, j) = partial(i, j);
            }
        }
        rowStart += partial.numRows();
    }

    return C;
}

bool MatrixMultiplier::areEqual(const Matrix& A, const Matrix& B, double eps) {
    if (A.numRows() != B.numRows() || A.numCols() != B.numCols()) return false;

    for (size_t i = 0; i < A.numRows(); i++) {
        for (size_t j = 0; j < A.numCols(); j++) {
            if (std::abs(A(i, j) - B(i, j)) > eps) return false;
        }
    }
    return true;
}

