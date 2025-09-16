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

    auto worker = [&](size_t rowStart, size_t rowEnd) {
        for (size_t i = rowStart; i < rowEnd; i++) {
            for (size_t j = 0; j < B.numCols(); j++) {
                double sum = 0.0;
                for (size_t k = 0; k < A.numCols(); k++) {
                    sum += A(i, k) * B(k, j);
                }
                C(i, j) = sum;
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

    auto worker = [&](size_t rowStart, size_t rowEnd) -> Matrix {
        Matrix partial(rowEnd - rowStart, B.numCols());
        for (size_t i = rowStart; i < rowEnd; ++i) {
            for (size_t j = 0; j < B.numCols(); ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < A.numCols(); ++k) {
                    sum += A(i, k) * B(k, j);
                }
                partial(i - rowStart, j) = sum;
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

