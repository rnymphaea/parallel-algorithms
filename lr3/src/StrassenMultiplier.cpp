#include "../include/StrassenMultiplier.hpp"
#include <iostream>
#include <future>

StrassenMultiplier::StrassenMultiplier(size_t threads) 
    : maxThreads(threads), currentDepth(0) {}

Matrix StrassenMultiplier::multiplyBasic(const Matrix& A, const Matrix& B) const {
    size_t n = A.getRows();
    Matrix C(n, n);
    
    // Простое умножение
    for(size_t i = 0; i < n; ++i) {
        for(size_t k = 0; k < n; ++k) {
            double aik = A(i, k);
            for(size_t j = 0; j < n; ++j) {
                C(i, j) += aik * B(k, j);
            }
        }
    }
    
    return C;
}

Matrix StrassenMultiplier::strassenRecursive(const Matrix& A, const Matrix& B, size_t depth) {
    size_t n = A.getRows();
    
    // Базовый случай
    if(n <= MIN_SIZE) {
        return multiplyBasic(A, B);
    }
    
    size_t half = n / 2;
    
    // Разделяем матрицы на подматрицы
    auto A11 = A.getSubmatrix(0, half, 0, half);
    auto A12 = A.getSubmatrix(0, half, half, n);
    auto A21 = A.getSubmatrix(half, n, 0, half);
    auto A22 = A.getSubmatrix(half, n, half, n);
    
    auto B11 = B.getSubmatrix(0, half, 0, half);
    auto B12 = B.getSubmatrix(0, half, half, n);
    auto B21 = B.getSubmatrix(half, n, 0, half);
    auto B22 = B.getSubmatrix(half, n, half, n);
    
    // Вычисляем промежуточные матрицы M1-M7
    Matrix M1, M2, M3, M4, M5, M6, M7;
    
    // Параллелизуем только первые уровни рекурсии
    if(depth < 2 && maxThreads > 1) {
        std::vector<std::future<Matrix>> futures;
        
        // M1 = (A11 + A22) * (B11 + B22)
        futures.push_back(std::async(std::launch::async, 
            [this, &A11, &A22, &B11, &B22, depth]() {
                return strassenRecursive(A11 + A22, B11 + B22, depth + 1);
            }));
        
        // M2 = (A21 + A22) * B11
        futures.push_back(std::async(std::launch::async,
            [this, &A21, &A22, &B11, depth]() {
                return strassenRecursive(A21 + A22, B11, depth + 1);
            }));
        
        // M3 = A11 * (B12 - B22)
        futures.push_back(std::async(std::launch::async,
            [this, &A11, &B12, &B22, depth]() {
                return strassenRecursive(A11, B12 - B22, depth + 1);
            }));
        
        // Остальные вычисляем в текущем потоке
        M4 = strassenRecursive(A22, B21 - B11, depth + 1);
        M5 = strassenRecursive(A11 + A12, B22, depth + 1);
        M6 = strassenRecursive(A21 - A11, B11 + B12, depth + 1);
        M7 = strassenRecursive(A12 - A22, B21 + B22, depth + 1);
        
        // Получаем результаты из асинхронных задач
        M1 = futures[0].get();
        M2 = futures[1].get();
        M3 = futures[2].get();
        
    } else {
        // Последовательное выполнение
        M1 = strassenRecursive(A11 + A22, B11 + B22, depth + 1);
        M2 = strassenRecursive(A21 + A22, B11, depth + 1);
        M3 = strassenRecursive(A11, B12 - B22, depth + 1);
        M4 = strassenRecursive(A22, B21 - B11, depth + 1);
        M5 = strassenRecursive(A11 + A12, B22, depth + 1);
        M6 = strassenRecursive(A21 - A11, B11 + B12, depth + 1);
        M7 = strassenRecursive(A12 - A22, B21 + B22, depth + 1);
    }
    
    // Вычисляем итоговые подматрицы
    auto C11 = M1 + M4 - M5 + M7;
    auto C12 = M3 + M5;
    auto C21 = M2 + M4;
    auto C22 = M1 - M2 + M3 + M6;
    
    return Matrix::combine(C11, C12, C21, C22);
}

Matrix StrassenMultiplier::multiply(const Matrix& A, const Matrix& B) {
    if(A.getCols() != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions don't match");
    }
    
    size_t n = A.getRows();
    size_t m = A.getCols();
    
    // Для неквадратных или маленьких матриц используем базовое умножение
    if(n != m || n <= MIN_SIZE || n != B.getCols()) {
        return multiplyBasic(A, B);
    }
    
    // Проверяем, является ли размер степенью двойки
    if((n & (n - 1)) != 0) {
        // Ищем ближайшую степень двойки
        size_t newSize = 1;
        while(newSize < n) newSize <<= 1;
        
        // Создаем матрицы с дополнением нулями
        Matrix A_padded(newSize, newSize);
        Matrix B_padded(newSize, newSize);
        
        for(size_t i = 0; i < n; ++i) {
            for(size_t j = 0; j < m; ++j) {
                A_padded(i, j) = A(i, j);
            }
        }
        
        for(size_t i = 0; i < m; ++i) {
            for(size_t j = 0; j < n; ++j) {
                B_padded(i, j) = B(i, j);
            }
        }
        
        // Умножаем дополненные матрицы
        Matrix C_padded = strassenRecursive(A_padded, B_padded, 0);
        
        // Извлекаем результат исходного размера
        Matrix C(n, n);
        for(size_t i = 0; i < n; ++i) {
            for(size_t j = 0; j < n; ++j) {
                C(i, j) = C_padded(i, j);
            }
        }
        
        return C;
    }
    
    return strassenRecursive(A, B, 0);
}

Matrix StrassenMultiplier::multiplySingleThread(const Matrix& A, const Matrix& B) {
    StrassenMultiplier singleThreadMultiplier(1);
    return singleThreadMultiplier.multiply(A, B);
}

Matrix StrassenMultiplier::naiveMultiply(const Matrix& A, const Matrix& B) {
    size_t n = A.getRows();
    size_t m = A.getCols();
    size_t p = B.getCols();
    
    if(m != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions don't match");
    }
    
    Matrix C(n, p);
    
    for(size_t i = 0; i < n; ++i) {
        for(size_t j = 0; j < p; ++j) {
            double sum = 0.0;
            for(size_t k = 0; k < m; ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
    
    return C;
}
