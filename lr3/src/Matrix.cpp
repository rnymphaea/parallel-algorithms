#include "../include/Matrix.hpp"
#include <iomanip>

Matrix::Matrix(size_t r, size_t c) : rows(r), cols(c) {
    data.resize(r, std::vector<double>(c, 0.0));
}

Matrix::Matrix(const std::vector<std::vector<double>>& d) 
    : data(d), rows(d.size()), cols(d.empty() ? 0 : d[0].size()) {}

double& Matrix::operator()(size_t i, size_t j) {
    return data[i][j];
}

const double& Matrix::operator()(size_t i, size_t j) const {
    return data[i][j];
}

Matrix Matrix::operator+(const Matrix& other) const {
    assert(rows == other.rows && cols == other.cols);
    Matrix result(rows, cols);
    for(size_t i = 0; i < rows; ++i) {
        for(size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] + other(i, j);
        }
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    assert(rows == other.rows && cols == other.cols);
    Matrix result(rows, cols);
    for(size_t i = 0; i < rows; ++i) {
        for(size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] - other(i, j);
        }
    }
    return result;
}

void Matrix::fillRandom() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 100.0);
    
    for(auto& row : data) {
        for(auto& val : row) {
            val = dis(gen);
        }
    }
}

void Matrix::fillRandomInt(int minVal, int maxVal) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(minVal, maxVal);
    
    for(auto& row : data) {
        for(auto& val : row) {
            val = dis(gen);
        }
    }
}

bool Matrix::operator==(const Matrix& other) const {
    if(rows != other.rows || cols != other.cols) return false;
    
    const double eps = 1e-6;
    for(size_t i = 0; i < rows; ++i) {
        for(size_t j = 0; j < cols; ++j) {
            if(std::abs(data[i][j] - other(i, j)) > eps) {
                return false;
            }
        }
    }
    return true;
}

bool Matrix::operator!=(const Matrix& other) const {
    return !(*this == other);
}

Matrix Matrix::getSubmatrix(size_t r1, size_t r2, size_t c1, size_t c2) const {
    size_t subRows = r2 - r1;
    size_t subCols = c2 - c1;
    Matrix result(subRows, subCols);
    
    for(size_t i = 0; i < subRows; ++i) {
        for(size_t j = 0; j < subCols; ++j) {
            result(i, j) = data[r1 + i][c1 + j];
        }
    }
    return result;
}

void Matrix::setSubmatrix(size_t r1, size_t c1, const Matrix& sub) {
    for(size_t i = 0; i < sub.getRows(); ++i) {
        for(size_t j = 0; j < sub.getCols(); ++j) {
            data[r1 + i][c1 + j] = sub(i, j);
        }
    }
}

Matrix Matrix::combine(const Matrix& C11, const Matrix& C12, 
                      const Matrix& C21, const Matrix& C22) {
    size_t n = C11.getRows() + C21.getRows();
    size_t m = C11.getCols() + C12.getCols();
    
    Matrix result(n, m);
    
    for(size_t i = 0; i < C11.getRows(); ++i) {
        for(size_t j = 0; j < C11.getCols(); ++j) {
            result(i, j) = C11(i, j);
        }
    }
    
    for(size_t i = 0; i < C12.getRows(); ++i) {
        for(size_t j = 0; j < C12.getCols(); ++j) {
            result(i, j + C11.getCols()) = C12(i, j);
        }
    }
    
    for(size_t i = 0; i < C21.getRows(); ++i) {
        for(size_t j = 0; j < C21.getCols(); ++j) {
            result(i + C11.getRows(), j) = C21(i, j);
        }
    }
    
    for(size_t i = 0; i < C22.getRows(); ++i) {
        for(size_t j = 0; j < C22.getCols(); ++j) {
            result(i + C11.getRows(), j + C11.getCols()) = C22(i, j);
        }
    }
    
    return result;
}

void Matrix::print() const {
    for(size_t i = 0; i < rows; ++i) {
        for(size_t j = 0; j < cols; ++j) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << data[i][j] << " ";
        }
        std::cout << "\n";
    }
}

std::vector<int> Matrix::toVector() const {
    std::vector<int> result;
    result.reserve(rows * cols);
    for(size_t i = 0; i < rows; ++i) {
        for(size_t j = 0; j < cols; ++j) {
            result.push_back(static_cast<int>(data[i][j]));
        }
    }
    return result;
}

Matrix Matrix::fromVector(const std::vector<int>& vec, size_t rows, size_t cols) {
    Matrix result(rows, cols);
    for(size_t i = 0; i < rows; ++i) {
        for(size_t j = 0; j < cols; ++j) {
            result(i, j) = vec[i * cols + j];
        }
    }
    return result;
}
