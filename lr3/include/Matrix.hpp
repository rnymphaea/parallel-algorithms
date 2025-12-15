#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <cassert>

class Matrix {
private:
    std::vector<std::vector<double>> data;
    size_t rows, cols;

public:
    Matrix(size_t r = 0, size_t c = 0);
    Matrix(const std::vector<std::vector<double>>& d);
    
    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
    
    double& operator()(size_t i, size_t j);
    const double& operator()(size_t i, size_t j) const;
    
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    
    void fillRandom();
    void fillRandomInt(int minVal, int maxVal);
    bool operator==(const Matrix& other) const;
    bool operator!=(const Matrix& other) const;
    
    Matrix getSubmatrix(size_t r1, size_t r2, size_t c1, size_t c2) const;
    void setSubmatrix(size_t r1, size_t c1, const Matrix& sub);
    
    static Matrix combine(const Matrix& C11, const Matrix& C12, 
                         const Matrix& C21, const Matrix& C22);
    
    void print() const;
    
    // For sorting comparison
    std::vector<int> toVector() const;
    static Matrix fromVector(const std::vector<int>& vec, size_t rows, size_t cols);
};

#endif
