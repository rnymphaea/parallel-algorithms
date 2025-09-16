#ifndef MATRIX_H 
#define MATRIX_H

#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <iomanip>

class Matrix {
public:
    Matrix(size_t r, size_t c);
    size_t numRows() const;
    size_t numCols() const;

    double& operator()(size_t i, size_t j);
    const double& operator()(size_t i, size_t j) const;

    void fillRandom(double minVal = 0.0, double maxVal = 10.0);
    void saveToFile(const std::string& filename) const;
    static Matrix loadFromFile(const std::string& filename);
    void print() const;

private:
    size_t rows, cols;
    std::vector<std::vector<double>> data;
};

#endif //MATRIX_H
