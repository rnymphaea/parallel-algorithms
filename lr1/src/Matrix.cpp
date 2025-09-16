#include "../include/Matrix.h"

Matrix::Matrix(size_t r, size_t c) : rows(r), cols(c), data(r, std::vector<double>(c, 0.0)) {}

size_t Matrix::numRows() const { return rows; }
size_t Matrix::numCols() const { return cols; }

double& Matrix::operator()(size_t i, size_t j) { return data[i][j]; }
const double& Matrix::operator()(size_t i, size_t j) const { return data[i][j]; }

void Matrix::fillRandom(double minVal, double maxVal) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(minVal, maxVal);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            data[i][j] = dist(gen);
        }
    }
}

void Matrix::saveToFile(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("error: cannot open file to write");

    out << rows << " " << cols << "\n";
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            out << std::fixed << std::setprecision(2) << data[i][j] << " ";
        }
        out << "\n";
    }
}

Matrix Matrix::loadFromFile(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("error: cannot open file to read");

    size_t r, c;
    in >> r >> c;
    Matrix m(r, c);

    for (size_t i = 0; i < r; i++) {
        for (size_t j = 0; j < c; j++) {
            in >> m(i, j);
        }
    }
    return m;
}

void Matrix::print() const {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << data[i][j] << " ";
        }
        std::cout << "\n";
    }
}

