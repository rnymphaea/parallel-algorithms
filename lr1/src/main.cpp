#include "../include/Matrix.h"
#include "../include/MatrixMultiplier.h"
#include "../include/Timer.h"
#include "../include/options.h"

int main(int argc, char* argv[]) {
    Options opts = parseOptions(argc, argv);

    Matrix A = opts.fileA.empty() ? Matrix(opts.rows, opts.cols) : Matrix::loadFromFile(opts.fileA);
    Matrix B = opts.fileB.empty() ? Matrix(opts.rows, opts.cols) : Matrix::loadFromFile(opts.fileB);

    if (opts.fileA.empty()) A.fillRandom();
    if (opts.fileB.empty()) B.fillRandom();
    
    if (A.numRows() <= 10 && A.numCols() <= 10) {
        std::cout << "Matrix A:\n";
        A.print();
    } else {
        std::cout << "Matrix A is too large to print (" << A.numRows() << "x" << A.numCols() << ")\n";
    }

    if (B.numRows() <= 10 && B.numCols() <= 10) {
        std::cout << "Matrix B:\n";
        B.print();
    } else {
        std::cout << "Matrix B is too large to print (" << B.numRows() << "x" << B.numCols() << ")\n";
    }
    
    Matrix C_single(A.numRows(), B.numCols());

    if (opts.measureTime) {
        double timeSingle = Timer::measureAverageTime([&]() {
            C_single = MatrixMultiplier::multiplySingleThread(A, B);
        });
        std::cout << "\nSingle-threaded multiplication time: " << timeSingle << " sec\n";
    } else {
         C_single = MatrixMultiplier::multiplySingleThread(A, B);
    }

    Matrix C_multi(A.numRows(), B.numCols());
    size_t numThreads = std::thread::hardware_concurrency();

    if (opts.measureTime) {
        double timeMulti = Timer::measureAverageTime([&]() {
            C_multi = MatrixMultiplier::multiplyMultiThread(A, B, numThreads);
        });
        std::cout << "Multi-threaded multiplication time (" << numThreads << " threads): " << timeMulti << " sec\n";
    } else {
        C_multi = MatrixMultiplier::multiplyMultiThread(A, B, numThreads);
    }

    Matrix C_async(A.numRows(), B.numCols());
    size_t numTasks = std::thread::hardware_concurrency();

    if (opts.measureTime) {
        double timeAsync = Timer::measureAverageTime([&]() {
            C_async = MatrixMultiplier::multiplyAsync(A, B, numTasks);
        });
        std::cout << "Async multiplication time (" << numTasks << " tasks): " << timeAsync << " sec\n";
    } else {
        C_async = MatrixMultiplier::multiplyAsync(A, B, numTasks);
    }

    bool equal = MatrixMultiplier::areEqual(C_single, C_multi);
    equal = MatrixMultiplier::areEqual(C_single, C_async);

    std::cout << "\nResults match? " << (equal ? "Yes" : "No") << std::endl;

    if (!opts.output.empty()) {
        C_single.saveToFile(opts.output);
        std::cout << "Result saved to " << opts.output << "\n";
    } else if (C_single.numRows() <= 10 && C_single.numCols() <= 10) {
        std::cout << "\nResult matrix:\n";
        C_single.print();
    } else {
        std::cout << "\nResult matrix is too large to print (" << C_single.numRows() << "x" << C_single.numCols() << ")\n";
    }

    return 0;
}
