#include "../include/Matrix.h"
#include "../include/MatrixMultiplier.h"
#include "../include/Timer.h"
#include "../include/options.h"

#define MAX_PRINT_MATRIX_SIZE 10

void printMatrixInfo(const Matrix &m, std::string name) {
    if (m.numRows() <= MAX_PRINT_MATRIX_SIZE && m.numCols() <= MAX_PRINT_MATRIX_SIZE) {
        std::cout << "Matrix " << name <<":\n";
        m.print();
    } else {
        std::cout << "Matrix " << name << " is too large to print (" << m.numRows() << "x" << m.numCols() << ")\n";
    }

}

int main(int argc, char* argv[]) {
    Options opts = parseOptions(argc, argv);

    Matrix A = opts.fileA.empty() ? Matrix(opts.rows, opts.cols) : Matrix::loadFromFile(opts.fileA);
    Matrix B = opts.fileB.empty() ? Matrix(opts.rows, opts.cols) : Matrix::loadFromFile(opts.fileB);

    if (opts.fileA.empty()) A.fillRandom();
    if (opts.fileB.empty()) B.fillRandom();
    
    printMatrixInfo(A, "A");
    printMatrixInfo(B, "B");
    
    Matrix C_single(A.numRows(), B.numCols());

    if (opts.measureTime) {
        double timeSingle = Timer::measureAverageTime([&]() {
            C_single = MatrixMultiplier::multiplySingleThread(A, B);
        }, opts.repeats);
        std::cout << "\nSingle-threaded multiplication time: " << timeSingle << " sec\n";
    } else {
         C_single = MatrixMultiplier::multiplySingleThread(A, B);
    }

    Matrix C_multi(A.numRows(), B.numCols());
    size_t numThreads = opts.threads > 0 ? opts.threads : std::thread::hardware_concurrency();

    if (opts.measureTime) {
        double timeMulti = Timer::measureAverageTime([&]() {
            C_multi = MatrixMultiplier::multiplyMultiThread(A, B, numThreads);
        }, opts.repeats);
        std::cout << "Multi-threaded multiplication time (" << numThreads << " threads): " << timeMulti << " sec\n";
    } else {
        C_multi = MatrixMultiplier::multiplyMultiThread(A, B, numThreads);
    }

    Matrix C_async(A.numRows(), B.numCols());
    size_t numTasks = numThreads;

    if (opts.measureTime) {
        double timeAsync = Timer::measureAverageTime([&]() {
            C_async = MatrixMultiplier::multiplyAsync(A, B, numTasks);
        }, opts.repeats);
        std::cout << "Async multiplication time (" << numTasks << " tasks): " << timeAsync << " sec\n";
    } else {
        C_async = MatrixMultiplier::multiplyAsync(A, B, numTasks);
    }

    bool equal = MatrixMultiplier::areEqual(C_single, C_multi);
    equal = MatrixMultiplier::areEqual(C_single, C_async);

    std::cout << "\nResults match: " << (equal ? "yes" : "no") << std::endl;

    if (!opts.output.empty()) {
        C_single.saveToFile(opts.output);
        std::cout << "Result saved to " << opts.output << "\n";
    } else {
        printMatrixInfo(C_single, "Result");
    }

    return 0;
}
