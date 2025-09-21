#include "../include/Matrix.h"
#include "../include/MatrixMultiplier.h"
#include "../include/Timer.h"
#include "../include/options.h"

#include <fstream>

#define MAX_PRINT_MATRIX_SIZE 10

void printMatrixInfo(const Matrix &m, std::string name, bool debug) {
    if ((m.numRows() <= MAX_PRINT_MATRIX_SIZE && m.numCols() <= MAX_PRINT_MATRIX_SIZE) || debug) {
        std::cout << "Matrix " << name <<":\n";
        m.print();
    } else {
        std::cout << "Matrix " << name << " is too large to print (" << m.numRows() << "x" << m.numCols() << ")\n";
    }

}

int main(int argc, char* argv[]) {
    Options opts = parseOptions(argc, argv);
    if (opts.debug) {
        std::cout << opts << "\n";
    }

    MatrixMultiplier multiplier(opts.blockSize);

    Matrix A = opts.fileA.empty() ? Matrix(opts.rows, opts.cols) : Matrix::loadFromFile(opts.fileA);
    Matrix B = opts.fileB.empty() ? Matrix(opts.rows, opts.cols) : Matrix::loadFromFile(opts.fileB);

    if (opts.fileA.empty()) A.fillRandom();
    if (opts.fileB.empty()) B.fillRandom();
    
    printMatrixInfo(A, "A", opts.debug);
    printMatrixInfo(B, "B", opts.debug);
    
    Matrix C_single(A.numRows(), B.numCols());

    double timeSingle, timeMulti, timeAsync;

    if (opts.measureTime) {
        timeSingle = Timer::measureAverageTime([&]() {
            C_single = multiplier.multiplySingleThread(A, B);
        }, opts.repeats);
        std::cout << "\nSingle-threaded multiplication time: " << timeSingle << " sec\n";
    } else {
         C_single = multiplier.multiplySingleThread(A, B);
    }

    Matrix C_multi(A.numRows(), B.numCols());
    size_t numThreads = opts.threads > 0 ? opts.threads : std::thread::hardware_concurrency();

    if (opts.measureTime) {
        timeMulti = Timer::measureAverageTime([&]() {
            C_multi = multiplier.multiplyMultiThread(A, B, numThreads);
        }, opts.repeats);
        std::cout << "Multi-threaded multiplication time (" << numThreads << " threads): " << timeMulti << " sec\n";
    } else {
        C_multi = multiplier.multiplyMultiThread(A, B, numThreads);
    }

    Matrix C_async(A.numRows(), B.numCols());
    size_t numTasks = numThreads;

    if (opts.measureTime) {
        timeAsync = Timer::measureAverageTime([&]() {
            C_async = multiplier.multiplyAsync(A, B, numTasks);
        }, opts.repeats);
        std::cout << "Async multiplication time (" << numTasks << " tasks): " << timeAsync << " sec\n";
    } else {
        C_async = multiplier.multiplyAsync(A, B, numTasks);
    }

    bool equal = MatrixMultiplier::areEqual(C_single, C_multi);
    equal = MatrixMultiplier::areEqual(C_single, C_async);

    std::cout << "\nResults match: " << (equal ? "yes" : "no") << std::endl;

    if (!opts.output.empty()) {
        if (opts.debug) {
            C_multi.saveToFile(opts.output);
            C_async.saveToFile(opts.output);
        }
        C_single.saveToFile(opts.output);
        std::cout << "Result saved to " << opts.output << "\n";
    } else {
        printMatrixInfo(C_multi, "Multi", opts.debug);
        printMatrixInfo(C_async, "Async", opts.debug);
        printMatrixInfo(C_single, "Result", opts.debug);
    }

    if (!opts.csv.empty() && opts.measureTime) {
        std::ofstream csv(opts.csv, std::ios::app);
        if (!csv) {
            std::cerr << "Error: cannot open CSV file for writing\n";
            return 1;
        }
        if (csv.tellp() == 0) {
            csv << "threads,single,multi,async\n";
        }
        csv << numThreads << "," << timeSingle << "," << timeMulti << "," << timeAsync << "\n";
        if (opts.debug) {
            std::cout << "Exported timings to " << opts.csv << std::endl;
        }
    }

    return 0;
}
