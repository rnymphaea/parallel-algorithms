#include "../include/Matrix.h"
#include "../include/MatrixMultiplier.h"

#include <unistd.h>
#include <string>
#include <cstdlib>
#include <getopt.h> 

struct Options {
  std::string fileA;
  std::string fileB;
  size_t rows = 4;
  size_t cols = 4;
};

Options parseOptions(int argc, char* argv[]) {
    Options opts;
    int opt;
    int longIndex = 0;
  
    struct option longOpts[] = {
        {"rows",    required_argument, 0, 'r'},
        {"columns", required_argument, 0, 'c'},
        {"path-a",  required_argument, 0, 'a'},
        {"path-b",  required_argument, 0, 'b'},
        {"help",    no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "r:c:a:b:h", longOpts, &longIndex)) != -1) {
        switch (opt) {
        case 'r':
            opts.rows = std::stoi(optarg);
            break;
        case 'c':
            opts.cols = std::stoi(optarg);
            break;
        case 'a':
            opts.fileA = optarg;
            break;
        case 'b':
            opts.fileB = optarg;
            break;
        case 'h':
        default:
            std::cout << "Usage:\n";
            std::cout << "  -r M            Num rows\n";
            std::cout << "  -c N            Num columns\n";
            std::cout << "  -path-a FILE    Matrix A from file\n";
            std::cout << "  -path-b FILE    Matrix B from file\n";
            exit(0);
        }
    }

    return opts;
}

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
    double timeSingle = MatrixMultiplier::measureTime([&]() {
        C_single = MatrixMultiplier::multiplySingleThread(A, B);
    });
    std::cout << "\nSingle-threaded multiplication time: " << timeSingle << " sec\n";

    Matrix C_multi(A.numRows(), B.numCols());
    size_t numThreads = std::thread::hardware_concurrency();
    double timeMulti = MatrixMultiplier::measureTime([&]() {
        C_multi = MatrixMultiplier::multiplyMultiThread(A, B, numThreads);
    });
    std::cout << "Multi-threaded multiplication time (" << numThreads << " threads): " << timeMulti << " sec\n";
    
    Matrix C_async(A.numRows(), B.numCols());
    size_t numTasks = std::thread::hardware_concurrency();
    double timeAsync = MatrixMultiplier::measureTime([&]() {
        C_async = MatrixMultiplier::multiplyAsync(A, B, numTasks);
    });
    std::cout << "Async multiplication time (" << numTasks << " tasks): " << timeAsync << " sec\n";

    bool equal = MatrixMultiplier::areEqual(C_single, C_multi);
    equal = MatrixMultiplier::areEqual(C_single, C_async);
    std::cout << "Results match? " << (equal ? "Yes" : "No") << std::endl;

    if (C_single.numRows() <= 10 && C_single.numCols() <= 10) {
        std::cout << "\nResult matrix:\n";
        C_single.print();
    } else {
        std::cout << "\nResult matrix is too large to print (" << C_single.numRows() << "x" << C_single.numCols() << ")\n";
    }

    return 0;
}
