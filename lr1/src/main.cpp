#include "../include/Matrix.h"
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

  std::cout << "Matrix A:\n"; A.print();
  std::cout << "Matrix B:\n"; B.print();

  return 0;
}
