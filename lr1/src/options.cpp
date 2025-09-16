#include "../include/options.h"

Options parseOptions(int argc, char* argv[]) {
    Options opts;
    int opt;
    int longIndex = 0;
  
    struct option longOpts[] = {
        {"rows",    required_argument, 0, 'r'},
        {"columns", required_argument, 0, 'c'},
        {"path-a",  required_argument, 0, 'a'},
        {"path-b",  required_argument, 0, 'b'},
        {"time",    no_argument,       0, 't'},
        {"repeats", required_argument, 0, 'n'},
        {"output",  required_argument, 0, 'o'},
        {"help",    no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "r:c:a:b:tn:o:h", longOpts, &longIndex)) != -1) {
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
        case 't': 
            opts.measureTime = true;
            break;
        case 'n':
            opts.repeats = std::stoi(optarg);
            break;
        case 'o':
            opts.output = optarg;
            break;
        case 'h':
        default:
            std::cout << "Usage: ./mm [OPTIONS]\n\n";
            std::cout << "Matrix multiplication program with single-threaded, multi-threaded, and async implementations.\n\n";
            std::cout << "Options:\n";
            std::cout << "  --rows [-r] M            Number of rows for randomly generated matrices (default: 4)\n";
            std::cout << "  --columns [-c] N         Number of columns for randomly generated matrices (default: 4)\n";
            std::cout << "  --path-a [-a] FILE       Load matrix A from the specified file\n";
            std::cout << "  --path-b [-b] FILE       Load matrix B from the specified file\n";
            std::cout << "  --time [-t]              Measure execution time for multi-threaded and async multiplication\n";
            std::cout << "  --repeats [-n] N         Number of repetitions to average timing results (default: 3)\n";
            std::cout << "  --output [-o] FILE       Specify output file to save result\n";
            std::cout << "  --help [-h]              Display this help message and exit\n\n";
            std::cout << "Notes:\n";
            std::cout << "- If --path-a or --path-b are not specified, the matrices will be generated randomly.\n";
            std::cout << "- If --time is not specified, the program will compute the result without measuring execution time,\n";
           std::cout << "  even if --repeats is given.\n";
            std::cout << "- Matrices larger than 10x10 are not printed to the console.\n";
            std::cout << "- You can optionally add --output FILE to save the result matrix to a file.\n";
            exit(0);
        }
    }
    return opts;
}
