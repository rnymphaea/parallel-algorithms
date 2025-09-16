#ifndef OPTIONS_H
#define OPTIONS_H 

#include <unistd.h>
#include <string>
#include <cstdlib>
#include <getopt.h> 
#include <iostream>


struct Options {
    std::string fileA;
    std::string fileB;
    size_t rows = 4;
    size_t cols = 4;
    bool measureTime = false;
    size_t repeats = 3;
    std::string output;
};

Options parseOptions(int argc, char* argv[]);

#endif //OPTIONS_H
