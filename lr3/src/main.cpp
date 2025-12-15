#include "../include/Matrix.hpp"
#include "../include/StrassenMultiplier.hpp"
#include "../include/BlockMultiplier.hpp"
#include "../include/ParallelSort.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <random>
#include <string>
#include <fstream>

class TestTimer {
private:
    std::chrono::high_resolution_clock::time_point start;
    
public:
    TestTimer() {
        start = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(now - start).count();
    }
};

void printHeader(const std::string& header) {
    std::cout << "\n=== " << header << " ===" << std::endl;
}

void exportMatrixResults() {
    std::ofstream outFile("matrix_results.csv");
    if (!outFile.is_open()) {
        std::cout << "Error opening matrix_results.csv" << std::endl;
        return;
    }
    
    outFile << "Algorithm,Size,Threads,Time\n";
    
    std::vector<size_t> sizes = {64, 128, 256, 512, 1024, 2048};
    std::vector<size_t> threads = {1, 2, 4, 8};
    
    for (size_t size : sizes) {
        std::cout << "\nTesting matrix " << size << "x" << size << "..." << std::endl;
        
        Matrix A(size, size);
        Matrix B(size, size);
        A.fillRandom();
        B.fillRandom();
        
        for (size_t t : threads) {
            {
                StrassenMultiplier sm(t);
                TestTimer timer;
                sm.multiply(A, B);
                double time = timer.elapsed();
                outFile << "Strassen," << size << "," << t << "," 
                        << std::fixed << std::setprecision(6) << time << "\n";
                std::cout << "  Strassen " << t << " threads: " << time << " s" << std::endl;
            }
            
            {
                BlockMultiplier bm(t);
                TestTimer timer;
                bm.multiply(A, B);
                double time = timer.elapsed();
                outFile << "Block," << size << "," << t << "," 
                        << std::fixed << std::setprecision(6) << time << "\n";
                std::cout << "  Block " << t << " threads: " << time << " s" << std::endl;
            }
        }
    }
    
    outFile.close();
    std::cout << "\nMatrix results exported to matrix_results.csv" << std::endl;
}

void exportSortingResults() {
    std::ofstream outFile("sorting_results.csv");
    if (!outFile.is_open()) {
        std::cout << "Error opening sorting_results.csv" << std::endl;
        return;
    }
    
    outFile << "Algorithm,Size,Threads,Time\n";
    
    std::vector<size_t> sizes = {10000, 50000, 100000, 500000, 1000000, 5000000, 10000000};
    std::vector<size_t> threads = {1, 2, 4, 8};
    
    for (size_t size : sizes) {
        std::cout << "\nTesting array " << size << " elements..." << std::endl;
        
        std::vector<int> baseArray(size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, 10000000);
        
        for (auto& val : baseArray) {
            val = dis(gen);
        }
        
        {
            std::vector<int> testArray = baseArray;
            TestTimer timer;
            ParallelSort::singleThreadSort(testArray);
            double time = timer.elapsed();
            outFile << "single_thread_merge_sort," << size << "," << 1 << "," 
                    << std::fixed << std::setprecision(6) << time << "\n";
            std::cout << "  Single-thread merge sort: " << time << " s" << std::endl;
        }
        
        for (size_t t : threads) {
            if (t == 1) continue;
            
            std::vector<int> testArray = baseArray;
            ParallelSort sorter(t);
            TestTimer timer;
            sorter.sort(testArray);
            double time = timer.elapsed();
            outFile << "parallel_merge_sort," << size << "," << t << "," 
                    << std::fixed << std::setprecision(6) << time << "\n";
            std::cout << "  Parallel merge sort (" << t << " threads): " << time << " s" << std::endl;
        }
    }
    
    outFile.close();
    std::cout << "\nSorting results exported to sorting_results.csv" << std::endl;
}

void testCorrectness() {
    printHeader("CORRECTNESS TESTS");
    
    std::cout << "\nMatrix multiplication (64x64):" << std::endl;
    Matrix A(64, 64);
    Matrix B(64, 64);
    A.fillRandom();
    B.fillRandom();
    
    Matrix naive = StrassenMultiplier::naiveMultiply(A, B);
    StrassenMultiplier sm(1);
    BlockMultiplier bm(1);
    
    if (sm.multiply(A, B) == naive && bm.multiply(A, B) == naive) {
        std::cout << "  Matrix algorithms: OK" << std::endl;
    } else {
        std::cout << "  Matrix algorithms: FAIL" << std::endl;
    }
    
    std::cout << "\nSorting (100000 elements):" << std::endl;
    std::vector<int> array(100000);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10000000);
    
    for (auto& val : array) {
        val = dis(gen);
    }
    
    std::vector<int> single = array;
    ParallelSort sorter(4);
    std::vector<int> parallel = array;
    
    ParallelSort::singleThreadSort(single);
    sorter.sort(parallel);
    
    bool same = true;
    for (size_t i = 0; i < single.size(); i++) {
        if (single[i] != parallel[i]) {
            same = false;
            break;
        }
    }
    
    if (ParallelSort::isSorted(single) && ParallelSort::isSorted(parallel) && same) {
        std::cout << "  Sorting algorithms: OK" << std::endl;
    } else {
        std::cout << "  Sorting algorithms: FAIL" << std::endl;
    }
}

int main() {
    std::cout << "PARALLEL ALGORITHMS BENCHMARK\n";
    std::cout << "=============================\n";
    
    try {
        testCorrectness();
        exportMatrixResults();
        exportSortingResults();
        
        std::cout << "\n=== ALL TESTS COMPLETED ===" << std::endl;
        std::cout << "\nFiles created:" << std::endl;
        std::cout << "  matrix_results.csv - Matrix multiplication results" << std::endl;
        std::cout << "  sorting_results.csv - Sorting results" << std::endl;
        std::cout << "\nRun visualization scripts:" << std::endl;
        std::cout << "  python3 plot_matrix.py" << std::endl;
        std::cout << "  python3 plot_sorting.py" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
