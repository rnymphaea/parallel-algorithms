#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include "SorterGpu.hpp"
#include "SorterCpu.hpp"
#include "SortUtils.hpp"

class CorrectnessTester {
private:
    std::ofstream csv_file;
    
public:
    CorrectnessTester() {
        system("mkdir -p results");
        csv_file.open("results/sort_correctness.csv");
        csv_file << "TestID,ArraySize,DataType,GPUCorrect,CPUSingleCorrect,CPUParallelCorrect,CPUStdCorrect,AllCorrect\n";
    }
    
    ~CorrectnessTester() {
        if (csv_file.is_open()) {
            csv_file.close();
        }
    }
    
    void testRandomArray(int size, int test_id) {
        std::cout << "Test " << test_id << ": Random array, size " << size << std::endl;
        
        auto original_data = generateRandomArray(size);
        auto data_gpu = original_data;
        auto data_cpu_single = original_data;
        auto data_cpu_parallel = original_data;
        auto data_cpu_std = original_data;
        
        SorterGpu gpu_sorter;
        gpu_sorter.sort(data_gpu);
        bool gpu_correct = isSorted(data_gpu);
        
        SorterCpu::sort(data_cpu_single);
        bool cpu_single_correct = isSorted(data_cpu_single);
        
        SorterCpu::parallelSort(data_cpu_parallel);
        bool cpu_parallel_correct = isSorted(data_cpu_parallel);
        
        SorterCpu::stdSort(data_cpu_std);
        bool cpu_std_correct = isSorted(data_cpu_std);
        
        bool all_correct = gpu_correct && cpu_single_correct && cpu_parallel_correct && cpu_std_correct;
        
        csv_file << test_id << "," << size << ",random,"
                << gpu_correct << "," << cpu_single_correct << ","
                << cpu_parallel_correct << "," << cpu_std_correct << ","
                << all_correct << "\n";
        
        std::cout << "  GPU: " << (gpu_correct ? "OK" : "FAIL") << "  "
                  << "CPU Single: " << (cpu_single_correct ? "OK" : "FAIL") << "  "
                  << "CPU Parallel: " << (cpu_parallel_correct ? "OK" : "FAIL") << "  "
                  << "CPU STD: " << (cpu_std_correct ? "OK" : "FAIL") << "  "
                  << "Overall: " << (all_correct ? "PASS" : "FAIL") << std::endl;
        
        if (!all_correct && size <= 20) {
            std::cout << "  Original: ";
            printArray(original_data);
            std::cout << "  GPU:      ";
            printArray(data_gpu);
            std::cout << "  CPU STD:  ";
            printArray(data_cpu_std);
        }
    }
    
    void runAllTests() {
        std::cout << "=== Sorting Correctness Tests ===" << std::endl;
        std::cout << std::endl;
        
        int test_id = 1;
        
        testRandomArray(10, test_id++);
        std::cout << std::endl;
        testRandomArray(16, test_id++);
        std::cout << std::endl;
        testRandomArray(32, test_id++);
        std::cout << std::endl;
        testRandomArray(1000, test_id++);
        std::cout << std::endl;
        testRandomArray(5000, test_id++);
        std::cout << std::endl;
        testRandomArray(10000, test_id++);
        std::cout << std::endl;
        testRandomArray(50000, test_id++);
        std::cout << std::endl;
        
        std::cout << "=== Tests Complete ===" << std::endl;
        std::cout << "Results saved to results/sort_correctness.csv" << std::endl;
    }
};

int main() {
    try {
        CorrectnessTester tester;
        tester.runAllTests();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
