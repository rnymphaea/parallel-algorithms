#include "../include/sorter/cpu.hpp"
#include "../include/sorter/gpu.hpp"
#include "../include/sorter/utils.hpp"
#include <iostream>
#include <vector>

int main() {
    std::cout << "Sorting Demo" << std::endl;
    
    std::vector<int> data = {5, 2, 9, 1, 5, 6, 3, 8, 7, 4};
    std::cout << "Original array: ";
    printArray(data);
    
    std::cout << "\nCPU Sort (std::sort):" << std::endl;
    auto cpu_data = data;
    SorterCPU::stdSort(cpu_data);
    std::cout << "Result: ";
    printArray(cpu_data);
    std::cout << "Sorted: " << (isSorted(cpu_data) ? "Yes" : "No") << std::endl;
    
    std::cout << "\nGPU Sort:" << std::endl;
    try {
        auto gpu_data = data;
        SorterGPU sorter;
        sorter.sort(gpu_data);
        std::cout << "Result: ";
        printArray(gpu_data);
        std::cout << "Sorted: " << (isSorted(gpu_data) ? "Yes" : "No") << std::endl;
    } catch (const std::exception& e) {
        std::cout << "GPU sort failed: " << e.what() << std::endl;
    }
    
    return 0;
}
