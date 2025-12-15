#include "SorterCpu.hpp"
#include "SorterGpu.hpp"
#include "SortUtils.hpp"
#include <iostream>
#include <vector>

int main() {
    std::cout << "Sorting Demo" << std::endl;
    std::cout << "============" << std::endl;
    std::cout << std::endl;
    
    std::vector<int> data = {5, 2, 9, 1, 5, 6, 3, 8, 7, 4};
    std::cout << "Original array: ";
    printArray(data);
    
    std::cout << std::endl;
    std::cout << "CPU Sort (std::sort):" << std::endl;
    auto cpu_data = data;
    SorterCpu::stdSort(cpu_data);
    std::cout << "Result: ";
    printArray(cpu_data);
    std::cout << "Sorted: " << (isSorted(cpu_data) ? "Yes" : "No") << std::endl;
    
    std::cout << std::endl;
    std::cout << "GPU Sort:" << std::endl;
    try {
        auto gpu_data = data;
        SorterGpu sorter;
        sorter.sort(gpu_data);
        std::cout << "Result: ";
        printArray(gpu_data);
        std::cout << "Sorted: " << (isSorted(gpu_data) ? "Yes" : "No") << std::endl;
    } catch (const std::exception& e) {
        std::cout << "GPU sort failed: " << e.what() << std::endl;
    }
    
    return 0;
}
