#ifndef PARALLEL_SORT_HPP
#define PARALLEL_SORT_HPP

#include <vector>
#include <thread>
#include <algorithm>
#include <future>

class ParallelSort {
private:
    size_t maxThreads;
    
    void parallelMergeSort(std::vector<int>& arr, size_t left, size_t right, size_t depth);
    
public:
    ParallelSort(size_t threads = std::thread::hardware_concurrency());
    
    void sort(std::vector<int>& arr);
    
    static void singleThreadSort(std::vector<int>& arr);
    
    static bool isSorted(const std::vector<int>& arr);
};

#endif
