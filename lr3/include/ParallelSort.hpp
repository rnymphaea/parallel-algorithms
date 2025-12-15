#ifndef PARALLEL_SORT_HPP
#define PARALLEL_SORT_HPP

#include <vector>
#include <thread>
#include <algorithm>
#include <future>

class ParallelSort {
private:
    size_t maxThreads;
    
    void mergeSort(std::vector<int>& arr, size_t left, size_t right);
    void parallelMergeSort(std::vector<int>& arr, size_t left, size_t right, size_t depth);
    void merge(std::vector<int>& arr, size_t left, size_t mid, size_t right);
    
public:
    ParallelSort(size_t threads = std::thread::hardware_concurrency());
    
    void sort(std::vector<int>& arr);
    void sortSingleThread(std::vector<int>& arr);
    
    static bool isSorted(const std::vector<int>& arr);
};

#endif
