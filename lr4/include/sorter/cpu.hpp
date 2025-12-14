#ifndef SORTER_CPU_HPP
#define SORTER_CPU_HPP

#include <vector>
#include <thread>
#include <algorithm>
#include <future>
#include <atomic>

struct CPUConfig {
    int num_threads = 1;
    bool use_std_sort = false;
};

class SorterCPU {
public:
    static void sort(std::vector<int>& array, const CPUConfig& config = CPUConfig());
    static void parallelSort(std::vector<int>& array, const CPUConfig& config = CPUConfig());
    static void stdSort(std::vector<int>& array);
    
    static double sortWithProfiling(std::vector<int>& array, const CPUConfig& config = CPUConfig());
    
private:
    static void iterativeMergeSort(std::vector<int>& array, std::vector<int>& temp);
    static void parallelIterativeMergeSort(std::vector<int>& array, std::vector<int>& temp, int num_threads);
    static void merge(std::vector<int>& array, std::vector<int>& temp, 
                     int left, int middle, int right);
    
    static void workerThread(std::vector<int>& array, std::vector<int>& temp, 
                            int width, int n, int thread_id, int num_threads,
                            std::atomic<int>& barrier);
};

#endif
