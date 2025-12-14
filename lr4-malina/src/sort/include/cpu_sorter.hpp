#ifndef CPU_SORTER_HPP
#define CPU_SORTER_HPP

#include <vector>
#include <thread>
#include <algorithm>
#include <future>
#include <atomic>

struct CPUConfig {
    int num_threads = 1;
    bool use_std_sort = false;
};

class CPUSorter {
public:
    static void merge_sort(std::vector<int>& array, const CPUConfig& config = CPUConfig());
    static void parallel_merge_sort(std::vector<int>& array, const CPUConfig& config = CPUConfig());
    static void std_sort(std::vector<int>& array);
    
    // Вспомогательные методы для бенчмарков
    static double sort_with_profiling(std::vector<int>& array, const CPUConfig& config = CPUConfig());
    
private:
    static void iterative_merge_sort(std::vector<int>& array, std::vector<int>& temp);
    
    // Многопоточная итеративная версия
    static void parallel_iterative_merge_sort(std::vector<int>& array, std::vector<int>& temp, int num_threads);
    static void merge(std::vector<int>& array, std::vector<int>& temp, 
                     int left, int middle, int right);
    
    // Вспомогательные функции для многопоточности
    static void worker_thread(std::vector<int>& array, std::vector<int>& temp, 
                            int width, int n, int thread_id, int num_threads,
                            std::atomic<int>& barrier);
};

#endif