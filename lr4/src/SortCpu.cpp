#include "SorterCpu.hpp"
#include <iostream>
#include <future>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <chrono>

void SorterCpu::sort(std::vector<int>& array, const CpuConfig& config) {
    if (array.size() <= 1) return;
    
    if (config.use_std_sort) {
        stdSort(array);
    } else if (config.num_threads > 1) {
        parallelSort(array, config);
    } else {
        std::vector<int> temp(array.size());
        iterativeMergeSort(array, temp);
    }
}

void SorterCpu::parallelSort(std::vector<int>& array, const CpuConfig& config) {
    if (array.size() <= 1) return;
    
    std::vector<int> temp(array.size());
    parallelIterativeMergeSort(array, temp, config.num_threads);
}

void SorterCpu::stdSort(std::vector<int>& array) {
    std::sort(array.begin(), array.end());
}

double SorterCpu::sortWithProfiling(std::vector<int>& array, const CpuConfig& config) {
    auto start_time = std::chrono::high_resolution_clock::now();
    sort(array, config);
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end_time - start_time).count();
}

void SorterCpu::iterativeMergeSort(std::vector<int>& array, std::vector<int>& temp) {
    int n = array.size();
    
    for (int width = 1; width < n; width *= 2) {
        for (int i = 0; i < n; i += 2 * width) {
            int left = i;
            int middle = std::min(i + width, n);
            int right = std::min(i + 2 * width, n);
            
            merge(array, temp, left, middle, right);
        }
        for (int i = 0; i < n; i++) {
            array[i] = temp[i];
        }
    }
}

void SorterCpu::parallelIterativeMergeSort(std::vector<int>& array, std::vector<int>& temp, int num_threads) {
    int n = array.size();
    
    num_threads = std::min(num_threads, n / 2);
    if (num_threads < 1) num_threads = 1;
    
    std::vector<std::thread> threads;
    std::atomic<int> barrier(0);
    
    for (int width = 1; width < n; width *= 2) {
        for (int t = 0; t < num_threads; t++) {
            threads.emplace_back(workerThread, 
                               std::ref(array), std::ref(temp),
                               width, n, t, num_threads,
                               std::ref(barrier));
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        threads.clear();
        barrier.store(0);
        
        for (int i = 0; i < n; i++) {
            array[i] = temp[i];
        }
    }
}

void SorterCpu::workerThread(std::vector<int>& array, std::vector<int>& temp, 
                            int width, int n, int thread_id, int num_threads,
                            std::atomic<int>& barrier) {
    int total_merges = (n + 2 * width - 1) / (2 * width);
    int merges_per_thread = (total_merges + num_threads - 1) / num_threads;
    
    int start_merge = thread_id * merges_per_thread;
    int end_merge = std::min(start_merge + merges_per_thread, total_merges);
    
    for (int merge_idx = start_merge; merge_idx < end_merge; merge_idx++) {
        int i = merge_idx * 2 * width;
        int left = i;
        int middle = std::min(i + width, n);
        int right = std::min(i + 2 * width, n);
        
        merge(array, temp, left, middle, right);
    }
    
    barrier.fetch_add(1);
    while (barrier.load() < num_threads) {
        std::this_thread::yield();
    }
}

void SorterCpu::merge(std::vector<int>& array, std::vector<int>& temp, 
                     int left, int middle, int right) {
    int i = left, j = middle, k = left;
    
    while (i < middle && j < right) {
        if (array[i] <= array[j]) {
            temp[k] = array[i];
            i++;
        } else {
            temp[k] = array[j];
            j++;
        }
        k++;
    }
    
    while (i < middle) {
        temp[k] = array[i];
        i++;
        k++;
    }
    
    while (j < right) {
        temp[k] = array[j];
        j++;
        k++;
    }
}
