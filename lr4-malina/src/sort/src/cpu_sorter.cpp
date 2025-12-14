#include "cpu_sorter.hpp"
#include <iostream>
#include <future>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>

void CPUSorter::merge_sort(std::vector<int>& array, const CPUConfig& config) {
    if (array.size() <= 1) return;
    
    if (config.use_std_sort) {
        std_sort(array);
    } else if (config.num_threads > 1) {
        parallel_merge_sort(array, config);
    } else {
        std::vector<int> temp(array.size());
        iterative_merge_sort(array, temp);
    }
}

void CPUSorter::parallel_merge_sort(std::vector<int>& array, const CPUConfig& config) {
    if (array.size() <= 1) return;
    
    std::vector<int> temp(array.size());
    parallel_iterative_merge_sort(array, temp, config.num_threads);
}

void CPUSorter::std_sort(std::vector<int>& array) {
    std::sort(array.begin(), array.end());
}

double CPUSorter::sort_with_profiling(std::vector<int>& array, const CPUConfig& config) {
    auto start_time = std::chrono::high_resolution_clock::now();
    merge_sort(array, config);
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end_time - start_time).count();
}

void CPUSorter::iterative_merge_sort(std::vector<int>& array, std::vector<int>& temp) {
    int n = array.size();
    
    for (int width = 1; width < n; width *= 2) {
        for (int i = 0; i < n; i += 2 * width) {
            int left = i;
            int middle = std::min(i + width, n);
            int right = std::min(i + 2 * width, n);
            
            merge(array, temp, left, middle, right);
        }
        // Копируем результат обратно
        for (int i = 0; i < n; i++) {
            array[i] = temp[i];
        }
    }
}

void CPUSorter::parallel_iterative_merge_sort(std::vector<int>& array, std::vector<int>& temp, int num_threads) {
    int n = array.size();
    
    // Используем не больше потоков, чем нужно
    num_threads = std::min(num_threads, n / 2);
    if (num_threads < 1) num_threads = 1;
    
    std::vector<std::thread> threads;
    std::atomic<int> barrier(0);
    
    for (int width = 1; width < n; width *= 2) {
        // Запускаем потоки для текущего уровня слияния
        for (int t = 0; t < num_threads; t++) {
            threads.emplace_back(worker_thread, 
                               std::ref(array), std::ref(temp),
                               width, n, t, num_threads,
                               std::ref(barrier));
        }
        
        // Ждем завершения всех потоков
        for (auto& thread : threads) {
            thread.join();
        }
        threads.clear();
        barrier.store(0);
        
        // Копируем результат обратно в основной массив
        for (int i = 0; i < n; i++) {
            array[i] = temp[i];
        }
    }
}

void CPUSorter::worker_thread(std::vector<int>& array, std::vector<int>& temp, 
                            int width, int n, int thread_id, int num_threads,
                            std::atomic<int>& barrier) {
    // Вычисляем диапазон для этого потока
    int total_merges = (n + 2 * width - 1) / (2 * width);
    int merges_per_thread = (total_merges + num_threads - 1) / num_threads;
    
    int start_merge = thread_id * merges_per_thread;
    int end_merge = std::min(start_merge + merges_per_thread, total_merges);
    
    // Выполняем назначенные слияния
    for (int merge_idx = start_merge; merge_idx < end_merge; merge_idx++) {
        int i = merge_idx * 2 * width;
        int left = i;
        int middle = std::min(i + width, n);
        int right = std::min(i + 2 * width, n);
        
        merge(array, temp, left, middle, right);
    }
    
    // Синхронизация (простой барьер)
    barrier.fetch_add(1);
    while (barrier.load() < num_threads) {
        std::this_thread::yield();
    }
}

void CPUSorter::merge(std::vector<int>& array, std::vector<int>& temp, 
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