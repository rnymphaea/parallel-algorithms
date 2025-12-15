#include "../include/ParallelSort.hpp"
#include <iostream>
#include <future>

ParallelSort::ParallelSort(size_t threads) 
    : maxThreads(threads) {}

static void merge(std::vector<int>& arr, size_t left, size_t mid, size_t right) {
    std::vector<int> leftArr(arr.begin() + left, arr.begin() + mid + 1);
    std::vector<int> rightArr(arr.begin() + mid + 1, arr.begin() + right + 1);
    
    size_t i = 0, j = 0, k = left;
    
    while(i < leftArr.size() && j < rightArr.size()) {
        if(leftArr[i] <= rightArr[j]) {
            arr[k++] = leftArr[i++];
        } else {
            arr[k++] = rightArr[j++];
        }
    }
    
    while(i < leftArr.size()) {
        arr[k++] = leftArr[i++];
    }
    
    while(j < rightArr.size()) {
        arr[k++] = rightArr[j++];
    }
}

static void singleThreadMergeSort(std::vector<int>& arr, size_t left, size_t right) {
    if(left >= right) return;
    
    size_t mid = left + (right - left) / 2;
    singleThreadMergeSort(arr, left, mid);
    singleThreadMergeSort(arr, mid + 1, right);
    merge(arr, left, mid, right);
}

void ParallelSort::singleThreadSort(std::vector<int>& arr) {
    if(arr.empty()) return;
    singleThreadMergeSort(arr, 0, arr.size() - 1);
}

void ParallelSort::parallelMergeSort(std::vector<int>& arr, size_t left, size_t right, size_t depth) {
    if(left >= right) return;
    
    if(depth >= 3 || (right - left) < 10000) {
        singleThreadMergeSort(arr, left, right);
        return;
    }
    
    size_t mid = left + (right - left) / 2;
    
    auto future = std::async(std::launch::async, [&arr, left, mid, depth]() {
        ParallelSort tempSorter(2);
        tempSorter.parallelMergeSort(arr, left, mid, depth + 1);
    });
    
    parallelMergeSort(arr, mid + 1, right, depth + 1);
    
    future.get();
    
    merge(arr, left, mid, right);
}

void ParallelSort::sort(std::vector<int>& arr) {
    if(arr.empty()) return;
    
    if(maxThreads > 1 && arr.size() > 10000) {
        parallelMergeSort(arr, 0, arr.size() - 1, 0);
    } else {
        singleThreadMergeSort(arr, 0, arr.size() - 1);
    }
}

bool ParallelSort::isSorted(const std::vector<int>& arr) {
    for(size_t i = 1; i < arr.size(); ++i) {
        if(arr[i] < arr[i - 1]) {
            return false;
        }
    }
    return true;
}
