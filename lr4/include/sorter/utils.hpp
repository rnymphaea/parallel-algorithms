#ifndef SORTER_UTILS_HPP
#define SORTER_UTILS_HPP

#include <vector>
#include <random>

std::vector<int> generateRandomArray(int size, int min_val = 0, int max_val = 1000000);
bool isSorted(const std::vector<int>& array);
void printArray(const std::vector<int>& array, int limit = 20);

#endif
