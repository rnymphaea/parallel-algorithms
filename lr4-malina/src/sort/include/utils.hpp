#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <random>

std::vector<int> generate_random_array(int size, int min_val = 0, int max_val = 1000000);
bool is_sorted(const std::vector<int>& array);
void print_array(const std::vector<int>& array, int limit = 20);

#endif