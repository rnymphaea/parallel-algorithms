#include "utils.hpp"
#include <iostream>
#include <random>

std::vector<int> generate_random_array(int size, int min_val, int max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(min_val, max_val);
    
    std::vector<int> array(size);
    for (int i = 0; i < size; i++) {
        array[i] = dist(gen);
    }
    return array;
}

bool is_sorted(const std::vector<int>& array) {
    for (size_t i = 1; i < array.size(); i++) {
        if (array[i] < array[i - 1]) {
            std::cout << "Sort error at position " << i 
                      << ": " << array[i-1] << " > " << array[i] << std::endl;
            return false;
        }
    }
    return true;
}

void print_array(const std::vector<int>& array, int limit) {
    if (array.size() > limit) {
        std::cout << "[";
        for (int i = 0; i < limit/2; i++) {
            std::cout << array[i] << ", ";
        }
        std::cout << "... , ";
        for (size_t i = array.size() - limit/2; i < array.size(); i++) {
            std::cout << array[i];
            if (i < array.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    } else {
        std::cout << "[";
        for (size_t i = 0; i < array.size(); i++) {
            std::cout << array[i];
            if (i < array.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}