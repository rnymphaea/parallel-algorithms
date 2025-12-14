#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include "gpu_sorter.hpp"
#include "cpu_sorter.hpp"
#include "utils.hpp"

class CorrectnessTester {
private:
    std::ofstream csv_file;
    
public:
    CorrectnessTester() {
        // Открываем CSV файл для записи результатов
        csv_file.open("../results/correctness_results.csv");
        csv_file << "TestID,ArraySize,DataType,GPUCorrect,CPUSingleCorrect,CPUParallelCorrect,CPUStdCorrect,AllCorrect\n";
    }
    
    ~CorrectnessTester() {
        if (csv_file.is_open()) {
            csv_file.close();
        }
    }
    
    void test_random_array(int size, int test_id) {
        std::cout << "Test " << test_id << ": Random array, size " << size << std::endl;
        
        auto original_data = generate_random_array(size);
        auto data_gpu = original_data;
        auto data_cpu_single = original_data;
        auto data_cpu_parallel = original_data;
        auto data_cpu_std = original_data;
        
        // GPU сортировка
        GPUMergeSorter gpu_sorter;
        gpu_sorter.sort(data_gpu);
        bool gpu_correct = is_sorted(data_gpu);
        
        // CPU сортировки
        CPUSorter::merge_sort(data_cpu_single);
        bool cpu_single_correct = is_sorted(data_cpu_single);
        
        CPUSorter::parallel_merge_sort(data_cpu_parallel);
        bool cpu_parallel_correct = is_sorted(data_cpu_parallel);
        
        CPUSorter::std_sort(data_cpu_std);
        bool cpu_std_correct = is_sorted(data_cpu_std);
        
        bool all_correct = gpu_correct && cpu_single_correct && cpu_parallel_correct && cpu_std_correct;
        
        // Запись в CSV
        csv_file << test_id << "," << size << ",random,"
                << gpu_correct << "," << cpu_single_correct << ","
                << cpu_parallel_correct << "," << cpu_std_correct << ","
                << all_correct << "\n";
        
        // Вывод результатов
        std::cout << "  GPU: " << (gpu_correct ? "✓" : "✗") << "  "
                  << "CPU Single: " << (cpu_single_correct ? "✓" : "✗") << "  "
                  << "CPU Parallel: " << (cpu_parallel_correct ? "✓" : "✗") << "  "
                  << "CPU STD: " << (cpu_std_correct ? "✓" : "✗") << "  "
                  << "Overall: " << (all_correct ? "PASS" : "FAIL") << std::endl;
        
        if (!all_correct && size <= 20) {
            std::cout << "  Original: ";
            print_array(original_data);
            std::cout << "  GPU:      ";
            print_array(data_gpu);
            std::cout << "  CPU STD:  ";
            print_array(data_cpu_std);
        }
    }
    
    void test_sorted_array(int size, int test_id) {
        std::cout << "Test " << test_id << ": Already sorted array, size " << size << std::endl;
        
        std::vector<int> original_data(size);
        for (int i = 0; i < size; i++) {
            original_data[i] = i;
        }
        
        auto data_gpu = original_data;
        auto data_cpu_single = original_data;
        auto data_cpu_parallel = original_data;
        auto data_cpu_std = original_data;
        
        GPUMergeSorter gpu_sorter;
        gpu_sorter.sort(data_gpu);
        bool gpu_correct = is_sorted(data_gpu);
        
        CPUSorter::merge_sort(data_cpu_single);
        CPUSorter::parallel_merge_sort(data_cpu_parallel);
        CPUSorter::std_sort(data_cpu_std);
        
        bool all_correct = gpu_correct;
        
        csv_file << test_id << "," << size << ",sorted,"
                << gpu_correct << ",1,1,1,"
                << all_correct << "\n";
        
        std::cout << "  GPU: " << (gpu_correct ? "✓" : "✗") 
                  << "  Overall: " << (all_correct ? "PASS" : "FAIL") << std::endl;
    }
    
    void test_reverse_sorted_array(int size, int test_id) {
        std::cout << "Test " << test_id << ": Reverse sorted array, size " << size << std::endl;
        
        std::vector<int> original_data(size);
        for (int i = 0; i < size; i++) {
            original_data[i] = size - i - 1;
        }
        
        auto data_gpu = original_data;
        auto data_cpu_single = original_data;
        auto data_cpu_parallel = original_data;
        auto data_cpu_std = original_data;
        
        GPUMergeSorter gpu_sorter;
        gpu_sorter.sort(data_gpu);
        bool gpu_correct = is_sorted(data_gpu);
        
        CPUSorter::merge_sort(data_cpu_single);
        bool cpu_single_correct = is_sorted(data_cpu_single);
        
        CPUSorter::parallel_merge_sort(data_cpu_parallel);
        bool cpu_parallel_correct = is_sorted(data_cpu_parallel);
        
        CPUSorter::std_sort(data_cpu_std);
        bool cpu_std_correct = is_sorted(data_cpu_std);
        
        bool all_correct = gpu_correct && cpu_single_correct && cpu_parallel_correct && cpu_std_correct;
        
        csv_file << test_id << "," << size << ",reverse_sorted,"
                << gpu_correct << "," << cpu_single_correct << ","
                << cpu_parallel_correct << "," << cpu_std_correct << ","
                << all_correct << "\n";
        
        std::cout << "  GPU: " << (gpu_correct ? "✓" : "✗") << "  "
                  << "Overall: " << (all_correct ? "PASS" : "FAIL") << std::endl;
    }
    
    void run_all_tests() {
        std::cout << "=== Correctness Tests ===" << std::endl;
        
        int test_id = 1;
        
        // Маленькие массивы
        test_random_array(10, test_id++);
        test_random_array(16, test_id++);
        test_random_array(32, test_id++);
        
        // Уже отсортированные
        test_sorted_array(100, test_id++);
        test_reverse_sorted_array(100, test_id++);
        
        // Средние массивы
        test_random_array(1000, test_id++);
        test_random_array(5000, test_id++);
        
        // Большие массивы
        test_random_array(10000, test_id++);
        test_random_array(50000, test_id++);
        
        std::cout << "=== Correctness Tests Complete ===" << std::endl;
        std::cout << "Results saved to results/correctness_results.csv" << std::endl;
    }
};

int main() {
    try {
        CorrectnessTester tester;
        tester.run_all_tests();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}