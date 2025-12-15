#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <cstdlib>
#include "SorterGpu.hpp"
#include "SorterCpu.hpp"
#include "SortUtils.hpp"

struct BenchmarkConfig {
    std::vector<int> array_sizes = {1000, 10000, 100000, 1000000, 10000000};
    std::vector<int> cpu_threads = {1, 2, 4, 8};
    std::vector<size_t> gpu_work_groups = {32, 64, 128, 256};
    int runs_per_test = 1;
    int min_size_for_threading = 1000;
};

class BenchmarkRunner {
private:
    std::ofstream csv_file;
    BenchmarkConfig config;
    
    void printTableHeader() {
        std::cout << std::string(55, '-') << std::endl;
        std::cout << std::left 
                  << std::setw(12) << "Size" 
                  << std::setw(15) << "Implementation" 
                  << std::setw(12) << "Time (s)" 
                  << std::setw(8) << "Status" 
                  << std::endl;
        std::cout << std::string(55, '-') << std::endl;
    }
    
    void printComparisonHeader() {
        std::cout << std::string(65, '-') << std::endl;
        std::cout << std::left 
                  << std::setw(12) << "Size" 
                  << std::setw(12) << "Best CPU" 
                  << std::setw(12) << "Best GPU" 
                  << std::setw(12) << "Speedup" 
                  << std::setw(12) << "Status" 
                  << std::endl;
        std::cout << std::string(65, '-') << std::endl;
    }
    
    void printResult(const std::string& size_str, const std::string& impl, double time, bool correct) {
        std::cout << std::left 
                  << std::setw(12) << size_str
                  << std::setw(15) << impl 
                  << std::setw(12) << std::fixed << std::setprecision(6) << time
                  << std::setw(8) << (correct ? "OK" : "FAIL") 
                  << std::endl;
    }
    
    void printComparisonResult(const std::string& size_str, double best_cpu_time, double best_gpu_time, double speedup) {
        std::string status = speedup > 1.0 ? "GPU FASTER" : "CPU FASTER";
        std::string speedup_str = std::to_string(speedup);
        size_t dot_pos = speedup_str.find('.');
        if (dot_pos != std::string::npos && speedup_str.length() > dot_pos + 3) {
            speedup_str = speedup_str.substr(0, dot_pos + 3);
        }
        
        std::cout << std::left 
                  << std::setw(12) << size_str
                  << std::setw(12) << std::fixed << std::setprecision(6) << best_cpu_time
                  << std::setw(12) << std::fixed << std::setprecision(6) << best_gpu_time
                  << std::setw(12) << speedup_str + "x"
                  << std::setw(12) << status 
                  << std::endl;
    }
    
    std::string formatSize(int size) {
        if (size >= 1000000) {
            return std::to_string(size/1000000) + "M";
        } else if (size >= 1000) {
            return std::to_string(size/1000) + "K";
        } else {
            return std::to_string(size);
        }
    }
    
public:
    BenchmarkRunner(const BenchmarkConfig& cfg = BenchmarkConfig()) : config(cfg) {
        std::string csv_path = "sort_benchmarks.csv";
        csv_file.open(csv_path);
        
        if (!csv_file.is_open()) {
            std::cerr << "ERROR: Cannot open CSV file: " << csv_path << std::endl;
            return;
        }
        
        csv_file << "ArraySize,Implementation,Config,Time,Correct,Speedup\n";
    }
    
    ~BenchmarkRunner() {
        if (csv_file.is_open()) {
            csv_file.close();
        }
    }
    
    void runCpuBenchmark(int size, std::vector<double>& cpu_times) {
        auto original_data = generateRandomArray(size);
        
        for (int num_threads : config.cpu_threads) {
            if (size < config.min_size_for_threading && num_threads > 1) {
                continue;
            }
            
            auto data = original_data;
            CpuConfig cpu_config;
            cpu_config.num_threads = num_threads;
            
            double time = SorterCpu::sortWithProfiling(data, cpu_config);
            bool correct = isSorted(data);
            
            cpu_times.push_back(time);
            
            std::string impl_name = "CPU " + std::to_string(num_threads) + " thr";
            printResult(formatSize(size), impl_name, time, correct);
            
            if (csv_file.is_open()) {
                csv_file << size << ",CPU," << num_threads << " threads," 
                        << time << "," << correct << ",0\n";
            }
        }
        
        auto data_std = original_data;
        CpuConfig std_config;
        std_config.num_threads = 1;
        std_config.use_std_sort = true;
        double std_time = SorterCpu::sortWithProfiling(data_std, std_config);
        bool std_correct = isSorted(data_std);
        
        cpu_times.push_back(std_time);
        printResult(formatSize(size), "std::sort", std_time, std_correct);
        
        if (csv_file.is_open()) {
            csv_file << size << ",CPU,std::sort," << std_time << "," << std_correct << ",0\n";
        }
    }
    
    void runGpuBenchmark(int size, std::vector<double>& gpu_times) {
        auto original_data = generateRandomArray(size);
        SorterGpu gpu_sorter;
        
        for (size_t work_group_size : config.gpu_work_groups) {
            auto data = original_data;
            GpuConfig gpu_config;
            gpu_config.work_group_size = work_group_size;
            
            double time = gpu_sorter.sortWithProfiling(data, gpu_config);
            bool correct = isSorted(data);
            
            gpu_times.push_back(time);
            
            std::string impl_name = "GPU WG" + std::to_string(work_group_size);
            printResult(formatSize(size), impl_name, time, correct);
            
            if (csv_file.is_open()) {
                csv_file << size << ",GPU,WG" << work_group_size << "," 
                        << time << "," << correct << ",0\n";
            }
        }
    }
    
    void runComprehensiveBenchmark() {
        std::cout << "\nSORTING ALGORITHMS BENCHMARK\n";
        std::cout << "=============================\n";
        
        try {
            SorterGpu gpu_sorter;
            std::cout << gpu_sorter.getDeviceInfo() << std::endl;
        } catch (const std::exception& e) {
            std::cout << "GPU not available: " << e.what() << std::endl;
        }
        
        std::vector<double> best_cpu_times;
        std::vector<double> best_gpu_times;
        
        for (int size : config.array_sizes) {
            std::cout << "\nArray Size: " << formatSize(size) << "\n";
            printTableHeader();
            
            std::vector<double> cpu_times;
            std::vector<double> gpu_times;
            
            runCpuBenchmark(size, cpu_times);
            runGpuBenchmark(size, gpu_times);
            
            if (!cpu_times.empty() && !gpu_times.empty()) {
                double best_cpu = *std::min_element(cpu_times.begin(), cpu_times.end());
                double best_gpu = *std::min_element(gpu_times.begin(), gpu_times.end());
                best_cpu_times.push_back(best_cpu);
                best_gpu_times.push_back(best_gpu);
            }
        }
        
        if (!best_cpu_times.empty() && !best_gpu_times.empty()) {
            std::cout << "\nCPU vs GPU COMPARISON\n";
            std::cout << "=====================\n";
            printComparisonHeader();
            
            for (size_t i = 0; i < config.array_sizes.size(); i++) {
                double speedup = best_cpu_times[i] / best_gpu_times[i];
                printComparisonResult(formatSize(config.array_sizes[i]), 
                                    best_cpu_times[i], best_gpu_times[i], speedup);
            }
            
            std::cout << std::string(65, '-') << std::endl;
        }
        
        std::cout << "\nResults saved to sort_benchmarks.csv\n";
    }
};

int main() {
    std::cout << "\nSORTING BENCHMARK: CPU vs GPU\n";
    std::cout << "==============================\n";
    
    try {
        BenchmarkConfig config;
        config.array_sizes = {1000, 10000, 100000, 1000000, 10000000};
        config.cpu_threads = {1, 2, 4, 8};
        config.gpu_work_groups = {32, 64, 128, 256};
        config.runs_per_test = 1;
        
        BenchmarkRunner runner(config);
        runner.runComprehensiveBenchmark();
        
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
