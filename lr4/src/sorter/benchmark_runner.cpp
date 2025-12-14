#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <chrono>
#include <sys/stat.h>
#include "../include/sorter/gpu.hpp"
#include "../include/sorter/cpu.hpp"
#include "../include/sorter/utils.hpp"

struct BenchmarkConfig {
    std::vector<int> array_sizes = {1000, 5000, 10000, 100000, 500000, 1000000, 10000000, 30000000};
    std::vector<int> cpu_threads = {4, 8};
    std::vector<size_t> gpu_work_groups = {32, 64, 128, 256};
    int runs_per_test = 2;
    int min_size_for_threading = 1000;
};

class BenchmarkRunner {
private:
    std::ofstream csv_file;
    BenchmarkConfig config;
    
    std::string getResultsPath(const std::string& filename) {
        system("mkdir -p results");
        return "results/" + filename;
    }
    
public:
    BenchmarkRunner(const BenchmarkConfig& cfg = BenchmarkConfig()) : config(cfg) {
        std::string csv_path = getResultsPath("sort_benchmarks.csv");
        csv_file.open(csv_path);
        
        if (!csv_file.is_open()) {
            std::cerr << "ERROR: Cannot open CSV file: " << csv_path << std::endl;
            return;
        }
        
        csv_file << "ArraySize,Implementation,Config,Time,Correct,Speedup\n";
        std::cout << "Results will be saved to: " << csv_path << std::endl;
    }
    
    ~BenchmarkRunner() {
        if (csv_file.is_open()) {
            csv_file.close();
            std::cout << "Results saved successfully!" << std::endl;
        }
    }
    
    void runCpuBenchmark(int size, int run_id) {
        auto original_data = generateRandomArray(size);
        
        for (int num_threads : config.cpu_threads) {
            if (size < config.min_size_for_threading && num_threads > 1) {
                continue;
            }
            
            auto data = original_data;
            CPUConfig cpu_config;
            cpu_config.num_threads = num_threads;
            
            double time = SorterCPU::sortWithProfiling(data, cpu_config);
            bool correct = isSorted(data);
            
            std::cout << "  CPU Threads=" << num_threads << ": " 
                     << std::fixed << std::setprecision(4) << time << "s "
                     << (correct ? "✓" : "✗") << std::endl;
            
            if (csv_file.is_open()) {
                csv_file << size << ",CPU," << num_threads << " threads," 
                        << time << "," << correct << ",0\n";
            }
        }
        
        auto data_std = original_data;
        CPUConfig std_config;
        std_config.num_threads = 1;
        std_config.use_std_sort = true;
        double std_time = SorterCPU::sortWithProfiling(data_std, std_config);
        bool std_correct = isSorted(data_std);
        
        std::cout << "  CPU std::sort: " << std::fixed << std::setprecision(4) 
                 << std_time << "s " << (std_correct ? "✓" : "✗") << std::endl;
        
        if (csv_file.is_open()) {
            csv_file << size << ",CPU,std::sort," << std_time << "," << std_correct << ",0\n";
        }
    }
    
    void runGpuBenchmark(int size, int run_id) {
        auto original_data = generateRandomArray(size);
        SorterGPU gpu_sorter;
        
        for (size_t work_group_size : config.gpu_work_groups) {
            auto data = original_data;
            GPUConfig gpu_config;
            gpu_config.work_group_size = work_group_size;
            
            double time = gpu_sorter.sortWithProfiling(data, gpu_config);
            bool correct = isSorted(data);
            
            std::cout << "  GPU WG=" << work_group_size << ": " 
                     << std::fixed << std::setprecision(4) << time << "s "
                     << (correct ? "✓" : "✗") << std::endl;
            
            if (csv_file.is_open()) {
                csv_file << size << ",GPU,WG" << work_group_size << "," 
                        << time << "," << correct << ",0\n";
            }
        }
    }
    
    void runSingleBenchmark(int size, int run_id) {
        if (run_id == 0) {
            std::cout << "\nBenchmarking size: " << size << std::endl;
            std::cout << "----------------------------------------" << std::endl;
        } else {
            std::cout << "Run " << (run_id + 1) << ":" << std::endl;
        }
        
        std::cout << "CPU implementations:" << std::endl;
        runCpuBenchmark(size, run_id);
        
        std::cout << "GPU implementations:" << std::endl;
        runGpuBenchmark(size, run_id);
    }
    
    void runComprehensiveBenchmark() {
        std::cout << "=== GPU Merge Sort Benchmark ===" << std::endl;
        std::cout << "Configuration:" << std::endl;
        std::cout << "  CPU threads: ";
        for (int t : config.cpu_threads) std::cout << t << " ";
        std::cout << std::endl;
        std::cout << "  GPU work groups: ";
        for (size_t wg : config.gpu_work_groups) std::cout << wg << " ";
        std::cout << std::endl;
        std::cout << "  Array sizes: ";
        for (int s : config.array_sizes) std::cout << s << " ";
        std::cout << std::endl;
        std::cout << "  Runs per test: " << config.runs_per_test << std::endl;
        std::cout << std::endl;
        
        try {
            SorterGPU gpu_sorter;
            std::cout << "GPU Device Info:" << std::endl;
            std::cout << gpu_sorter.getDeviceInfo() << std::endl << std::endl;
        } catch (const std::exception& e) {
            std::cout << "GPU not available: " << e.what() << std::endl;
        }
        
        for (int size : config.array_sizes) {
            for (int run = 0; run < config.runs_per_test; run++) {
                runSingleBenchmark(size, run);
            }
        }
        
        std::cout << "=== Benchmark Complete ===" << std::endl;
    }
};

int main() {
    try {
        BenchmarkConfig config;
        config.array_sizes = {1000, 5000, 10000, 100000, 500000, 1000000, 10000000, 30000000};
        config.cpu_threads = {4, 8};
        config.gpu_work_groups = {32, 64, 128, 256};
        config.runs_per_test = 2;
        
        std::cout << "Starting GPU Merge Sort Benchmark" << std::endl;
        
        BenchmarkRunner runner(config);
        runner.runComprehensiveBenchmark();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
