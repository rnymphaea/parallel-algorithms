#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <chrono>
#include <sys/stat.h>
#include "gpu_sorter.hpp"
#include "cpu_sorter.hpp"
#include "utils.hpp"

struct BenchmarkConfig {
    // Размеры массивов для тестирования
    std::vector<int> array_sizes = {1000, 5000, 10000, 50000, 100000, 500000, 1000000};
    
    // Конфигурации CPU
    std::vector<int> cpu_threads = {1, 2, 4, 8, 16};
    
    // Конфигурации GPU - только оптимизированная версия
    std::vector<size_t> gpu_work_groups = {8, 16, 32, 64, 128, 256}; //группа потоков, выполняющихся вместе
    //Work-item = отдельный поток выполнения
    
    // Количество прогонов для каждого теста
    int runs_per_test = 2;
    
    // Минимальный размер массива для многопоточности CPU
    int min_size_for_threading = 1000;
};

class BenchmarkRunner {
private:
    std::ofstream csv_file;
    BenchmarkConfig config;
    
    std::string get_results_path(const std::string& filename) {
        std::vector<std::string> possible_paths = {
            "../results/" + filename,
            "../../results/" + filename, 
            "results/" + filename,
            "./results/" + filename
        };
        
        for (const auto& path : possible_paths) {
            std::string dir_path = path.substr(0, path.find_last_of('/'));
            struct stat info;
            if (stat(dir_path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR)) {
                return path;
            }
        }
        
        system("mkdir -p results");
        return "results/" + filename;
    }
    
public:
    BenchmarkRunner(const BenchmarkConfig& cfg = BenchmarkConfig()) : config(cfg) {
        std::string csv_path = get_results_path("benchmark_results.csv");
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
    
    void run_cpu_benchmark(int size, int run_id) {
        auto original_data = generate_random_array(size);
        
        for (int num_threads : config.cpu_threads) {
            if (size < config.min_size_for_threading && num_threads > 1) {
                continue;
            }
            
            auto data = original_data;
            CPUConfig cpu_config;
            cpu_config.num_threads = num_threads;
            
            double time = CPUSorter::sort_with_profiling(data, cpu_config);
            bool correct = is_sorted(data);
            
            std::cout << "  CPU Threads=" << num_threads << ": " 
                     << std::fixed << std::setprecision(4) << time << "s "
                     << (correct ? "✓" : "✗") << std::endl;
            
            if (csv_file.is_open()) {
                csv_file << size << ",CPU," << num_threads << " threads," 
                        << time << "," << correct << ",0\n";
            }
        }
        
        // std::sort
        auto data_std = original_data;
        CPUConfig std_config;
        std_config.num_threads = 1;
        std_config.use_std_sort = true;
        double std_time = CPUSorter::sort_with_profiling(data_std, std_config);
        bool std_correct = is_sorted(data_std);
        
        std::cout << "  CPU std::sort: " << std::fixed << std::setprecision(4) 
                 << std_time << "s " << (std_correct ? "✓" : "✗") << std::endl;
        
        if (csv_file.is_open()) {
            csv_file << size << ",CPU,std::sort," << std_time << "," << std_correct << ",0\n";
        }
    }
    
    void run_gpu_benchmark(int size, int run_id) {
        auto original_data = generate_random_array(size);
        GPUMergeSorter gpu_sorter;
        
        // Только оптимизированная версия
        for (size_t work_group_size : config.gpu_work_groups) {
            auto data = original_data;
            GPUConfig gpu_config;
            gpu_config.work_group_size = work_group_size;
            
            double time = gpu_sorter.sort_with_profiling(data, gpu_config);
            bool correct = is_sorted(data);
            
            std::cout << "  GPU WG=" << work_group_size << ": " 
                     << std::fixed << std::setprecision(4) << time << "s "
                     << (correct ? "✓" : "✗") << std::endl;
            
            if (csv_file.is_open()) {
                csv_file << size << ",GPU,WG" << work_group_size << "," 
                        << time << "," << correct << ",0\n";
            }
        }
    }
    
    void run_single_benchmark(int size, int run_id) {
        if (run_id == 0) {
            std::cout << "\nBenchmarking size: " << size << std::endl;
            std::cout << "----------------------------------------" << std::endl;
        } else {
            std::cout << "Run " << (run_id + 1) << ":" << std::endl;
        }
        
        std::cout << "CPU implementations:" << std::endl;
        run_cpu_benchmark(size, run_id);
        
        std::cout << "GPU implementations:" << std::endl;
        run_gpu_benchmark(size, run_id);
    }
    
    void run_comprehensive_benchmark() {
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
        
        // Информация о GPU
        try {
            GPUMergeSorter gpu_sorter;
            std::cout << "GPU Device Info:" << std::endl;
            std::cout << gpu_sorter.get_device_info() << std::endl << std::endl;
        } catch (const std::exception& e) {
            std::cout << "GPU not available: " << e.what() << std::endl;
        }
        
        for (int size : config.array_sizes) {
            for (int run = 0; run < config.runs_per_test; run++) {
                run_single_benchmark(size, run);
            }
        }
        
        std::cout << "=== Benchmark Complete ===" << std::endl;
    }
};

int main() {
    try {
        BenchmarkConfig config;
        
        // Оптимальная конфигурация на основе ваших результатов
        config.array_sizes = {1000, 5000, 10000, 100000, 500000, 1000000,10000000,30000000};
        config.cpu_threads = {4, 8};
        config.gpu_work_groups = {32, 64, 128, 256};  // Лучшие размеры
        config.runs_per_test = 2;
        
        std::cout << "Starting GPU Merge Sort Benchmark (Optimized Version Only)" << std::endl;
        
        BenchmarkRunner runner(config);
        runner.run_comprehensive_benchmark();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}