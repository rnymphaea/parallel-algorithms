#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <atomic>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <getopt.h>
#include "list_fine.h"
#include "list_coarse.h"
#include "utils.h"

// Bench configuration with command line overrides
struct BenchConfig {
    int threads = 1;
    int ops_per_thread = 100000;
    double p_insert = 0.1;
    double p_remove = 0.1;
    int key_range = 4000000;
    int repeats = 3;
    std::string output_file = "results.csv";
    bool verbose = false;
};

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [OPTIONS]\n"
              << "Benchmark fine-grained vs coarse-grained linked list\n\n"
              << "Options:\n"
              << "  -t, --threads N          Number of threads (default: 1,2,4,8)\n"
              << "  -o, --operations N       Operations per thread (default: 100000)\n"
              << "  -i, --insert RATIO       Insert operation ratio (default: 0.1)\n"
              << "  -r, --remove RATIO       Remove operation ratio (default: 0.1)\n"
              << "  -f, --find RATIO         Find operation ratio (default: 0.8)\n"
              << "  -k, --key-range N        Key range (default: 4000000)\n"
              << "  -n, --repeats N          Number of repeats for averaging (default: 3)\n"
              << "  -O, --output FILE        Output CSV file (default: results.csv)\n"
              << "  -v, --verbose            Verbose output\n"
              << "  -h, --help               Show this help message\n"
              << "\nExamples:\n"
              << "  " << prog_name << " -t 1,4,16 -i 0.2 -r 0.2 -f 0.6\n"
              << "  " << prog_name << " --threads=8 --operations=50000 --insert=0.05\n";
}

std::vector<int> parse_thread_list(const std::string& thread_str) {
    std::vector<int> threads;
    std::stringstream ss(thread_str);
    std::string item;
    
    while (std::getline(ss, item, ',')) {
        threads.push_back(std::stoi(item));
    }
    
    return threads;
}

BenchConfig parse_args(int argc, char** argv) {
    BenchConfig config;
    std::string thread_str = "1,2,4,8"; // default thread counts
    
    static struct option long_options[] = {
        {"threads", required_argument, 0, 't'},
        {"operations", required_argument, 0, 'o'},
        {"insert", required_argument, 0, 'i'},
        {"remove", required_argument, 0, 'r'},
        {"find", required_argument, 0, 'f'},
        {"key-range", required_argument, 0, 'k'},
        {"repeats", required_argument, 0, 'n'},
        {"output", required_argument, 0, 'O'},
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    
    int c;
    while ((c = getopt_long(argc, argv, "t:o:i:r:f:k:n:O:vh", long_options, nullptr)) != -1) {
        switch (c) {
            case 't':
                thread_str = optarg;
                break;
            case 'o':
                config.ops_per_thread = std::stoi(optarg);
                break;
            case 'i':
                config.p_insert = std::stod(optarg);
                break;
            case 'r':
                config.p_remove = std::stod(optarg);
                break;
            case 'f':
                {
                    double find_ratio = std::stod(optarg);
                    // Normalize ratios if all three are provided
                    double total = config.p_insert + config.p_remove + find_ratio;
                    if (total > 1.0) {
                        config.p_insert /= total;
                        config.p_remove /= total;
                        find_ratio /= total;
                    }
                    // Set insert and remove based on find ratio
                    config.p_insert = (1.0 - find_ratio) * 0.5;
                    config.p_remove = (1.0 - find_ratio) * 0.5;
                }
                break;
            case 'k':
                config.key_range = std::stoi(optarg);
                break;
            case 'n':
                config.repeats = std::stoi(optarg);
                break;
            case 'O':
                config.output_file = optarg;
                break;
            case 'v':
                config.verbose = true;
                break;
            case 'h':
                print_usage(argv[0]);
                exit(0);
            default:
                print_usage(argv[0]);
                exit(1);
        }
    }
    
    // Parse thread counts
    config.threads = 0; // Not used directly, thread_counts used instead
    if (config.verbose) {
        std::cout << "Thread counts: " << thread_str << "\n";
    }
    
    return config;
}

template <typename ListType, typename... Args>
double run_once(const BenchConfig& cfg, const std::string &label, Args&&... args) {
    ListType list(std::forward<Args>(args)...);
    std::atomic<int> started{0};

    auto worker = [&](int tid) {
        std::mt19937 rng(tid + 0xC0FFEE);
        std::uniform_int_distribution<int> val_dist(1, cfg.key_range);
        std::uniform_real_distribution<double> op_dist(0.0, 1.0);

        started.fetch_add(1);
        while (started.load() < cfg.threads) std::this_thread::yield();

        for (int i = 0; i < cfg.ops_per_thread / cfg.threads; ++i) {
            double op = op_dist(rng);
            int v = val_dist(rng);
            if (op < cfg.p_insert) {
                list.insert(v);
            } else if (op < cfg.p_insert + cfg.p_remove) {
                list.remove(v);
            } else {
                list.find(v);
            }
        }
    };

    std::vector<std::thread> thr;
    uint64_t t0 = now_ns();
    for (int i = 0; i < cfg.threads; ++i) thr.emplace_back(worker, i);
    for (auto &t : thr) t.join();
    uint64_t t1 = now_ns();
    double secs = double(t1 - t0) / 1e9;
    double total_ops = double(cfg.ops_per_thread);
    double ops_per_sec = total_ops / secs;
    
    if (cfg.verbose) {
        std::cout << label << ": threads=" << cfg.threads
                  << " ops=" << total_ops << " time=" << secs << "s"
                  << " ops/s=" << ops_per_sec << std::endl;
    }
    return ops_per_sec;
}

int main(int argc, char** argv) {
    BenchConfig config = parse_args(argc, argv);
    
    // Parse thread counts from command line or use default
    std::vector<int> thread_counts = parse_thread_list([&]() -> std::string {
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "-t" || std::string(argv[i]) == "--threads") {
                if (i + 1 < argc) return argv[i + 1];
            }
        }
        return "1,2,4,8";
    }());

    if (config.verbose) {
        std::cout << "Benchmark Configuration:\n"
                  << "  Thread counts: ";
        for (size_t i = 0; i < thread_counts.size(); ++i) {
            std::cout << thread_counts[i];
            if (i < thread_counts.size() - 1) std::cout << ", ";
        }
        std::cout << "\n"
                  << "  Operations per thread: " << config.ops_per_thread << "\n"
                  << "  Insert ratio: " << config.p_insert << "\n"
                  << "  Remove ratio: " << config.p_remove << "\n"
                  << "  Find ratio: " << (1.0 - config.p_insert - config.p_remove) << "\n"
                  << "  Key range: " << config.key_range << "\n"
                  << "  Repeats: " << config.repeats << "\n"
                  << "  Output file: " << config.output_file << "\n";
    }

    std::ofstream csv(config.output_file);
    csv << "impl,threads,ops_per_sec,p_insert,p_remove,ops_per_thread,key_range\n";

    for (int t : thread_counts) {
        BenchConfig run_config = config;
        run_config.threads = t;

        if (config.verbose) {
            std::cout << "\nRunning with " << t << " threads...\n";
        }

        // Coarse-grained list
        double avg_coarse = 0;
        for (int r = 0; r < config.repeats; ++r) {
            double result = run_once<CoarseList>(run_config, "coarse");
            avg_coarse += result;
            if (config.verbose) {
                std::cout << "  Coarse run " << (r + 1) << "/" << config.repeats 
                          << ": " << result << " ops/s\n";
            }
        }
        avg_coarse /= config.repeats;
        
        if (config.verbose) {
            std::cout << "  Coarse average: " << avg_coarse << " ops/s\n";
        }
        
        csv << "coarse," << t << "," << avg_coarse << "," 
            << config.p_insert << "," << config.p_remove << "," 
            << config.ops_per_thread << "," << config.key_range << "\n";

        // Fine-grained list
        double avg_fine = 0;
        for (int r = 0; r < config.repeats; ++r) {
            double result = run_once<FineList>(run_config, "fine");
            avg_fine += result;
            if (config.verbose) {
                std::cout << "  Fine run " << (r + 1) << "/" << config.repeats 
                          << ": " << result << " ops/s\n";
            }
        }
        avg_fine /= config.repeats;
        
        if (config.verbose) {
            std::cout << "  Fine average: " << avg_fine << " ops/s\n";
        }
        
        csv << "fine," << t << "," << avg_fine << "," 
            << config.p_insert << "," << config.p_remove << "," 
            << config.ops_per_thread << "," << config.key_range << "\n";
    }

    csv.close();
    std::cout << "Results written to " << config.output_file << "\n";
    return 0;
}
