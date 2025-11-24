#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <atomic>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include "list_fine.h"
#include "list_coarse.h"
#include "utils.h"

// Bench configuration tuned to favor fine-grained (lazy) list:
// - high fraction of find (reads) because find is lock-free
// - large key range to reduce hot-spot contention
// - moderate ops_per_thread repeated for stability

struct BenchConfig {
    int threads;
    int ops_per_thread;
    double p_insert; // fraction of insert ops
    double p_remove; // fraction of remove ops
    int key_range;
};

template <typename ListType, typename... Args>
double run_once(BenchConfig cfg, const std::string &label, Args&&... args) {
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
    std::cout << label << ": threads=" << cfg.threads
              << " ops=" << total_ops << " time=" << secs << "s"
              << " ops/s=" << ops_per_sec << std::endl;
    return ops_per_sec;
}

int main(int argc, char** argv) {
    std::vector<int> thread_counts = {1, 2, 4, 8};
    BenchConfig cfg;
    cfg.ops_per_thread = 100000; // increase for stable measurement
    cfg.p_insert = 0.1;
    cfg.p_remove = 0.1; // find = 0.90
    cfg.key_range = 4000000; // large key range to reduce collisions

    std::ofstream csv("results.csv");
    csv << "impl,threads,ops_per_sec,p_insert,p_remove,ops_per_thread\n";

    const int repeats = 3;
    for (int t : thread_counts) {
        cfg.threads = t;
        // Coarse
        double avg_coarse = 0;
        for (int r = 0; r < repeats; ++r) avg_coarse += run_once<CoarseList>(cfg, "coarse");
        avg_coarse /= repeats;
        csv << "coarse," << t << "," << avg_coarse << "," << cfg.p_insert << "," << cfg.p_remove << "," << cfg.ops_per_thread << "\n";

        // Fine
        double avg_fine = 0;
        for (int r = 0; r < repeats; ++r) avg_fine += run_once<FineList>(cfg, "fine");
        avg_fine /= repeats;
        csv << "fine," << t << "," << avg_fine << "," << cfg.p_insert << "," << cfg.p_remove << "," << cfg.ops_per_thread << "\n";
    }

    csv.close();
    std::cout << "Results written to results.csv\n";
    return 0;
}

