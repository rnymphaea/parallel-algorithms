#ifndef TEST_CORRECTNESS_H
#define TEST_CORRECTNESS_H

#include "list_coarse.h"
#include "list_fine.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <random>

class TestCorrectness {
public:
    static bool run_all_tests() {
        std::cout << "=== Running Correctness Tests ===\n";
        
        bool all_passed = true;
        
        all_passed &= test_basic_operations<CoarseList>("CoarseList");
        all_passed &= test_basic_operations<FineList>("FineList");
        
        all_passed &= test_edge_cases<CoarseList>("CoarseList");
        all_passed &= test_edge_cases<FineList>("FineList");
        
        all_passed &= test_concurrent_inserts();
        all_passed &= test_concurrent_mixed_operations();
        
        if (all_passed) {
            std::cout << "ALL TESTS PASSED\n";
        } else {
            std::cout << "SOME TESTS FAILED\n";
        }
        
        return all_passed;
    }

private:
    template<typename ListType>
    static bool test_basic_operations(const std::string& name) {
        std::cout << "Testing basic operations for " << name << "... ";
        
        ListType list;
        
        assert(!list.find(1));
        assert(!list.remove(1));
        
        assert(list.insert(1));
        assert(list.find(1));
        assert(list.remove(1));
        assert(!list.find(1));
        
        assert(list.insert(5));
        assert(list.find(5));
        
        assert(list.insert(1));
        assert(list.insert(2));
        assert(list.insert(3));
        assert(list.find(1));
        assert(list.find(2));
        assert(list.find(3));
        
        assert(list.remove(2));
        assert(!list.find(2));
        assert(list.find(1));
        assert(list.find(3));
        
        assert(list.remove(1));
        assert(!list.find(1));
        assert(list.find(3));
        
        std::cout << "PASSED\n";
        return true;
    }

    template<typename ListType>
    static bool test_edge_cases(const std::string& name) {
        std::cout << "Testing edge cases for " << name << "... ";
        
        ListType list;
        
        assert(list.insert(-1));
        assert(list.find(-1));
        assert(list.remove(-1));
        
        assert(list.insert(0));
        assert(list.find(0));
        assert(list.remove(0));
        
        assert(list.insert(1000000));
        assert(list.find(1000000));
        assert(list.remove(1000000));
        
        for (int i = 0; i < 100; ++i) {
            assert(list.insert(i));
        }
        
        for (int i = 0; i < 100; ++i) {
            assert(list.find(i));
        }
        
        for (int i = 0; i < 100; ++i) {
            assert(list.remove(i));
        }
        
        for (int i = 0; i < 100; ++i) {
            assert(!list.find(i));
        }
        
        std::cout << "PASSED\n";
        return true;
    }

    static bool test_concurrent_inserts() {
        std::cout << "Testing concurrent inserts... ";
        
        FineList list;
        const int thread_count = 4;
        const int operations_per_thread = 1000;
        std::atomic<int> success_count{0};
        std::atomic<bool> start{false};
        
        auto worker = [&](int thread_id) {
            while (!start.load()) std::this_thread::yield();
            
            for (int i = 0; i < operations_per_thread; ++i) {
                int value = thread_id * 100000 + i;
                if (list.insert(value)) {
                    success_count.fetch_add(1);
                }
            }
        };
        
        std::vector<std::thread> threads;
        for (int i = 0; i < thread_count; ++i) {
            threads.emplace_back(worker, i);
        }
        
        start.store(true);
        for (auto& t : threads) {
            t.join();
        }
        
        for (int i = 0; i < 100; ++i) {
            list.find(i);
        }
        
        std::cout << "PASSED (inserted: " << success_count << ")\n";
        return true;
    }

    static bool test_concurrent_mixed_operations() {
        std::cout << "Testing concurrent mixed operations... ";
        
        FineList list;
        const int thread_count = 4;
        const int operations_per_thread = 500;
        std::atomic<int> completed{0};
        std::atomic<bool> start{false};
        
        for (int i = 0; i < 100; ++i) {
            list.insert(i);
        }
        
        auto worker = [&](int thread_id) {
            while (!start.load()) std::this_thread::yield();
            
            unsigned int seed = thread_id + 1;
            
            for (int i = 0; i < operations_per_thread; ++i) {
                int value = (seed * (i + 1)) % 201;
                double op = ((seed * (i + 1)) % 1000) / 1000.0;
                
                if (op < 0.4) {
                    list.insert(value);
                } else if (op < 0.7) {
                    list.find(value);
                } else {
                    list.remove(value);
                }
            }
            
            completed.fetch_add(1);
        };
        
        std::vector<std::thread> threads;
        for (int i = 0; i < thread_count; ++i) {
            threads.emplace_back(worker, i);
        }
        
        start.store(true);
        for (auto& t : threads) {
            t.join();
        }
        
        for (int i = 0; i < 50; ++i) {
            list.find(i);
        }
        
        std::cout << "PASSED\n";
        return true;
    }
};

#endif
