#ifndef TEST_CORRECTNESS_H
#define TEST_CORRECTNESS_H

#include "list_coarse.h"
#include "list_fine.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>

class TestCorrectness {
public:
    static bool run_all_tests() {
        std::cout << "=== Running Correctness Tests ===\n";
        
        bool all_passed = true;
        
        // Basic functionality tests
        all_passed &= test_basic_operations<CoarseList>("CoarseList");
        all_passed &= test_basic_operations<FineList>("FineList");
        
        // Edge cases
        all_passed &= test_edge_cases<CoarseList>("CoarseList");
        all_passed &= test_edge_cases<FineList>("FineList");
        
        // Concurrent tests
        all_passed &= test_concurrent_inserts();
        all_passed &= test_concurrent_mixed_operations();
        all_passed &= test_no_duplicates();
        
        if (all_passed) {
            std::cout << "✅ ALL TESTS PASSED\n";
        } else {
            std::cout << "❌ SOME TESTS FAILED\n";
        }
        
        return all_passed;
    }

private:
    template<typename ListType>
    static bool test_basic_operations(const std::string& name) {
        std::cout << "Testing basic operations for " << name << "... ";
        
        ListType list;
        
        // Test empty list
        assert(!list.find(1));
        assert(!list.remove(1));
        
        // Test single insert/find/remove
        assert(list.insert(1));
        assert(list.find(1));
        assert(list.remove(1));
        assert(!list.find(1));
        
        // Test duplicate prevention
        assert(list.insert(5));
        assert(!list.insert(5)); // Should fail
        assert(list.find(5));
        
        // Test multiple elements
        assert(list.insert(1));
        assert(list.insert(2));
        assert(list.insert(3));
        assert(list.find(1));
        assert(list.find(2));
        assert(list.find(3));
        
        // Test middle removal
        assert(list.remove(2));
        assert(!list.find(2));
        assert(list.find(1));
        assert(list.find(3));
        
        // Test head removal
        assert(list.remove(1));
        assert(!list.find(1));
        assert(list.find(3));
        
        std::cout << "✅ PASSED\n";
        return true;
    }

    template<typename ListType>
    static bool test_edge_cases(const std::string& name) {
        std::cout << "Testing edge cases for " << name << "... ";
        
        ListType list;
        
        // Test negative numbers
        assert(list.insert(-1));
        assert(list.find(-1));
        assert(list.remove(-1));
        
        // Test zero
        assert(list.insert(0));
        assert(list.find(0));
        assert(list.remove(0));
        
        // Test large numbers
        assert(list.insert(1000000));
        assert(list.find(1000000));
        assert(list.remove(1000000));
        
        // Test many insertions
        for (int i = 0; i < 100; ++i) {
            assert(list.insert(i));
        }
        
        // Verify all are present
        for (int i = 0; i < 100; ++i) {
            assert(list.find(i));
        }
        
        // Remove all
        for (int i = 0; i < 100; ++i) {
            assert(list.remove(i));
        }
        
        // Verify all are gone
        for (int i = 0; i < 100; ++i) {
            assert(!list.find(i));
        }
        
        std::cout << "✅ PASSED\n";
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
        
        // Verify we can find some of the inserted values without crashing
        for (int i = 0; i < 100; ++i) {
            list.find(i); // Should not crash
        }
        
        std::cout << "✅ PASSED (inserted: " << success_count << ")\n";
        return true;
    }

    static bool test_concurrent_mixed_operations() {
        std::cout << "Testing concurrent mixed operations... ";
        
        FineList list;
        const int thread_count = 4;
        const int operations_per_thread = 500;
        std::atomic<int> completed{0};
        std::atomic<bool> start{false};
        
        // Pre-populate with some values
        for (int i = 0; i < 100; ++i) {
            list.insert(i);
        }
        
        auto worker = [&](int thread_id) {
            while (!start.load()) std::this_thread::yield();
            
            std::mt19937 rng(thread_id);
            std::uniform_int_distribution<int> val_dist(0, 200);
            std::uniform_real_distribution<double> op_dist(0.0, 1.0);
            
            for (int i = 0; i < operations_per_thread; ++i) {
                int value = val_dist(rng);
                double op = op_dist(rng);
                
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
        
        // Final verification - list should still be consistent
        for (int i = 0; i < 50; ++i) {
            list.find(i); // Should not crash
        }
        
        std::cout << "✅ PASSED\n";
        return true;
    }

    static bool test_no_duplicates() {
        std::cout << "Testing no duplicates under concurrency... ";
        
        FineList list;
        const int thread_count = 4;
        const int target_value = 12345;
        std::atomic<int> insert_success_count{0};
        std::atomic<bool> start{false};
        
        auto worker = [&]() {
            while (!start.load()) std::this_thread::yield();
            
            if (list.insert(target_value)) {
                insert_success_count.fetch_add(1);
            }
        };
        
        std::vector<std::thread> threads;
        for (int i = 0; i < thread_count; ++i) {
            threads.emplace_back(worker);
        }
        
        start.store(true);
        for (auto& t : threads) {
            t.join();
        }
        
        // Only one thread should have successfully inserted
        assert(insert_success_count == 1);
        assert(list.find(target_value));
        
        std::cout << "✅ PASSED (successful inserts: " << insert_success_count << ")\n";
        return true;
    }
};

#endif // TEST_CORRECTNESS_H
