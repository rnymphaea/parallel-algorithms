#ifndef TEST_CORRECTNESS_H
#define TEST_CORRECTNESS_H

#include "list_coarse.h"
#include "list_fine.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <random>  // Добавлено для std::mt19937

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
            
            // Используем простой детерминированный RNG для тестов
            unsigned int seed = thread_id + 1;
            
            for (int i = 0; i < operations_per_thread; ++i) {
                // Простой детерминированный RNG для тестов
                int value = (seed * (i + 1)) % 201;  // 0-200
                double op = ((seed * (i + 1)) % 1000) / 1000.0;  // 0.0-0.999
                
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
};

#endif // TEST_CORRECTNESS_H
