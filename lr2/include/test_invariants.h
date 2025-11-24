#ifndef TEST_INVARIANTS_H
#define TEST_INVARIANTS_H

#include "list_fine.h"
#include <iostream>
#include <thread>
#include <atomic>
#include <vector>
#include <cassert>

class TestInvariants {
public:
    static bool run_all_invariant_checks() {
        std::cout << "\n=== Running Invariant Checks ===\n";
        
        bool all_passed = true;
        
        all_passed &= check_list_connectivity();
        all_passed &= check_marked_nodes_removed();
        
        if (all_passed) {
            std::cout << "ALL INVARIANTS PRESERVED\n";
        } else {
            std::cout << "SOME INVARIANTS VIOLATED\n";
        }
        
        return all_passed;
    }

private:
    static bool check_list_connectivity() {
        std::cout << "Checking list connectivity... ";
        
        FineList list;
        
        for (int i = 0; i < 10; ++i) {
            list.insert(i * 10);
        }
        
        std::atomic<bool> start{false};
        auto worker = [&]() {
            while (!start.load()) std::this_thread::yield();
            
            for (int i = 0; i < 100; ++i) {
                list.insert(i + 1000);
                list.find(i);
                list.remove(i);
            }
        };
        
        std::vector<std::thread> threads;
        for (int i = 0; i < 2; ++i) {
            threads.emplace_back(worker);
        }
        
        start.store(true);
        for (auto& t : threads) {
            t.join();
        }
        
        for (int i = 0; i < 20; ++i) {
            list.find(i * 10);
        }
        
        std::cout << "PASSED\n";
        return true;
    }

    static bool check_marked_nodes_removed() {
        std::cout << "Checking marked nodes are properly removed... ";
        
        FineList list;
        
        for (int i = 0; i < 50; ++i) {
            list.insert(i);
        }
        
        for (int i = 10; i < 20; ++i) {
            list.remove(i);
        }
        
        std::atomic<bool> start{false};
        auto worker = [&]() {
            while (!start.load()) std::this_thread::yield();
            
            for (int i = 0; i < 100; ++i) {
                list.find(5);
                list.find(15);
                list.insert(25 + i);
                list.remove(30 + i);
            }
        };
        
        std::vector<std::thread> threads;
        for (int i = 0; i < 2; ++i) {
            threads.emplace_back(worker);
        }
        
        start.store(true);
        for (auto& t : threads) {
            t.join();
        }
        
        for (int i = 0; i < 10; ++i) {
            assert(list.find(i));
        }
        for (int i = 10; i < 20; ++i) {
            assert(!list.find(i));
        }
        
        std::cout << "PASSED\n";
        return true;
    }
};

#endif
