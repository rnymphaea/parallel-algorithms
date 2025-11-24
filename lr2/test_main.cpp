#include <iostream>
#include "test_correctness.h"
#include "test_invariants.h"

int main() {
    std::cout << "=== Linked List Correctness Test Suite ===\n\n";
    
    bool tests_passed = TestCorrectness::run_all_tests();
    bool invariants_preserved = TestInvariants::run_all_invariant_checks();
    
    if (tests_passed && invariants_preserved) {
        std::cout << "\nALL TESTS AND INVARIANT CHECKS PASSED!\n";
        return 0;
    } else {
        std::cout << "\nSOME TESTS FAILED!\n";
        return 1;
    }
}
