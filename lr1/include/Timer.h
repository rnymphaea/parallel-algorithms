#ifndef TIMER_H 
#define TIMER_H 

#include <chrono>
#include <functional>

class Timer {
public:
    template <typename Func>
    static double measureAverageTime(Func f, size_t repeats = 3) {
        double total = 0.0;
        for (size_t i = 0; i < repeats; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            f();
            auto end = std::chrono::high_resolution_clock::now();
            total += std::chrono::duration<double>(end - start).count();
        }
        return total / repeats;
    }
};

#endif //TIMER_H
