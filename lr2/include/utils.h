#ifndef UTILS_H
#define UTILS_H

#include <chrono>
#include <cstdint>

inline uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

#endif // UTILS_H
