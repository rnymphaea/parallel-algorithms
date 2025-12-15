#include "../include/Matrix.hpp"
#include "../include/StrassenMultiplier.hpp"
#include "../include/ParallelSort.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <random>
#include <string>
#include <fstream>

// ============================================================================
// УТИЛИТЫ ДЛЯ ТЕСТИРОВАНИЯ
// ============================================================================

class TestTimer {
private:
    std::chrono::high_resolution_clock::time_point start;
    std::string testName;
    
public:
    TestTimer(const std::string& name) : testName(name) {
        start = std::chrono::high_resolution_clock::now();
    }
    
    ~TestTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end - start);
        std::cout << "  Time: " << std::fixed << std::setprecision(4) 
                  << duration.count() << " seconds" << std::endl;
    }
    
    double elapsed() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(now - start).count();
    }
};

void printTestHeader(const std::string& header) {
    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "  " << header << "\n";
    std::cout << "================================================================\n";
}

void printSection(const std::string& section) {
    std::cout << "\n";
    std::cout << "  --- " << section << " ---\n";
}

// ============================================================================
// ТЕСТИРОВАНИЕ УМНОЖЕНИЯ МАТРИЦ
// ============================================================================

void testMatrixMultiplicationCorrectness() {
    printTestHeader("TEST 1: MATRIX MULTIPLICATION CORRECTNESS");
    
    std::vector<size_t> matrixSizes = {32, 64, 128, 256};
    
    for (size_t size : matrixSizes) {
        std::cout << "\n  Testing matrix size: " << size << "x" << size << std::endl;
        
        Matrix A(size, size);
        Matrix B(size, size);
        A.fillRandom();
        B.fillRandom();
        
        // Наивное умножение как эталон
        std::cout << "    Running naive multiplication..." << std::endl;
        TestTimer naiveTimer("Naive");
        Matrix naiveResult = StrassenMultiplier::naiveMultiply(A, B);
        double naiveTime = naiveTimer.elapsed();
        
        // Тестируем разное количество потоков
        std::vector<size_t> threadCounts = {1, 2, 4, 8};
        
        for (size_t threads : threadCounts) {
            std::cout << "    Running Strassen with " << threads << " thread(s)..." << std::endl;
            TestTimer strassenTimer("Strassen " + std::to_string(threads) + " threads");
            
            StrassenMultiplier multiplier(threads);
            Matrix strassenResult = multiplier.multiply(A, B);
            double strassenTime = strassenTimer.elapsed();
            
            // Проверка корректности
            if (naiveResult == strassenResult) {
                std::cout << "    Result: CORRECT" << std::endl;
                std::cout << "    Speedup vs naive: " << std::fixed << std::setprecision(2) 
                          << (naiveTime / strassenTime) << "x" << std::endl;
            } else {
                std::cout << "    Result: INCORRECT" << std::endl;
            }
        }
    }
}

void benchmarkMatrixMultiplication() {
    printTestHeader("TEST 2: MATRIX MULTIPLICATION PERFORMANCE BENCHMARK");
    
    std::vector<size_t> matrixSizes = {128, 256, 512, 1024};
    std::vector<size_t> threadCounts = {1, 2, 4, 8};
    
    for (size_t size : matrixSizes) {
        std::cout << "\n  Matrix size: " << size << "x" << size << std::endl;
        
        Matrix A(size, size);
        Matrix B(size, size);
        A.fillRandom();
        B.fillRandom();
        
        // Базовое время для наивного умножения (только для маленьких матриц)
        if (size <= 256) {
            std::cout << "    Naive multiplication:" << std::endl;
            TestTimer naiveTimer("Naive");
            StrassenMultiplier::naiveMultiply(A, B);
        }
        
        // Тестируем разное количество потоков
        for (size_t threads : threadCounts) {
            std::cout << "    Strassen with " << threads << " thread(s):" << std::endl;
            TestTimer timer("Strassen");
            
            StrassenMultiplier multiplier(threads);
            multiplier.multiply(A, B);
        }
    }
}

void testStrassenScalability() {
    printTestHeader("TEST 3: STRASSEN ALGORITHM SCALABILITY ANALYSIS");
    
    size_t matrixSize = 512;
    std::cout << "\n  Fixed matrix size: " << matrixSize << "x" << matrixSize << std::endl;
    
    Matrix A(matrixSize, matrixSize);
    Matrix B(matrixSize, matrixSize);
    A.fillRandom();
    B.fillRandom();
    
    std::vector<size_t> threadCounts = {1, 2, 4, 8};
    double singleThreadTime = 0.0;
    
    // Сначала получаем время для одного потока
    {
        std::cout << "\n    Single thread baseline:" << std::endl;
        TestTimer timer("1 thread");
        StrassenMultiplier multiplier(1);
        multiplier.multiply(A, B);
        singleThreadTime = timer.elapsed();
    }
    
    std::cout << "\n    Multi-threaded performance:" << std::endl;
    for (size_t threads : threadCounts) {
        if (threads == 1) continue; // Уже измерили
        
        std::cout << "    " << threads << " threads:" << std::endl;
        TestTimer timer(std::to_string(threads) + " threads");
        
        StrassenMultiplier multiplier(threads);
        multiplier.multiply(A, B);
        double time = timer.elapsed();
        
        double speedup = singleThreadTime / time;
        double efficiency = (speedup / threads) * 100;
        
        std::cout << "      Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        std::cout << "      Efficiency: " << std::fixed << std::setprecision(1) << efficiency << "%" << std::endl;
    }
}

// ============================================================================
// ТЕСТИРОВАНИЕ СОРТИРОВКИ
// ============================================================================

void testSortingCorrectness() {
    printTestHeader("TEST 4: SORTING CORRECTNESS TEST");
    
    std::vector<size_t> arraySizes = {1000, 10000, 100000};
    
    for (size_t size : arraySizes) {
        std::cout << "\n  Array size: " << size << " elements" << std::endl;
        
        std::vector<int> array(size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, 1000000);
        
        for (auto& val : array) {
            val = dis(gen);
        }
        
        // Эталонная сортировка std::sort
        std::vector<int> reference = array;
        std::cout << "    Reference sort (std::sort):" << std::endl;
        TestTimer refTimer("std::sort");
        std::sort(reference.begin(), reference.end());
        
        // Тестируем разное количество потоков
        std::vector<size_t> threadCounts = {1, 2, 4, 8};
        
        for (size_t threads : threadCounts) {
            std::vector<int> testArray = array;
            std::cout << "    Parallel sort with " << threads << " thread(s):" << std::endl;
            TestTimer sortTimer("Parallel sort");
            
            ParallelSort sorter(threads);
            sorter.sort(testArray);
            
            // Проверяем корректность
            bool isCorrect = true;
            if (testArray.size() == reference.size()) {
                for (size_t i = 0; i < testArray.size(); i++) {
                    if (testArray[i] != reference[i]) {
                        isCorrect = false;
                        break;
                    }
                }
            } else {
                isCorrect = false;
            }
            
            if (isCorrect && ParallelSort::isSorted(testArray)) {
                std::cout << "    Result: CORRECT" << std::endl;
            } else {
                std::cout << "    Result: INCORRECT" << std::endl;
            }
        }
    }
}

void benchmarkSortingPerformance() {
    printTestHeader("TEST 5: SORTING PERFORMANCE BENCHMARK");
    
    std::vector<size_t> arraySizes = {100000, 500000, 1000000, 5000000};
    std::vector<size_t> threadCounts = {1, 2, 4, 8};
    
    for (size_t size : arraySizes) {
        std::cout << "\n  Array size: " << size << " elements" << std::endl;
        
        std::vector<int> baseArray(size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, 10000000);
        
        for (auto& val : baseArray) {
            val = dis(gen);
        }
        
        double singleThreadTime = 0.0;
        
        // Измеряем производительность для разного количества потоков
        for (size_t threads : threadCounts) {
            std::vector<int> testArray = baseArray;
            std::cout << "    Sorting with " << threads << " thread(s):" << std::endl;
            TestTimer timer(std::to_string(threads) + " threads");
            
            ParallelSort sorter(threads);
            sorter.sort(testArray);
            
            double time = timer.elapsed();
            
            // Сохраняем время для одного потока как базу
            if (threads == 1) {
                singleThreadTime = time;
            } else if (singleThreadTime > 0) {
                double speedup = singleThreadTime / time;
                std::cout << "      Speedup vs single thread: " 
                          << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
            }
            
            // Проверяем что массив отсортирован
            if (!ParallelSort::isSorted(testArray)) {
                std::cout << "      WARNING: Array may not be properly sorted!" << std::endl;
            }
        }
    }
}

void testSortingScalability() {
    printTestHeader("TEST 6: SORTING SCALABILITY ANALYSIS");
    
    size_t arraySize = 1000000;
    std::cout << "\n  Fixed array size: " << arraySize << " elements" << std::endl;
    
    std::vector<int> baseArray(arraySize);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10000000);
    
    for (auto& val : baseArray) {
        val = dis(gen);
    }
    
    std::vector<size_t> threadCounts = {1, 2, 4, 8};
    double singleThreadTime = 0.0;
    
    // Сначала получаем время для одного потока
    {
        std::vector<int> testArray = baseArray;
        std::cout << "\n    Single thread baseline:" << std::endl;
        TestTimer timer("1 thread");
        ParallelSort sorter(1);
        sorter.sort(testArray);
        singleThreadTime = timer.elapsed();
        
        if (!ParallelSort::isSorted(testArray)) {
            std::cout << "    WARNING: Single thread sort may be incorrect!" << std::endl;
        }
    }
    
    std::cout << "\n    Multi-threaded performance:" << std::endl;
    for (size_t threads : threadCounts) {
        if (threads == 1) continue;
        
        std::vector<int> testArray = baseArray;
        std::cout << "    " << threads << " threads:" << std::endl;
        TestTimer timer(std::to_string(threads) + " threads");
        
        ParallelSort sorter(threads);
        sorter.sort(testArray);
        double time = timer.elapsed();
        
        double speedup = singleThreadTime / time;
        double efficiency = (speedup / threads) * 100;
        
        std::cout << "      Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        std::cout << "      Efficiency: " << std::fixed << std::setprecision(1) << efficiency << "%" << std::endl;
        
        if (!ParallelSort::isSorted(testArray)) {
            std::cout << "      WARNING: Array may not be properly sorted!" << std::endl;
        }
    }
}

// ============================================================================
// СРАВНИТЕЛЬНЫЕ ТЕСТЫ
// ============================================================================

void compareAlgorithms() {
    printTestHeader("TEST 7: ALGORITHM COMPARISON");
    
    std::cout << "\n  Comparison of algorithm complexities:" << std::endl;
    std::cout << "    Naive matrix multiplication: O(n^3)" << std::endl;
    std::cout << "    Strassen algorithm: O(n^2.807)" << std::endl;
    std::cout << "    Parallel Strassen: O(n^2.807 / p) where p = number of threads" << std::endl;
    std::cout << "    Merge sort: O(n log n)" << std::endl;
    std::cout << "    Parallel merge sort: O(n log n / p)" << std::endl;
    
    // Практическое сравнение для матриц 256x256
    std::cout << "\n  Practical comparison for 256x256 matrices:" << std::endl;
    
    Matrix A(256, 256);
    Matrix B(256, 256);
    A.fillRandom();
    B.fillRandom();
    
    // Наивное умножение
    std::cout << "    Naive multiplication:" << std::endl;
    TestTimer naiveTimer("Naive");
    StrassenMultiplier::naiveMultiply(A, B);
    
    // Штрассен с 1 потоком
    std::cout << "    Strassen (1 thread):" << std::endl;
    TestTimer strassen1Timer("Strassen 1 thread");
    StrassenMultiplier strassen1(1);
    strassen1.multiply(A, B);
    
    // Штрассен с 4 потоками
    std::cout << "    Strassen (4 threads):" << std::endl;
    TestTimer strassen4Timer("Strassen 4 threads");
    StrassenMultiplier strassen4(4);
    strassen4.multiply(A, B);
    
    // Штрассен с 8 потоками
    std::cout << "    Strassen (8 threads):" << std::endl;
    TestTimer strassen8Timer("Strassen 8 threads");
    StrassenMultiplier strassen8(8);
    strassen8.multiply(A, B);
    
    std::cout << "\n  Practical comparison for sorting 1,000,000 elements:" << std::endl;
    
    std::vector<int> array(1000000);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10000000);
    
    for (auto& val : array) {
        val = dis(gen);
    }
    
    // std::sort
    std::vector<int> array1 = array;
    std::cout << "    std::sort:" << std::endl;
    TestTimer stdSortTimer("std::sort");
    std::sort(array1.begin(), array1.end());
    
    // Parallel sort с 1 потоком
    std::vector<int> array2 = array;
    std::cout << "    Parallel sort (1 thread):" << std::endl;
    TestTimer parallel1Timer("Parallel 1 thread");
    ParallelSort sort1(1);
    sort1.sort(array2);
    
    // Parallel sort с 4 потоками
    std::vector<int> array3 = array;
    std::cout << "    Parallel sort (4 threads):" << std::endl;
    TestTimer parallel4Timer("Parallel 4 threads");
    ParallelSort sort4(4);
    sort4.sort(array3);
    
    // Parallel sort с 8 потоками
    std::vector<int> array4 = array;
    std::cout << "    Parallel sort (8 threads):" << std::endl;
    TestTimer parallel8Timer("Parallel 8 threads");
    ParallelSort sort8(8);
    sort8.sort(array4);
}

// ============================================================================
// ЭКСПОРТ РЕЗУЛЬТАТОВ
// ============================================================================

void exportResults() {
    printTestHeader("EXPORTING RESULTS TO FILE");
    
    std::ofstream outFile("benchmark_results.txt");
    if (!outFile.is_open()) {
        std::cout << "\n  ERROR: Could not open file for writing results!" << std::endl;
        return;
    }
    
    outFile << "PARALLEL ALGORITHMS BENCHMARK RESULTS\n";
    outFile << "=======================================\n\n";
    outFile << "System information:\n";
    outFile << "  Hardware concurrency: " << std::thread::hardware_concurrency() << " threads\n\n";
    
    // Тестируем умножение матриц
    outFile << "MATRIX MULTIPLICATION TESTS\n";
    outFile << "---------------------------\n";
    
    std::vector<size_t> matrixSizes = {128, 256, 512};
    std::vector<size_t> threadCounts = {1, 2, 4, 8};
    
    for (size_t size : matrixSizes) {
        outFile << "\n  Matrix size: " << size << "x" << size << "\n";
        
        Matrix A(size, size);
        Matrix B(size, size);
        A.fillRandom();
        B.fillRandom();
        
        for (size_t threads : threadCounts) {
            auto start = std::chrono::high_resolution_clock::now();
            StrassenMultiplier multiplier(threads);
            multiplier.multiply(A, B);
            auto end = std::chrono::high_resolution_clock::now();
            double time = std::chrono::duration<double>(end - start).count();
            
            outFile << "    " << threads << " thread(s): " 
                    << std::fixed << std::setprecision(4) << time << " seconds\n";
        }
    }
    
    // Тестируем сортировку
    outFile << "\n\nSORTING TESTS\n";
    outFile << "-------------\n";
    
    std::vector<size_t> arraySizes = {100000, 1000000, 5000000};
    
    for (size_t size : arraySizes) {
        outFile << "\n  Array size: " << size << " elements\n";
        
        std::vector<int> array(size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, 10000000);
        
        for (auto& val : array) {
            val = dis(gen);
        }
        
        for (size_t threads : threadCounts) {
            std::vector<int> testArray = array;
            auto start = std::chrono::high_resolution_clock::now();
            ParallelSort sorter(threads);
            sorter.sort(testArray);
            auto end = std::chrono::high_resolution_clock::now();
            double time = std::chrono::duration<double>(end - start).count();
            
            outFile << "    " << threads << " thread(s): " 
                    << std::fixed << std::setprecision(4) << time << " seconds\n";
        }
    }
    
    outFile.close();
    std::cout << "\n  Results exported to: benchmark_results.txt" << std::endl;
}

// ============================================================================
// ТЕСТИРОВАНИЕ НА БОЛЬШИХ РАЗМЕРАХ
// ============================================================================

void testLargeScale() {
    printTestHeader("TEST 8: LARGE SCALE PERFORMANCE TESTING");
    
    std::cout << "\n  Testing with large matrix size: 1024x1024" << std::endl;
    
    Matrix A(1024, 1024);
    Matrix B(1024, 1024);
    A.fillRandom();
    B.fillRandom();
    
    std::vector<size_t> threadCounts = {1, 2, 4, 8};
    
    for (size_t threads : threadCounts) {
        std::cout << "    Strassen with " << threads << " thread(s):" << std::endl;
        TestTimer timer("Strassen " + std::to_string(threads) + " threads");
        
        StrassenMultiplier multiplier(threads);
        multiplier.multiply(A, B);
    }
    
    std::cout << "\n  Testing with large array size: 10,000,000 elements" << std::endl;
    
    std::vector<int> largeArray(10000000);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10000000);
    
    for (auto& val : largeArray) {
        val = dis(gen);
    }
    
    for (size_t threads : threadCounts) {
        std::vector<int> testArray = largeArray;
        std::cout << "    Parallel sort with " << threads << " thread(s):" << std::endl;
        TestTimer timer("Sort " + std::to_string(threads) + " threads");
        
        ParallelSort sorter(threads);
        sorter.sort(testArray);
        
        if (!ParallelSort::isSorted(testArray)) {
            std::cout << "    WARNING: Sorting may be incorrect!" << std::endl;
        }
    }
}

// ============================================================================
// ОСНОВНАЯ ФУНКЦИЯ
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "================================================================" << std::endl;
    std::cout << "  PARALLEL ALGORITHMS TEST SUITE" << std::endl;
    std::cout << "  Testing with 1, 2, 4, 8 threads" << std::endl;
    std::cout << "================================================================" << std::endl;
    
    std::cout << "\nStarting comprehensive testing..." << std::endl;
    std::cout << "This may take several minutes depending on your system." << std::endl;
    
    try {
        // Тестирование умножения матриц
        testMatrixMultiplicationCorrectness();
        benchmarkMatrixMultiplication();
        testStrassenScalability();
        
        // Тестирование сортировки
        testSortingCorrectness();
        benchmarkSortingPerformance();
        testSortingScalability();
        
        // Тестирование на больших размерах
        testLargeScale();
        
        // Сравнительные тесты
        compareAlgorithms();
        
        // Экспорт результатов
        exportResults();
        
        std::cout << "\n";
        std::cout << "================================================================" << std::endl;
        std::cout << "  ALL TESTS COMPLETED SUCCESSFULLY" << std::endl;
        std::cout << "================================================================" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\nERROR during testing: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\nUNKNOWN ERROR during testing" << std::endl;
        return 1;
    }
    
    std::cout << "\nTest summary:" << std::endl;
    std::cout << "  - 8 test suites executed" << std::endl;
    std::cout << "  - Matrix multiplication tested with 1, 2, 4, 8 threads" << std::endl;
    std::cout << "  - Sorting tested with 1, 2, 4, 8 threads" << std::endl;
    std::cout << "  - Matrix sizes from 32x32 to 1024x1024" << std::endl;
    std::cout << "  - Array sizes from 1,000 to 10,000,000 elements" << std::endl;
    std::cout << "  - Results exported to benchmark_results.txt" << std::endl;
    std::cout << "\n";
    
    return 0;
}
