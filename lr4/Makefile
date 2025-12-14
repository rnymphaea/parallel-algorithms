CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall -Iinclude -DCL_TARGET_OPENCL_VERSION=120
OPENCL_FLAGS = -lOpenCL -lpthread
BUILDDIR = build

# Разделы проекта
MATRIX_SOURCES = $(wildcard src/matrix_ops/*.cpp)
SORTER_SOURCES = $(wildcard src/sorter/*.cpp)

MATRIX_OBJECTS = $(MATRIX_SOURCES:src/matrix_ops/%.cpp=$(BUILDDIR)/matrix_ops/%.o)
SORTER_OBJECTS = $(SORTER_SOURCES:src/sorter/%.cpp=$(BUILDDIR)/sorter/%.o)

# Исполняемые файлы
MATRIX_TARGETS = $(BUILDDIR)/matrix_tests $(BUILDDIR)/matrix_benchmarks $(BUILDDIR)/matrix_demo
SORTER_TARGETS = $(BUILDDIR)/sort_benchmarks $(BUILDDIR)/sort_tests $(BUILDDIR)/sort_demo

.PHONY: all matrix sorter clean plots

all: matrix sorter

# Сборка раздела матриц
matrix: $(MATRIX_TARGETS)

$(BUILDDIR)/matrix_ops/%.o: src/matrix_ops/%.cpp | $(BUILDDIR)
	@mkdir -p $(BUILDDIR)/matrix_ops
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Сборка раздела сортировки
sorter: $(SORTER_TARGETS)

$(BUILDDIR)/sorter/%.o: src/sorter/%.cpp | $(BUILDDIR)
	@mkdir -p $(BUILDDIR)/sorter
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Создание директории build
$(BUILDDIR):
	@mkdir -p $(BUILDDIR)
	@mkdir -p $(BUILDDIR)/kernels
	@cp kernels/* $(BUILDDIR)/kernels/

# Правила сборки для матриц
$(BUILDDIR)/matrix_tests: $(BUILDDIR)/matrix_ops/tests.o $(BUILDDIR)/matrix_ops/cpu.o $(BUILDDIR)/matrix_ops/gpu.o
	$(CXX) $^ -o $@ $(OPENCL_FLAGS)

$(BUILDDIR)/matrix_benchmarks: $(BUILDDIR)/matrix_ops/benchmarks.o $(BUILDDIR)/matrix_ops/cpu.o $(BUILDDIR)/matrix_ops/gpu.o
	$(CXX) $^ -o $@ $(OPENCL_FLAGS)

$(BUILDDIR)/matrix_demo: $(BUILDDIR)/matrix_ops/main.o $(BUILDDIR)/matrix_ops/cpu.o $(BUILDDIR)/matrix_ops/gpu.o
	$(CXX) $^ -o $@ $(OPENCL_FLAGS)

# Правила сборки для сортировки
$(BUILDDIR)/sort_benchmarks: $(BUILDDIR)/sorter/benchmark_runner.o $(BUILDDIR)/sorter/cpu.o $(BUILDDIR)/sorter/gpu.o $(BUILDDIR)/sorter/utils.o
	$(CXX) $^ -o $@ $(OPENCL_FLAGS)

$(BUILDDIR)/sort_tests: $(BUILDDIR)/sorter/correctness_tester.o $(BUILDDIR)/sorter/cpu.o $(BUILDDIR)/sorter/gpu.o $(BUILDDIR)/sorter/utils.o
	$(CXX) $^ -o $@ $(OPENCL_FLAGS)

$(BUILDDIR)/sort_demo: $(BUILDDIR)/sorter/main.o $(BUILDDIR)/sorter/cpu.o $(BUILDDIR)/sorter/gpu.o $(BUILDDIR)/sorter/utils.o
	$(CXX) $^ -o $@ $(OPENCL_FLAGS)

# Цели для запуска
run_matrix_tests: $(BUILDDIR)/matrix_tests
	./$(BUILDDIR)/matrix_tests

run_matrix_benchmarks: $(BUILDDIR)/matrix_benchmarks
	./$(BUILDDIR)/matrix_benchmarks

run_sort_benchmarks: $(BUILDDIR)/sort_benchmarks
	./$(BUILDDIR)/sort_benchmarks

run_sort_tests: $(BUILDDIR)/sort_tests
	./$(BUILDDIR)/sort_tests

plot_matrix: scripts/plot_matrix_results.py
	python3 scripts/plot_matrix_results.py

plot_sort: scripts/plot_sort_results.py
	python3 scripts/plot_sort_results.py

plots: plot_matrix plot_sort

clean:
	rm -rf $(BUILDDIR)
	rm -f *.csv *.png
	rm -rf results/*

info:
	@echo "OpenCL headers:"
	@find /usr -name "opencl.h" 2>/dev/null || echo "Not found"
	@echo ""
	@echo "OpenCL libraries:"
	@find /usr -name "libOpenCL*" 2>/dev/null || echo "Not found"
