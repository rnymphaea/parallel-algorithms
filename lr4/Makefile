CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall -Iinclude -DCL_TARGET_OPENCL_VERSION=120
OPENCL_FLAGS = -lOpenCL -pthread
BINDIR = bin

SOURCES := $(wildcard src/*.cpp)
OBJECTS := $(SOURCES:src/%.cpp=$(BINDIR)/%.o)

MATRIX_BENCH_OBJS = $(BINDIR)/MatrixBenchmarks.o $(BINDIR)/MatrixCpu.o $(BINDIR)/MatrixGpu.o
MATRIX_TEST_OBJS = $(BINDIR)/MatrixTests.o $(BINDIR)/MatrixCpu.o $(BINDIR)/MatrixGpu.o
MATRIX_DEMO_OBJS = $(BINDIR)/MatrixDemo.o $(BINDIR)/MatrixCpu.o $(BINDIR)/MatrixGpu.o

SORT_BENCH_OBJS = $(BINDIR)/SortBenchmarks.o $(BINDIR)/SortCpu.o $(BINDIR)/SortGpu.o $(BINDIR)/SortUtils.o
SORT_TEST_OBJS = $(BINDIR)/SortCorrectness.o $(BINDIR)/SortCpu.o $(BINDIR)/SortGpu.o $(BINDIR)/SortUtils.o
SORT_DEMO_OBJS = $(BINDIR)/SortDemo.o $(BINDIR)/SortCpu.o $(BINDIR)/SortGpu.o $(BINDIR)/SortUtils.o

.PHONY: all clean plots

all: $(BINDIR)/MatrixBenchmarks $(BINDIR)/MatrixTests $(BINDIR)/MatrixDemo \
     $(BINDIR)/SortBenchmarks $(BINDIR)/SortCorrectness $(BINDIR)/SortDemo

$(BINDIR):
	@mkdir -p $(BINDIR)

$(BINDIR)/%.o: src/%.cpp | $(BINDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BINDIR)/MatrixBenchmarks: $(MATRIX_BENCH_OBJS)
	$(CXX) $^ -o $@ $(OPENCL_FLAGS)

$(BINDIR)/MatrixTests: $(MATRIX_TEST_OBJS)
	$(CXX) $^ -o $@ $(OPENCL_FLAGS)

$(BINDIR)/MatrixDemo: $(MATRIX_DEMO_OBJS)
	$(CXX) $^ -o $@ $(OPENCL_FLAGS)

$(BINDIR)/SortBenchmarks: $(SORT_BENCH_OBJS)
	$(CXX) $^ -o $@ $(OPENCL_FLAGS)

$(BINDIR)/SortCorrectness: $(SORT_TEST_OBJS)
	$(CXX) $^ -o $@ $(OPENCL_FLAGS)

$(BINDIR)/SortDemo: $(SORT_DEMO_OBJS)
	$(CXX) $^ -o $@ $(OPENCL_FLAGS)

run_matrix_tests: $(BINDIR)/MatrixTests
	@echo "Running Matrix Tests..."
	@./$(BINDIR)/MatrixTests

run_matrix_benchmarks: $(BINDIR)/MatrixBenchmarks
	@echo "Running Matrix Benchmarks..."
	@./$(BINDIR)/MatrixBenchmarks

run_sort_benchmarks: $(BINDIR)/SortBenchmarks
	@echo "Running Sort Benchmarks..."
	@./$(BINDIR)/SortBenchmarks

run_sort_tests: $(BINDIR)/SortCorrectness
	@echo "Running Sort Correctness Tests..."
	@./$(BINDIR)/SortCorrectness

plot_matrix: plot_matrix_results.py
	@echo "Generating matrix plots..."
	@python3 plot_matrix_results.py

plot_sort: plot_sort_results.py
	@echo "Generating sort plots..."
	@python3 plot_sort_results.py

plots: plot_matrix plot_sort

clean:
	rm -rf $(BINDIR)
	rm -f *.csv *.png
