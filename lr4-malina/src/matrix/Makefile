CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall -Iinclude
OPENCL_FLAGS = -lOpenCL

SRCDIR = src
BUILDDIR = build

SOURCES = $(wildcard $(SRCDIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRCDIR)/%.cpp=$(BUILDDIR)/%.o)

TARGET_TESTS = $(BUILDDIR)/tests
TARGET_BENCHMARKS = $(BUILDDIR)/benchmarks
TARGET_MAIN = $(BUILDDIR)/main

.PHONY: all clean tests benchmarks

all: $(TARGET_TESTS) $(TARGET_BENCHMARKS) $(TARGET_MAIN)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)
	mkdir -p $(BUILDDIR)/kernels

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Copy kernel file to build directory
$(BUILDDIR)/kernels/matrix_multiply.cl: kernels/matrix_multiply.cl | $(BUILDDIR)
	cp $< $@

# Ensure kernel is copied before building executables
$(TARGET_TESTS): $(BUILDDIR)/kernels/matrix_multiply.cl $(BUILDDIR)/tests.o $(BUILDDIR)/cpu_matrix_operations.o $(BUILDDIR)/gpu_matrix_operations.o
	$(CXX) $(filter %.o,$^) -o $@ $(OPENCL_FLAGS)

$(TARGET_BENCHMARKS): $(BUILDDIR)/kernels/matrix_multiply.cl $(BUILDDIR)/benchmarks.o $(BUILDDIR)/cpu_matrix_operations.o $(BUILDDIR)/gpu_matrix_operations.o
	$(CXX) $(filter %.o,$^) -o $@ $(OPENCL_FLAGS)

$(TARGET_MAIN): $(BUILDDIR)/kernels/matrix_multiply.cl $(BUILDDIR)/main.o $(BUILDDIR)/cpu_matrix_operations.o $(BUILDDIR)/gpu_matrix_operations.o
	$(CXX) $(filter %.o,$^) -o $@ $(OPENCL_FLAGS)

tests: $(TARGET_TESTS)
	./$(TARGET_TESTS)

benchmarks: $(TARGET_BENCHMARKS)
	./$(TARGET_BENCHMARKS)

clean:
	rm -rf $(BUILDDIR)
	rm -f *.csv *.png

plot:
	python3 scripts/plot_results.py