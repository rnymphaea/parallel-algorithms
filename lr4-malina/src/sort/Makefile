.PHONY: all build benchmark test plot clean clean-all setup info help

# Основные цели
all: build benchmark plot

build:
	@echo "Building project..."
	@mkdir -p build
	@cd build && cmake .. && make

benchmark: build
	@echo "=========================================="
	@echo "Running Benchmarks..."
	@echo "=========================================="
	@./build/bin/benchmark_runner

test: build
	@echo "=========================================="
	@echo "Running Correctness Tests..."
	@echo "=========================================="
	@./build/bin/correctness_tester

plot:
	@echo "=========================================="
	@echo "Generating Plots..."
	@echo "=========================================="
	@python3 scripts/plot_results.py

run: benchmark plot

# Утилиты
clean:
	@echo "Cleaning build files..."
	@rm -rf build/bin/*

clean-all:
	@echo "Removing entire build directory..."
	@rm -rf build
	@echo "Cleaning results..."
	@rm -rf results/*.csv

setup:
	@mkdir -p include src kernels scripts results
	@echo "Project structure created"

info:
	@echo "Project structure:"
	@echo "  include/    - Header files"
	@echo "  src/        - Source files" 
	@echo "  kernels/    - OpenCL kernel files"
	@echo "  scripts/    - Python plotting scripts"
	@echo "  results/    - CSV results and plots"
	@echo "  build/      - Build directory"

help:
	@echo "Available targets:"
	@echo "  all        - Build, benchmark and plot"
	@echo "  build      - Build project only"
	@echo "  benchmark  - Run performance benchmarks"
	@echo "  test       - Run correctness tests"
	@echo "  plot       - Generate plots from CSV data"
	@echo "  run        - Run benchmarks and generate plots"
	@echo "  clean      - Clean binary files"
	@echo "  clean-all  - Remove build directory and results"
	@echo "  setup      - Create project structure"
	@echo "  info       - Show project info"