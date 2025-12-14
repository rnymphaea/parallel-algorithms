TARGET = gpu_lab
CXX = g++
CXXFLAGS = -std=c++11 -Iinclude -O2 -Wall -Wextra
LDFLAGS = -lOpenCL

SRC_DIR = src
OBJ_DIR = obj

SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SOURCES))

.PHONY: all clean test bench

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

test: $(TARGET)
	@echo "Running basic test..."
	@./$(TARGET) --task all --size 10000 --rows 256 --cols 256 --k 256 --verify

bench: $(TARGET)
	@echo "Running benchmark..."
	@./$(TARGET) --task all --size 100000 --rows 512 --cols 512 --k 512 --threads 1,2,4,8

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

help:
	@echo "Available targets:"
	@echo "  all    - Build the program (default)"
	@echo "  test   - Run basic test with verification"
	@echo "  bench  - Run benchmark with different thread counts"
	@echo "  clean  - Remove build artifacts"
