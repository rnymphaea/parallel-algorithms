TARGET = lab3

CXX = g++
CXXFLAGS = -std=c++17 -Iinclude -O3 -Wall -Wextra -pthread
LDFLAGS = -pthread

INCLUDE_DIR = include
SRC_DIR = src
OBJ_DIR = obj

SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SOURCES))

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(TARGET) *.png *.pdf *.csv *.txt

test: $(TARGET)
	./$(TARGET) --test

benchmark: $(TARGET)
	./$(TARGET) --benchmark

.PHONY: all clean test benchmark
