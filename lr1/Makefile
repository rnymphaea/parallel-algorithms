TARGET = mm

CXX = g++
CXXFLAGS = -std=c++17 -Iinclude -g -Wall -Wextra -Werror
LDFLAGS  = -pthread
INCLUDE_DIR = include
SRC_DIR = src
OBJ_DIR = obj
PYTHON = python3

SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SOURCES))

THREADS ?= 1 2 4 8 16 32 64
ROWS ?= 1024
COLS ?= 1024
REPEATS ?= 5
CSV ?= time.csv
PLOT ?= performance.png

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(LDFLAGS)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

bench: $(TARGET)
	@rm -f $(CSV) $(PLOT)
	@for t in $(THREADS); do \
		echo ">>> Running with $$t thread(s)"; \
		./$(TARGET) --rows $(ROWS) --columns $(COLS) --time --repeats $(REPEATS) --threads $$t --export-csv $(CSV); \
		echo ""; \
	done
	@$(PYTHON) $(SRC_DIR)/plot.py $(CSV) $(PLOT);

clean:
	rm -rf $(OBJ_DIR) $(TARGET) $(CSV) $(PLOT)
