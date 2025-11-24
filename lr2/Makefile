CXX := g++
CXXFLAGS := -std=c++17 -O2 -pthread -Wall -Wextra
INCLUDES := -Iinclude
SRCDIR := src
OBJDIR := obj
BINDIR := bin

# Sources for benchmark
BENCH_SOURCES := main.cpp $(SRCDIR)/list_fine.cpp $(SRCDIR)/list_coarse.cpp $(SRCDIR)/utils.cpp
BENCH_OBJECTS := $(patsubst %.cpp,$(OBJDIR)/%.o,$(notdir $(BENCH_SOURCES)))

# Sources for tests
TEST_SOURCES := test_main.cpp $(SRCDIR)/list_fine.cpp $(SRCDIR)/list_coarse.cpp $(SRCDIR)/utils.cpp
TEST_OBJECTS := $(patsubst %.cpp,$(OBJDIR)/%.o,$(notdir $(TEST_SOURCES)))

TARGET := $(BINDIR)/list_bench
TEST_TARGET := $(BINDIR)/test_list

.PHONY: all clean run test bench

all: bench

bench: $(TARGET)

test: $(TEST_TARGET)

$(OBJDIR)/%.o: %.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(TARGET): $(BENCH_OBJECTS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(BENCH_OBJECTS) -o $@

$(TEST_TARGET): $(TEST_OBJECTS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(TEST_OBJECTS) -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(BINDIR):
	mkdir -p $(BINDIR)

clean:
	rm -rf $(OBJDIR) $(BINDIR) *.csv plots

run: $(TARGET)
	./$(TARGET)

# Run tests and then benchmark
full-test: test
	./$(TEST_TARGET)
