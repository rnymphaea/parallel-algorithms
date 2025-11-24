CXX := g++
CXXFLAGS := -std=c++17 -O2 -pthread -Wall -Wextra
INCLUDES := -Iinclude
SRCDIR := src
OBJDIR := obj
BINDIR := bin
TEST_TARGET := $(BINDIR)/test_list

SOURCES := main.cpp $(SRCDIR)/list_fine.cpp $(SRCDIR)/list_coarse.cpp $(SRCDIR)/utils.cpp
OBJECTS := $(patsubst %.cpp,$(OBJDIR)/%.o,$(notdir $(SOURCES)))

TARGET := $(BINDIR)/list_bench

.PHONY: all clean run test

all: $(TARGET)

$(OBJDIR)/%.o: %.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(TARGET): $(OBJECTS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJECTS) -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(BINDIR):
	mkdir -p $(BINDIR)

test: $(TEST_TARGET)
	./$(TEST_TARGET)

$(TEST_TARGET): $(OBJECTS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJECTS) -o $@

clean:
	rm -rf $(OBJDIR) $(BINDIR) *.csv plots

run: $(TARGET)
	./$(TARGET)
