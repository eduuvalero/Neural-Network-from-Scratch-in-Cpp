CXX ?= g++
STD ?= c++17
WARN ?= -Wall -Wextra -Wpedantic
OPT ?= -O2
CPPFLAGS ?= -Iinclude
CXXFLAGS ?= -std=$(STD) $(WARN) $(OPT)

MAIN ?= main.cc
SRC_DIR := src
BUILD_DIR := build
TARGET ?= $(BUILD_DIR)/app

LIB_SRCS := \
	$(SRC_DIR)/DataLoader.cc \
	$(SRC_DIR)/Layer.cc \
	$(SRC_DIR)/LinearRegression.cc \
	$(SRC_DIR)/Metrics.cc \
	$(SRC_DIR)/Matrix.cc \
	$(SRC_DIR)/NeuralNetwork.cc \
	$(SRC_DIR)/StandardScaler.cc \
	$(SRC_DIR)/TrainingUtils.cc

SRCS := $(LIB_SRCS) $(MAIN)
OBJS := $(patsubst %.cc,$(BUILD_DIR)/%.o,$(SRCS))

.PHONY: all run clean debug release

all: $(TARGET)

$(TARGET): $(OBJS) | $(BUILD_DIR)
	$(CXX) $(OBJS) -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.cc | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

debug: OPT := -O0 -g
debug: clean all

release: OPT := -O3
release: clean all

clean:
	rm -rf $(BUILD_DIR)
