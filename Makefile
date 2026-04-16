CXX ?= g++
STD ?= c++17
WARN ?= -Wall -Wextra -Wpedantic
OPT ?= -O2
CPPFLAGS ?= -Iinclude
CXXFLAGS ?= -std=$(STD) $(WARN) $(OPT)

MAIN ?= main.cpp
SRC_DIR := src
BUILD_DIR := build
TARGET ?= $(BUILD_DIR)/app

LIB_SRCS := \
	$(SRC_DIR)/DataLoader.cpp \
	$(SRC_DIR)/utils/Random.cpp \
	$(SRC_DIR)/Layer.cpp \
	$(SRC_DIR)/LinearRegression.cpp \
	$(SRC_DIR)/Metrics.cpp \
	$(SRC_DIR)/Matrix.cpp \
	$(SRC_DIR)/NeuralNetwork.cpp \
	$(SRC_DIR)/StandardScaler.cpp \
	$(SRC_DIR)/utils/TrainingUtils.cpp

SRCS := $(LIB_SRCS) $(MAIN)
OBJS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(SRCS))

.PHONY: all run clean debug release

all: $(TARGET)

$(TARGET): $(OBJS) | $(BUILD_DIR)
	$(CXX) $(OBJS) -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
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
