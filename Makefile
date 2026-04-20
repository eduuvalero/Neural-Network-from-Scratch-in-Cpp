CC = g++
OPTS ?= -O2 -Wall -Wextra -Wpedantic -std=c++17
INCLUDES ?= -Iinclude

MAIN ?= main.cpp
BUILD_DIR ?= build
TARGET ?= $(BUILD_DIR)/app

OBJS = \
	$(BUILD_DIR)/src/DataLoader.o \
	$(BUILD_DIR)/src/utils/Random.o \
	$(BUILD_DIR)/src/Layer.o \
	$(BUILD_DIR)/src/LinearRegression.o \
	$(BUILD_DIR)/src/Metrics.o \
	$(BUILD_DIR)/src/Matrix.o \
	$(BUILD_DIR)/src/NeuralNetwork.o \
	$(BUILD_DIR)/src/StandardScaler.o \
	$(BUILD_DIR)/src/utils/TrainingUtils.o

MAIN_OBJ = $(BUILD_DIR)/$(basename $(MAIN)).o

.PHONY: all run clean debug release

all: $(TARGET)

$(TARGET): $(OBJS) $(MAIN_OBJ)
	@mkdir -p $(dir $@)
	$(CC) $(OPTS) $(OBJS) $(MAIN_OBJ) -o $@

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CC) $(INCLUDES) $(OPTS) -c $< -o $@

$(BUILD_DIR)/%.o: %.cc
	@mkdir -p $(dir $@)
	$(CC) $(INCLUDES) $(OPTS) -c $< -o $@

run: all
	./$(TARGET)

debug: OPTS := -O0 -g -Wall -Wextra -Wpedantic -std=c++17
debug: clean all

release: OPTS := -O3 -Wall -Wextra -Wpedantic -std=c++17
release: clean all

clean:
	rm -rf $(BUILD_DIR)
