#include "TrainingUtils.h"

#include <numeric>
#include <random>
#include <stdexcept>

int TrainingUtils::resolveEffectiveBatchSize(int batchSize, int numSamples, const std::string& context) {
    if (numSamples <= 0) {
        throw std::invalid_argument(context + " dataset must contain at least one sample");
    }
    if (batchSize < 0) {
        throw std::invalid_argument(context + " batch size must be >= 0");
    }
    if (batchSize > numSamples) {
        throw std::invalid_argument(
            context + " batch size must be <= number of samples: " +
            std::to_string(batchSize) + " | " + std::to_string(numSamples)
        );
    }

    return (batchSize == 0) ? numSamples : batchSize;
}

std::mt19937 TrainingUtils::createShuffleGenerator(int shuffleSeed, const std::string& context) {
    if (shuffleSeed < -1) {
        throw std::invalid_argument(context + " shuffle seed must be >= -1");
    }

    std::mt19937 generator;
    if (shuffleSeed >= 0) {
        generator.seed((unsigned int)shuffleSeed);
    }
    else {
        std::random_device rd;
        generator.seed(rd());
    }

    return generator;
}

std::vector<int> TrainingUtils::makeSequentialRowOrder(int numSamples) {
    if (numSamples <= 0) {
        return {};
    }

    std::vector<int> rowOrder(numSamples);
    std::iota(rowOrder.begin(), rowOrder.end(), 0);
    return rowOrder;
}

