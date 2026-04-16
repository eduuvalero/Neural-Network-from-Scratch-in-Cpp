#ifndef _TRAINING_UTILS_H_
#define _TRAINING_UTILS_H_

#include <algorithm>
#include <random>
#include <string>
#include <vector>

class TrainingUtils {
	public:
		static int resolveEffectiveBatchSize(int batchSize, int numSamples, const std::string& context);
		static std::mt19937 createShuffleGenerator(int shuffleSeed, const std::string& context);
		static std::vector<int> makeSequentialRowOrder(int numSamples);
		static void shuffleRowOrder(std::vector<int>& rowOrder, std::mt19937& generator) { std::shuffle(rowOrder.begin(), rowOrder.end(), generator); }
};

#endif
