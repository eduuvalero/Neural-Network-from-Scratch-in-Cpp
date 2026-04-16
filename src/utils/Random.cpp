#include "Random.h"

Random::Random() : gen(std::random_device{}()) {}

double Random::uniform(double a, double b) {
    std::uniform_real_distribution<double> dist(a, b);
    return dist(gen);
}

double Random::normal(double mean, double stddev) {
    std::normal_distribution<double> dist(mean, stddev);
    return dist(gen);
}