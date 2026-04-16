#ifndef _RANDOM_H_
#define _RANDOM_H_

#include <random>

class Random {
    private:
        std::mt19937 gen;;

    public:
        Random();
        double uniform(double a, double b);
        double normal(double mean, double stddev);
};

#endif