#ifndef _LINEAR_REGRESSION_H_
#define _LINEAR_REGRESSION_H_

#include <string>

#include "Matrix.h"
#include "Layer.h"

class LinearRegression{
    private:
        Layer neuron;
    public:
        LinearRegression(int inputs, int outputs);
        void train(const Matrix& X, const Matrix& Y, int epochs, double lr, int batchSize = 0, int shuffleSeed = -1, bool logMetrics = false, int metricsEvery = 1);
        Matrix predict(const Matrix& x);
        void save(const std::string& path);
        void load(const std::string& path);
};

#endif