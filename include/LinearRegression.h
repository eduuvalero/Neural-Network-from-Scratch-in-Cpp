#ifndef _LINEAR_REGRESSION_H_
#define _LINEAR_REGRESSION_H_

#include <string>

#include "Matrix.h"

class LinearRegression{
    private:
        Matrix weights;
        Matrix bias;
        bool hasCompileConfig;
        double compiledLr;
        int compiledBatchSize;
        int compiledShuffleSeed;
        bool compiledLogMetrics;
        int compiledMetricsEvery;
        Matrix forwardLinear(const Matrix& X) const;
        double sumColumnVector(const Matrix& vec) const;
        void train(const Matrix& X, const Matrix& Y, int epochs, double lr, int batchSize, int shuffleSeed, bool logMetrics, int metricsEvery);

    public:
        LinearRegression(int inputs);
        void compile(double lr = 0.01, int batchSize = 0, int shuffleSeed = -1, bool logMetrics = false, int metricsEvery = 1);
        void fit(const Matrix& X, const Matrix& Y, int epochs);
        Matrix predict(const Matrix& x);
        void save(const std::string& path);
        void load(const std::string& path);
};

#endif