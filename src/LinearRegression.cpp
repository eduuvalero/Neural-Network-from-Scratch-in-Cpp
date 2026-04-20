#include "LinearRegression.h"
#include "Metrics.h"
#include "Random.h"
#include "TrainingUtils.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

LinearRegression::LinearRegression(int inputs):
    weights(inputs, 1),
    bias(1, 1),
    hasCompileConfig(false),
    compiledLr(0.01),
    compiledBatchSize(0),
    compiledShuffleSeed(-1),
    compiledLogMetrics(false),
    compiledMetricsEvery(1) {
    if (inputs <= 0) {
        throw std::invalid_argument("LinearRegression::LinearRegression invalid input dimension: " + std::to_string(inputs));
    }

    Random rng;
    for (int i = 0; i < inputs; i++) {
        weights(i, 0) = rng.uniform(-0.01, 0.01);
    }
    bias(0, 0) = 0.0;
}

void LinearRegression::compile(double lr, int batchSize, int shuffleSeed, bool logMetrics, int metricsEvery) {
    if (lr <= 0.0) {
        throw std::invalid_argument("LinearRegression::compile learning rate must be > 0");
    }
    if (logMetrics && metricsEvery <= 0) {
        throw std::invalid_argument("LinearRegression::compile metricsEvery must be > 0 when logMetrics is enabled");
    }

    compiledLr = lr;
    compiledBatchSize = batchSize;
    compiledShuffleSeed = shuffleSeed;
    compiledLogMetrics = logMetrics;
    compiledMetricsEvery = metricsEvery;
    hasCompileConfig = true;
}

void LinearRegression::fit(const Matrix& X, const Matrix& Y, int epochs) {
    if (!hasCompileConfig) {
        throw std::logic_error("LinearRegression::fit compile() must be called before fit()");
    }

    train(X, Y, epochs, compiledLr, compiledBatchSize, compiledShuffleSeed, compiledLogMetrics, compiledMetricsEvery);
}

Matrix LinearRegression::forwardLinear(const Matrix& X) const {
    Matrix pred = X.dot(weights);
    double b = bias(0, 0);
    for (int i = 0; i < pred.getRows(); i++) {
        pred(i, 0) += b;
    }
    return pred;
}

double LinearRegression::sumColumnVector(const Matrix& vec) const {
    if (vec.getCols() != 1) {
        throw std::invalid_argument("LinearRegression::sumColumnVector expected a column vector");
    }

    double sum = 0.0;
    for (int i = 0; i < vec.getRows(); i++) {
        sum += vec(i, 0);
    }
    return sum;
}

void LinearRegression::train(const Matrix& X, const Matrix& Y, int epochs, double lr, int batchSize, int shuffleSeed, bool logMetrics, int metricsEvery){
    int expectedInputs = weights.getRows();

    if (epochs <= 0) {
        throw std::invalid_argument("LinearRegression::train epochs must be > 0");
    }
    if (lr <= 0.0) {
        throw std::invalid_argument("LinearRegression::train learning rate must be > 0");
    }
    if (logMetrics && metricsEvery <= 0) {
        throw std::invalid_argument("LinearRegression::train metricsEvery must be > 0 when logMetrics is enabled");
    }

    if (X.getCols() != expectedInputs) {
        throw std::invalid_argument("LinearRegression::train input dimension mismatch: " + std::to_string(X.getRows()) + "x" + std::to_string(X.getCols()) + " | expected Nx" + std::to_string(expectedInputs));
    }

    int n = X.getRows();

    if (Y.getRows() != n || Y.getCols() != 1) {
        throw std::invalid_argument("LinearRegression::train label dimension mismatch: " + std::to_string(Y.getRows()) + "x" + std::to_string(Y.getCols()) + " | expected " + std::to_string(n) + "x1");
    }

    if (n <= 0) {
        throw std::invalid_argument("LinearRegression::train empty dataset");
    }

    int effectiveBatchSize = TrainingUtils::resolveEffectiveBatchSize(batchSize, n, "LinearRegression::train");
    std::mt19937 gen = TrainingUtils::createShuffleGenerator(shuffleSeed, "LinearRegression::train");
    std::vector<int> rowOrder = TrainingUtils::makeSequentialRowOrder(n);

    for (int e = 0; e < epochs; e++) {
        TrainingUtils::shuffleRowOrder(rowOrder, gen);

        Matrix xShuffled = X.shuffleRows(rowOrder);
        Matrix yShuffled = Y.shuffleRows(rowOrder);

        for (int start = 0; start < n; start += effectiveBatchSize) {
            int end = start + effectiveBatchSize;
            if (end > n) {
                end = n;
            }

            Matrix xBatch = xShuffled.vslice(start, end);
            Matrix yBatch = yShuffled.vslice(start, end);
            int batchN = xBatch.getRows();

            Matrix pred = forwardLinear(xBatch);
            Matrix grad = (pred - yBatch) * (2.0 / batchN);

            Matrix gradW = xBatch.transpose().dot(grad);
            double gradB = sumColumnVector(grad);

            weights = weights - (gradW * lr);
            bias(0, 0) -= lr * gradB;
        }

        bool shouldLog = logMetrics && (((e + 1) % metricsEvery == 0) || (e == epochs - 1));
        if (shouldLog) {
            Matrix epochPred = forwardLinear(X);
            double mse = Metrics::mse(Y, epochPred);
            double mae = Metrics::mae(Y, epochPred);
            double rmse = Metrics::rmse(Y, epochPred);
            double r2 = Metrics::r2Score(Y, epochPred);

            std::cout << "[LinearRegression] epoch " << (e + 1) << "/" << epochs
                        << " mse=" << mse << " mae=" << mae
                        << " rmse=" << rmse << " r2=" << r2 << "\n";
        }
    }

}

Matrix LinearRegression::predict(const Matrix& x){
    int expectedInputs = weights.getRows();

    if (x.getCols() != expectedInputs) {
        throw std::invalid_argument("LinearRegression::predict input dimension mismatch: " + std::to_string(x.getRows()) + "x" + std::to_string(x.getCols()) + " | expected Nx" + std::to_string(expectedInputs));
    }

    return forwardLinear(x);
}

void LinearRegression::save(const std::string& path){
    std::ofstream file(path, std::ios::binary);

    int inputs = weights.getRows();
    file.write((const char*)&inputs, sizeof(int));

    const std::vector<double>& wData = weights.getData();
    file.write((const char*)wData.data(), wData.size() * sizeof(double));

    double bData = bias(0, 0);
    file.write((const char*)&bData, sizeof(double));
}

void LinearRegression::load(const std::string& path){
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("LinearRegression::load failed to open file: " + path);
    }
    
    int inputs;
    file.read((char*)&inputs, sizeof(int));

    std::vector<double> wData(inputs);
    file.read((char*)wData.data(), wData.size() * sizeof(double));
    weights.setData(wData);

    double bData;
    file.read((char*)&bData, sizeof(double));
    bias(0, 0) = bData;
}