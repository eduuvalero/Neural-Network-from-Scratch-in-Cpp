#include "LinearRegression.h"
#include "Metrics.h"
#include "TrainingUtils.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

LinearRegression::LinearRegression(int inputs, int outputs){
    if (inputs <= 0 || outputs <= 0) {
        throw std::invalid_argument("LinearRegression::LinearRegression invalid dimensions: " + std::to_string(inputs) + "x" + std::to_string(outputs));
    }

    neuron = Layer(inputs, outputs, NONE);
}

void LinearRegression::train(const Matrix& X, const Matrix& Y, int epochs, double lr, int batchSize, int shuffleSeed, bool logMetrics, int metricsEvery){
    int expectedInputs = neuron.getWeights().getRows();
    int expectedOutputs = neuron.getOutputSize();

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

    if (Y.getRows() != n || Y.getCols() != expectedOutputs) {
        throw std::invalid_argument("LinearRegression::train label dimension mismatch: " + std::to_string(Y.getRows()) + "x" + std::to_string(Y.getCols()) + " | expected " + std::to_string(n) + "x" + std::to_string(expectedOutputs));
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

            Matrix pred = neuron.forward(xBatch);
            Matrix grad = (pred - yBatch) * (2.0 / batchN);
            neuron.backward(grad, lr);
        }

        bool shouldLog = logMetrics && (((e + 1) % metricsEvery == 0) || (e == epochs - 1));
        if (shouldLog) {
            Matrix epochPred = neuron.forward(X);
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
    int expectedInputs = neuron.getWeights().getRows();

    if (x.getCols() != expectedInputs) {
        throw std::invalid_argument("LinearRegression::predict input dimension mismatch: " + std::to_string(x.getRows()) + "x" + std::to_string(x.getCols()) + " | expected Nx" + std::to_string(expectedInputs));
    }

    return neuron.forward(x);
}

void LinearRegression::save(const std::string& path){
    std::ofstream file(path, std::ios::binary);
    if(!file.is_open()){
        throw std::invalid_argument("LinearRegression::save open file error: " + path);
    }

    int inputs = neuron.getWeights().getRows();
    file.write((const char*)&inputs, sizeof(int));
    if(!file){
        throw std::runtime_error("LinearRegression::save write error: rows");
    }

    int outputs = neuron.getWeights().getCols();
    file.write((const char*)&outputs, sizeof(int));
    if(!file){
        throw std::runtime_error("LinearRegression::save write error: cols");
    }

    Matrix weights = neuron.getWeights();
    const std::vector<double>& wData = weights.getData();
    file.write((const char*)wData.data(), wData.size() * sizeof(double));
    if(!file){
        throw std::runtime_error("LinearRegression::save write error: weights");
    }

    Matrix bias = neuron.getBias();
    const std::vector<double>& bData = bias.getData();
    file.write((const char*)bData.data(), bData.size() * sizeof(double));
    if(!file){
        throw std::runtime_error("LinearRegression::save write error: bias");
    }
}

void LinearRegression::load(const std::string& path){
    std::ifstream file(path, std::ios::binary);
    if(!file.is_open()){
        throw std::invalid_argument("LinearRegression::load open file error: " + path);
    }

    
    int inputs, outputs;
    file.read((char*)&inputs, sizeof(int));
    if(!file){
        throw std::runtime_error("LinearRegression::load invalid model file: rows");
    }
    file.read((char*)&outputs, sizeof(int));
    if(!file){
        throw std::runtime_error("LinearRegression::load invalid model file: cols");
    }
    if(inputs <= 0 || outputs <= 0){
        throw std::runtime_error("LinearRegression::load dimension mismatch: " + std::to_string(inputs) + "x" + std::to_string(outputs) + " | expected rows>0 and cols>0");
    }

    neuron = Layer(inputs, outputs, NONE);

    Matrix w(inputs, outputs);
    std::vector<double> wData(inputs * outputs);
    file.read((char*)wData.data(), wData.size() * sizeof(double));
    if(!file){
        throw std::runtime_error("LinearRegression::load invalid model file: weights");
    }
    w.setData(wData);

    Matrix b(1, outputs);
    std::vector<double> bData(outputs);
    file.read((char*)bData.data(), bData.size() * sizeof(double));
    if(!file){
        throw std::runtime_error("LinearRegression::load invalid model file: bias");
    }
    b.setData(bData);

    neuron.setWeights(w);
    neuron.setBias(b);
}