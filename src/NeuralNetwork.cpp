#include "NeuralNetwork.h"
#include "Metrics.h"
#include "TrainingUtils.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

NeuralNetwork::NeuralNetwork():
    inputFeatures(-1),
    hasCompileConfig(false),
    compiledLoss(AUTO_LOSS),
    compiledLr(0.01),
    compiledBatchSize(0),
    compiledShuffleSeed(-1),
    compiledLogMetrics(false),
    compiledMetricsEvery(1) {}

NeuralNetwork& NeuralNetwork::input(int features) {
    if (features <= 0) {
        throw std::invalid_argument("NeuralNetwork::input features must be > 0");
    }
    if (!layers.empty()) {
        throw std::logic_error("NeuralNetwork::input must be called before adding layers");
    }

    inputFeatures = features;
    return *this;
}

NeuralNetwork& NeuralNetwork::add(const Dense& layer) {
    if (layer.outputs <= 0) {
        throw std::invalid_argument("NeuralNetwork::add outputs must be > 0");
    }

    if (layers.empty()) {
        if (inputFeatures <= 0) {
            throw std::logic_error("NeuralNetwork::add first layer requires input(features)");
        }
        addLayer(inputFeatures, layer.outputs, layer.act, layer.init);
    }
    else {
        int inferredInputs = layers.back().getOutputSize();
        addLayer(inferredInputs, layer.outputs, layer.act, layer.init);
    }

    return *this;
}

void NeuralNetwork::compile(Loss loss, double lr, int batchSize, int shuffleSeed, bool logMetrics, int metricsEvery) {
    if (lr <= 0.0) {
        throw std::invalid_argument("NeuralNetwork::compile learning rate must be > 0");
    }
    if (logMetrics && metricsEvery <= 0) {
        throw std::invalid_argument("NeuralNetwork::compile metricsEvery must be > 0 when logMetrics is enabled");
    }

    compiledLoss = loss;
    compiledLr = lr;
    compiledBatchSize = batchSize;
    compiledShuffleSeed = shuffleSeed;
    compiledLogMetrics = logMetrics;
    compiledMetricsEvery = metricsEvery;
    hasCompileConfig = true;
}

void NeuralNetwork::fit(const Matrix& X, const Matrix& Y, int epochs) {
    if (!hasCompileConfig) {
        throw std::logic_error("NeuralNetwork::fit compile() must be called before fit()");
    }

    train(X, Y, epochs, compiledLr, compiledBatchSize, compiledLoss, compiledShuffleSeed, compiledLogMetrics, compiledMetricsEvery);
}

void NeuralNetwork::addLayer(int inputs, int outputs, Activation act, Inicialization init){
    if (inputs <= 0 || outputs <= 0) {
        throw std::invalid_argument("NeuralNetwork::addLayer invalid dimensions: " + std::to_string(inputs) + "x" + std::to_string(outputs));
    }

    if (!layers.empty()) {
        int expectedInputs = layers.back().getOutputSize();
        if (inputs != expectedInputs) {
            throw std::invalid_argument("NeuralNetwork::addLayer incompatible dimensions: expected " + std::to_string(expectedInputs) + " inputs, got " + std::to_string(inputs));
        }
    }

    if (layers.empty()) {
        inputFeatures = inputs;
    }

    layers.push_back(Layer(inputs, outputs, act, init));
}

Matrix NeuralNetwork::forward(const Matrix& X){
    Matrix out = X;
    for(Layer& l : layers){
        out = l.forward(out);
    }
    return out;
}

void NeuralNetwork::backward(const Matrix& grad, double lr, bool canonicalOutputGrad){
    if (layers.empty()) {
        return;
    }

    Matrix currentGrad = grad;
    for (int i = (int)layers.size() - 1; i >= 0; i--){
        bool applyActDerivative = true;
        if (canonicalOutputGrad && i == (int)layers.size() - 1) {
            Activation outAct = layers[i].getAct();
            if (supportsCanonicalCrossEntropyGradient(outAct)) {
                applyActDerivative = false;
            }
        }
        currentGrad = layers[i].backward(currentGrad, lr, applyActDerivative);
    }
}

void NeuralNetwork::train(const Matrix& X, const Matrix& Y, int epochs, double lr, int batchSize, Loss loss, int shuffleSeed, bool logMetrics, int metricsEvery){
    if (layers.empty()) {
        throw std::invalid_argument("NeuralNetwork::train invalid state: layers=0");
    }

    if (epochs <= 0) {
        throw std::invalid_argument("NeuralNetwork::train epochs must be > 0");
    }
    if (lr <= 0.0) {
        throw std::invalid_argument("NeuralNetwork::train learning rate must be > 0");
    }
    if (logMetrics && metricsEvery <= 0) {
        throw std::invalid_argument("NeuralNetwork::train metricsEvery must be > 0 when logMetrics is enabled");
    }

    int expectedInputs = layers.front().getWeights().getRows();
    int expectedOutputs = layers.back().getOutputSize();

    if (X.getCols() != expectedInputs) {
        throw std::invalid_argument("NeuralNetwork::train input dimension mismatch: " + std::to_string(X.getRows()) + "x" + std::to_string(X.getCols()) + " | expected Nx" + std::to_string(expectedInputs));
    }

    int n = X.getRows();

    if (n <= 0) {
        throw std::invalid_argument("NeuralNetwork::train empty dataset");
    }

    if (Y.getRows() != n || Y.getCols() != expectedOutputs) {
        throw std::invalid_argument("NeuralNetwork::train label dimension mismatch: " + std::to_string(Y.getRows()) + "x" + std::to_string(Y.getCols()) + " | expected " + std::to_string(n) + "x" + std::to_string(expectedOutputs));
    }

    int effectiveBatchSize = TrainingUtils::resolveEffectiveBatchSize(batchSize, n, "NeuralNetwork::train");

    Loss actualLoss = loss;
    if (loss == AUTO_LOSS) {
        Activation lastAct = layers.back().getAct();
        if (lastAct == SOFTMAX) {
            actualLoss = CROSS_ENTROPY;
        }
        else {
            actualLoss = MSE;
        }
    }

    Activation outputActivation = layers.back().getAct();
    if (actualLoss == CROSS_ENTROPY && !supportsCanonicalCrossEntropyGradient(outputActivation)) {
        throw std::invalid_argument("NeuralNetwork::train CROSS_ENTROPY requires or SOFTMAX output activation");
    }

    std::mt19937 gen = TrainingUtils::createShuffleGenerator(shuffleSeed, "NeuralNetwork::train");
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

            Matrix pred = forward(xBatch);

            Matrix grad;
            if (actualLoss == CROSS_ENTROPY) {
                grad = (pred - yBatch) * (1.0 / batchN);
            }
            else {
                grad = (pred - yBatch) * (2.0 / (batchN * pred.getCols()));
            }

            bool canonicalOutputGrad = (actualLoss == CROSS_ENTROPY);
            backward(grad, lr, canonicalOutputGrad);
        }

        bool shouldLog = logMetrics && (((e + 1) % metricsEvery == 0) || (e == epochs - 1));
        if (shouldLog) {
            Matrix epochPred = forward(X);

            if (actualLoss == CROSS_ENTROPY) {
                double ce = Metrics::crossEntropy(Y, epochPred);
                double acc = Metrics::accuracy(Y, epochPred);
                std::cout << "[NeuralNetwork] epoch " << (e + 1) << "/" << epochs
                        << " ce=" << ce << " acc=" << acc << "\n";
            }
            else {
                double mse = Metrics::mse(Y, epochPred);
                double mae = Metrics::mae(Y, epochPred);
                double rmse = Metrics::rmse(Y, epochPred);
                double r2 = Metrics::r2Score(Y, epochPred);
                std::cout << "[NeuralNetwork] epoch " << (e + 1) << "/" << epochs
                        << " mse=" << mse << " mae=" << mae
                        << " rmse=" << rmse << " r2=" << r2 << "\n";
            }
        }
    }
}

Matrix NeuralNetwork::predict(const Matrix& x){
    if (layers.empty()) {
        throw std::invalid_argument("NeuralNetwork::predict invalid state: layers=0");
    }

    int expectedInputs = layers.front().getWeights().getRows();

    if (x.getCols() != expectedInputs) {
        throw std::invalid_argument("NeuralNetwork::predict input dimension mismatch: " + std::to_string(x.getRows()) + "x" + std::to_string(x.getCols()) + " | expected Nx" + std::to_string(expectedInputs));
    }

    return forward(x);
}

void NeuralNetwork::save(const std::string& path){
    std::ofstream file(path, std::ios::binary);
    if(!file.is_open()){
        throw std::invalid_argument("NeuralNetwork::save open file error: " + path);
    }

    int numLayers = layers.size();
    file.write((char*)&numLayers, sizeof(int));
    if(!file){
        throw std::runtime_error("NeuralNetwork::save write error: header");
    }
    for (const Layer& l : layers) {
        int inputs = l.getWeights().getRows();
        file.write((const char*)&inputs, sizeof(int));

        int outputs = l.getWeights().getCols();
        file.write((const char*)&outputs, sizeof(int));

        Activation act = l.getAct();
        file.write((const char*)&act, sizeof(int));

        Inicialization init = l.getInit();
        file.write((const char*)&init, sizeof(int));

        Matrix weights = l.getWeights();
        const std::vector<double>& wData = weights.getData();
        file.write((const char*)wData.data(), wData.size() * sizeof(double));

        Matrix bias = l.getBias();
        const std::vector<double>& bData = bias.getData();
        file.write((const char*)bData.data(), bData.size() * sizeof(double));
    }

    file.close();
}

void NeuralNetwork::load(const std::string& path){
    std::ifstream file(path, std::ios::binary);
    if(!file.is_open()){
        throw std::invalid_argument("NeuralNetwork::load open file error: " + path);
    }

    int numLayers;
    file.read((char*)&numLayers, sizeof(int));

    layers.clear();

    for(int i = 0; i<numLayers; i++){
        int inputs, outputs;
        Activation act;
        Inicialization init;

        file.read((char*)&inputs, sizeof(int));
    
        file.read((char*)&outputs, sizeof(int));
        
        if(inputs <= 0 || outputs <= 0){
            throw std::runtime_error("NeuralNetwork::load dimension mismatch: " + std::to_string(inputs) + "x" + std::to_string(outputs) + " | expected rows>0 and cols>0");
        }

        file.read((char*)&act, sizeof(int));

        file.read((char*)&init, sizeof(int));

        addLayer(inputs, outputs, act, init);

        Matrix w(inputs, outputs);
        std::vector<double> wData(inputs * outputs);
        file.read((char*)wData.data(), wData.size() * sizeof(double));
        w.setData(wData);

        Matrix b(1, outputs);
        std::vector<double> bData(outputs);
        file.read((char*)bData.data(), bData.size() * sizeof(double));
        b.setData(bData);

        layers.back().setWeights(w);
        layers.back().setBias(b);
    }

    file.close();
}