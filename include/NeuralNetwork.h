#ifndef _NEURAL_NETWORK_H_
#define _NEURAL_NETWORK_H_

#include<vector>
#include<string>
#include"Matrix.h"
#include"Layer.h"

enum Loss { MSE, CROSS_ENTROPY, AUTO_LOSS };

struct Dense {
    int outputs;
    Activation act;
    Inicialization init;

    Dense(int outputs, Activation act = NONE, Inicialization init = AUTO): outputs(outputs), act(act), init(init) {}
};

class NeuralNetwork{
    private:
        std::vector<Layer> layers;
        int inputFeatures;
        bool hasCompileConfig;
        Loss compiledLoss;
        double compiledLr;
        int compiledBatchSize;
        int compiledShuffleSeed;
        bool compiledLogMetrics;
        int compiledMetricsEvery;
        Matrix forward(const Matrix& x);
        void backward(const Matrix& grad, double lr, bool canonicalOutputGrad = false);
        void addLayer(int inputs, int outputs, Activation act, Inicialization init);
        void train(const Matrix& X, const Matrix& Y, int epochs, double lr, int batchSize, Loss loss, int shuffleSeed, bool logMetrics, int metricsEvery);
        static bool supportsCanonicalCrossEntropyGradient(Activation act) { return act == SOFTMAX; }
    public:
        NeuralNetwork();
        NeuralNetwork& input(int features);
        NeuralNetwork& add(const Dense& layer);
        void compile(Loss loss = AUTO_LOSS, double lr = 0.01, int batchSize = 0, int shuffleSeed = -1, bool logMetrics = false, int metricsEvery = 1);
        void fit(const Matrix& X, const Matrix& Y, int epochs);
        Matrix predict(const Matrix& x);
        void save(const std::string& path);
        void load(const std::string& path);
};

#endif