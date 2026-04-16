#ifndef _NEURAL_NETWORK_H_
#define _NEURAL_NETWORK_H_

#include<vector>
#include<string>
#include"Matrix.h"
#include"Layer.h"

enum Loss { MSE, CROSS_ENTROPY, AUTO_LOSS };

class NeuralNetwork{
    private:
        static bool supportsCanonicalCrossEntropyGradient(Activation act) { return act == SOFTMAX || act == SIGMOID; }
        std::vector<Layer> layers;
        Matrix forward(const Matrix& x);
        void backward(const Matrix& grad, double lr, bool canonicalOutputGrad = false);
    public:
        void addLayer(int inputs, int outputs, Activation act = NONE);
        void train(const Matrix& X, const Matrix& Y, int epochs, double lr, int batchSize = 0, Loss loss = AUTO_LOSS, int shuffleSeed = -1, bool logMetrics = false, int metricsEvery = 1);
        Matrix predict(const Matrix& x);
        void save(const std::string& path);
        void load(const std::string& path);
};

#endif