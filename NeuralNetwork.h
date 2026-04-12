#ifndef _NEURAL_NETWORK_H_
#define _NEURAL_NETWORK_H_

#include<vector>
#include<string>
#include"Matrix.h"
#include"Layer.h"

enum Loss { MSE, CROSS_ENTROPY, AUTO };

class NeuralNetwork{
    private:
        std::vector<Layer> layers;
    public:
        void addLayer(int inputs, int outputs, Activation act = NONE);
        Matrix forward(const Matrix& x);
        void backward(const Matrix& grad, double lr, bool canonicalOutputGrad = false);
        void train(const Matrix& X, const Matrix& Y, int epochs, double lr, Loss loss = AUTO);
        void save(const std::string& file);
        void load(const std::string& file);
};

#endif