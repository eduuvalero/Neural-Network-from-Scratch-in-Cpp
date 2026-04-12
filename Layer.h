#ifndef _LAYER_H_
#define _LAYER_H_

#include "Matrix.h"

enum Activation { RELU, LEAKY_RELU, SIGMOID, TANH, SOFTMAX, NONE };

class Layer {
    private:
        Matrix W;
        Matrix b;
        Matrix Z;
        Matrix X;
        Activation act;
    public:
        Layer(int inputs, int outputs, Activation act);
        Matrix forward(const Matrix& x);
        Matrix backward(const Matrix& grad, double lr, bool applyActivationDerivative = true);
        // Basic getters
        Activation getAct() const;
        Matrix getWeights() const;
        Matrix getBias() const;
        // Basic setters
        void setWeights(const Matrix& w);
        void setBias(const Matrix& b);
};

#endif