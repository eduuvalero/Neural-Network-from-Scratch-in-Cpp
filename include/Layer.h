#ifndef _LAYER_H_
#define _LAYER_H_

#include "Matrix.h"

enum Activation { RELU, LEAKY_RELU, SIGMOID, TANH, SOFTMAX, NONE };

class Layer {
    private:
        Matrix W;
        Matrix b;
        Matrix Z;
        Matrix A;
        Matrix X;
        Activation act;
        Matrix applyActivation(const Matrix& z) const;
        Matrix applyActivationGradient(const Matrix& grad) const;
        Matrix applySoftmaxJacobian(const Matrix& grad) const;
        static double stableSigmoid(double x);
    public:
        Layer(int inputs=1, int outputs=1, Activation act=NONE);
        Matrix forward(const Matrix& x);
        Matrix backward(const Matrix& grad, double lr, bool applyActivationDerivative = true);
        // Basic getters
        Activation getAct() const { return act; }
        Matrix getWeights() const { return W; }
        Matrix getBias() const { return b; }
        int getOutputSize() const { return b.getCols(); }
        // Basic setters
        void setWeights(const Matrix& w);
        void setBias(const Matrix& b);
};

#endif