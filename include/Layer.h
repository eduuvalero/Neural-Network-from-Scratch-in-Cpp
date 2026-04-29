#ifndef _LAYER_H_
#define _LAYER_H_

#include "Matrix.h"
#include "Random.h"

class Optimizer;

enum Activation { RELU, LEAKY_RELU, SIGMOID, TANH, SOFTMAX, NONE };

enum Inicialization { HE, XAVIER, AUTO };

class Layer {
    private:
        Matrix W;
        Matrix b;
        Matrix Z;
        Matrix A;
        Matrix X;
        Matrix dropoutMask;
        Activation act;
        Inicialization init;
        double dropoutRate;
        Random rand;
        void initWeights();
        void initHe();
        void initXavier();
        Matrix applyActivation(const Matrix& z) const;
        Matrix applyActivationGradient(const Matrix& grad) const;
        Matrix applySoftmaxJacobian(const Matrix& grad) const;
        static double stableSigmoid(double x);


    public:
        Layer(int inputs=1, int outputs=1, Activation act=NONE, Inicialization init= AUTO, double dropoutRate=0.0);
        Matrix forward(const Matrix& x, bool training);
        Matrix backward(const Matrix& grad, Optimizer& optimizer, int layerIndex, bool applyActivationDerivative = true);
        // Basic getters
        Activation getAct() const { return act; }
        Inicialization getInit() const { return init; }
        Matrix getWeights() const { return W; }
        Matrix getBias() const { return b; }
        int getOutputSize() const { return b.getCols(); }
        double getDropoutRate() const { return dropoutRate; }
        // Basic setters
        void setWeights(const Matrix& w);
        void setBias(const Matrix& b);
};

#endif